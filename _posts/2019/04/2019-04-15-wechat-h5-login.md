---
title: "微信网页登录逻辑与实现"
date: "2019-04-15"
permalink: /2019-04-15-wechat-h5-login/
categories: ["C工作实践分享"]
---

现在的网站开发，都绕不开微信登录（毕竟微信已经成为国民工具）。虽然文档已经写得很详细，但是对于没有经验的开发者还是容易踩坑。

所以，专门记录一下微信网页认证的交互逻辑，也方便自己日后回查：

1. 加载微信网页 sdk
1. 绘制登陆二维码：新 tab 页面绘制 / 本页面 iframe 绘制
1. 用户扫码登陆，前端跳入回调网址
1. 回调网址进一步做逻辑处理，如果是页内 iframe 绘制二维码，需要通知顶级页

### 微信网页 SDK 加载

在多人团队协作中，加载资源的代码需要格外小心。因为可能会有多个开发者在同一业务逻辑下调用，这会造成资源的重复加载。

处理方法有两种，第一种是对外暴露多余接口，专门 check 是否重复加载。但是考虑到调用者每次在加载前，都需要显式调用`check()`方法进行检查，难免会有遗漏。

所以采用第二种方法--设计模式中的缓存模式，代码如下：

```javascript
// 备忘录模式: 防止重复加载
export const loadWeChatJs = (() => {
  let exists = false; // 打点
  const src = "//res.wx.qq.com/connect/zh_CN/htmledition/js/wxLogin.js"; // 微信sdk网址

  return () =>
    new Promise((resolve, reject) => {
      // 防止重复加载
      if (exists) return resolve(window.WxLogin);

      let script = document.createElement("script");
      script.src = src;
      script.type = "text/javascript";
      script.onerror = reject; // TODO: 失败时候, 可以移除script标签
      script.onload = () => {
        exists = true;
        resolve(window.WxLogin);
      };
      document.body.appendChild(script);
    });
})();
```

### 绘制登陆二维码

根据[《微信登陆开发指南》](https://open.weixin.qq.com/cgi-bin/showdocument?action=dir_list&t=resource/res_list&verify=1&id=open1419316505&token=&lang=zh_CN)，将参数传递给`window.WxLogin()`即可。

```javascript
// 微信默认配置
const baseOption = {
  self_redirect: true, // true: 页内iframe跳转; false: 新标签页打开
  id: "wechat-container",
  appid: "wechat-appid",
  scope: "snsapi_login",
  redirect_uri: encodeURIComponent("//1.1.1.1/"),
  state: ""
};

export const loadQRCode = (option, intl = false, width, height) => {
  const _option = { ...baseOption, ...option };

  return new Promise((resolve, reject) => {
    try {
      window.WxLogin(_option);
      const ele = document.getElementById(_option["id"]);
      const iframe = ele.querySelector("iframe");
      iframe.width = width ? width : "300";
      iframe.height = height ? height : "420";
      // 处理国际化
      intl && (iframe.src = iframe.src + "&lang=en");
      resolve(true);
    } catch (error) {
      reject(error);
    }
  });
};
```

在需要使用的业务组件中，可以在周期函数`componentDidMount`调用，下面是 demo 代码：

```javascript
componentDidMount() {
    const wxOption = {
        // ...
    };
	loadWeChatJs()
		.then(WxLogin => loadQRCode(wxOption))
		.catch(error => console.log(`Error: ${error.message}`));
}
```

### 回调网址与 iframe 通信

这一块我觉得是微信登陆交互中最复杂和难以理解的一段逻辑。开头有讲过，微信二维码渲染有 2 中方式，一种是打开新的标签页，另一种是在指定 id 的容器中插入 iframe。

毫无疑问，第二种交互方式更友好，因为要涉及不同级层的页面通信，代码处理也更具挑战。

为了方便说明，请先看模拟的数据配置：

```javascript
// redirect 地址会被后端拿到, 后端重定向到此地址, 前端会访问此页面
// redirect 地址中的参数, 是前端人员留给自己使用的; 后端会根据业务需要, 添加更多的字段, 然后一起返回前端
const querystr =
  "?" +
  stringify({
    redirect: encodeURIComponent(
      `${window.location.origin}/account/redirect?` +
        stringify({
          to: encodeURIComponent(window.location.origin),
          origin: encodeURIComponent(window.location.origin),
          state: "login"
        })
    ),
    type: "login"
  });

const wxOption = {
  id: "wechat-container",
  self_redirect: true,
  redirect_uri: encodeURIComponent(
    `//1.1.1.1/api/socials/weixin/authorizations${querystr}`
  ) // 微信回调请求地址
};
```

### 前后端、微信服务器、用户端交互逻辑

按照上面的配置，我描述一下前端、用户端、微信服务器和后端交互的逻辑：

1. 前端根据 wxOption 加载了二维码，所有信息都放在了二维码中。同时监听微信服务器的消息。
1. 用户手机扫码，通知微信服务器确定登陆。
1. 微信服务器接受到用户的扫码请求，转发给前端。
1. 前端收到微信服务器传来消息，根据 wxOption 的 redirect_uri 参数，跳转到此 url 地址。注意：

- 这个接口地址是后端的，请求方式是 GET
- 前端通过拼接 params 携带参数
- 地址会被拼接微信服务器传来的一个临时 token，用于交给后端换取用户公众密钥

5. 后端接收到`/api/socials/weixin/authorizations${querystr}`的请求，decode 解码 querystr 中的信息。然后向微信服务端请求用户公众密钥。根绝前后端的约定（demo 中用的是 redirect 字段），重定向到前端指定的 redirect 字段，并且拼接用户公众密钥等更多信息。
6. 前端知悉重定向，跳到重定向的路由（demo 中用的是/account/redirect）
7. 在对应的路由处理后端传来的用户密钥等数据即可
8. 至此，微信认证的四端交互逻辑完成

### 跨 Iframe 通信

前面流程走完了，现在的情况是页面中 iframe 的二维码区域，已经被替换成了`/account/redirect?...`的内容。

为了实现通信，需要在页面的周期中监听`message`事件，并在组件卸载时，卸载此事件：

```javascript
componentDidMount() {
  // ... ...

  window.addEventListener('message', this.msgReceive, false);
}

componentWillUnmount() {
  window.removeEventListener('message', this.msgReceive);
}

msgReceive(event) {
  // 监测是否是安全iframe
  if(!event.isTrusted) {
    return;
  }
  console.log(event.data); // 获取iframe中传来的数据, 进一步进行逻辑处理
}
```

而在`/account/redirect?...`路由对应的组件中，我们需要解析路由中的 params 参数，按照业务逻辑检查后，将结果传递给前面的页面：

```javascript
componentDidMount() {
    // step1: 获取url中params参数
    const querys = getQueryVariable(this.props.location.search);
    // step2: 检查querys中的数据是否符合要求 ...
    // step3: 向顶级页面传递消息
    return window.parent && window.parent.postMessage('data', '*');
}
```

至此，微信网页认证的流程完成。

_更多：关于 iframe 通信的更多细节，请查看 MDN 的文档_
