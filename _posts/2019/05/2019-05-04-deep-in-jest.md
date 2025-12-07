---
title: "Jest进阶：接入ts、集成测试与覆盖率统计"
date: "2019-05-04"
permalink: /2019-05-04-deep-in-jest/
---

## 接入 TypeScript

在给 vemojs 做完各种测试之后，导师很快提出了新的要求，给 [clousebase-cli](https://github.com/TencentCloudBase/cloud-base-cli)  编写测试用例。有个问题摆在眼前：它是用 typescript 编写，所以需要配置相关环境。

好吧，不说废话了，直接上干货。

`jest.config.js`  配置内容如下，解释在注释里面：

```javascript
module.exports = {
  roots: [
    "<rootDir>/test" // 测试目录
  ],
  transform: {
    "^.+\\.tsx?$": "ts-jest" // 匹配 .ts 或者 .tsx 结尾的文件
  },
  collectCoverage: true, // 统计覆盖率
  testEnvironment: "node", // 测试环境
  setupFilesAfterEnv: [
    "<rootDir>/jest.setup.js" // 之后再说
  ],
  // 不算入覆盖率的文件夹
  coveragePathIgnorePatterns: [
    "<rootDir>/node_modules/",
    "<rootDir>/test/",
    "<rootDir>/deps/"
  ]
};
```

上面有个 `jest.setup.js` ，它的内容是： `jest.setTimeout(60000)` 。因为有时候网速很慢，api 请求延时会很高，所以这个就是设置请求超时时间为 1 分钟。

最坑的一点是，除了 `jest`  的配置文件，还要修改 typescript 对应的文件， `tsconfig.json`  内容如下。types 中必须添加 `jest` ，否则找不到 `expect` 、 `describe`  等变量的定义。

```json
{
  "compilerOptions": {
    "types": ["node", "jest"]
  }
}
```

**总之，cloudbase-cli 的测试用例写的比 vemo 好，哈哈**

## 集成测试

持续继承测试我们借助  [https://travis-ci.org/](https://travis-ci.org/)  这个平台，它的工作流程非常简单：

1. 在它平台上授权 github 仓库的权限，github 仓库下配置  .travis.yml 文件
1. 每次 commit 推上新代码的时候，travis-ci 平台都会接收到通知
1. 读取 .travis.yml 文件，**然后创建一个虚拟环境**，来跑配置好的脚本（比如启动测试脚本）

它的优点在于，测试代码推上去后，直接在账号下的控制台就能看到测试结果，非常方便；而且可以在配置文件中，指明多个测试环境，比如 node 有 6、8、10，让测试更具有信服力。

我把样例代码放在了 [try-travis-ci](https://github.com/dongyuanxin/try-travis-ci)  仓库下，可以跑一下看看。下面是 .travis.yml 文件内容。

```yaml
sudo: false
language: "node_js"
node_js:
  - "8"
  - "10"
install:
  - npm install
script: npm run test
```

看见了吗，就是下面贼酷炫的界面，登陆用户就能看到了：<br />![image.png](https://cdn.nlark.com/yuque/0/2019/png/233327/1556960049220-d204d334-21fb-4963-9095-e37d1600ac4b.png#align=left&display=inline&height=784&name=image.png&originHeight=980&originWidth=1908&size=147774&status=done&width=1526.4)

## 覆盖率统计

覆盖率统计也很简单（本来以为会很难），但是要安装 `coveralls`  这个库。除此之外，还要修改一下 package.json 中的 scripts 的指令。通过管道，将结果交给 coveralls。

```json
{
  "scripts": {
    "test": "jest --passWithNoTests --coverage --env=node --detectOpenHandles --coverageReporters=text-lcov | coveralls"
  }
}
```

后来发现，在统计覆盖率的时候，会把覆盖的信息放在根目录下的 `coverage`  文件夹下，这些信息都是多个平台约定好的数据格式。所以各个工具间可以共同使用。

剩下要做的就是，登陆 coveralls.io 平台，授权 github 仓库权限。当你在 travis 平台运行上述 scripts 脚本时候，它就自动把结果扔到了 coveralls.io 平台。登陆账号，就能看到覆盖率了。

## 参考资料

- 《持续集成服务 Travis CI 教程》：[http://www.ruanyifeng.com/blog/2017/12/travis_ci_tutorial.html?20190430165111](http://www.ruanyifeng.com/blog/2017/12/travis_ci_tutorial.html?20190430165111)
- Travis CI Document：[https://docs.travis-ci.com/user/languages/javascript-with-nodejs/#stq=&stp=0](https://docs.travis-ci.com/user/languages/javascript-with-nodejs/#stq=&stp=0)
- Coveralls IO JavaScript Document：[https://docs.coveralls.io/javascript](https://docs.coveralls.io/javascript)
- 第三方库 node-coveralls：[https://github.com/nickmerwin/node-coveralls](https://github.com/nickmerwin/node-coveralls)
