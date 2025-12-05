---
title: "NodeJS开发交互式命令行工具"
url: "2022-04-11-nodejs-shell"
date: 2022-04-11
---

## 颜色控制


### 实现原理


这个比较简单，写过 c 的应该知道控制命令行颜色，只需要 颜色宏定义 + 字体内容 拼接即可。这就是 chalk.js 的实现原理。


举个例子，对于`\\e[31;44;4;1m`来说，意义如下：

- `\\e` 代表开始 `ANSI Escape code`。就是ESC的转义符。在nodejs中，请用它的16进制码`0x1B`
- `[` 代表转义序列开始符 CSI，Control Sequence Introducer
- `31;44;4;1` 代表以; 分隔的文本样式控制符，其中 31 代表文本前景色为红色，44代表背景为蓝色，4代表下划线，1代表加粗。
- `m` 代表结束控制符序列

在shell中，直接输入：`echo -e "\\e[37;44;4;1mLEO\\e[0m"`；或者在nodejs中，输入：


```javascript
process.stdout.write('\\x1B[37;44;4;1mLEO\\x1B[0m')
```


### chalk.js 使用


相比于直接去拼写颜色宏定义，它提供更语义化的 api 将文本处理成拼接后的结果，然后交给控制台输出。


```typescript
import chalk from 'chalk'
const print = console.log;

print(chalk.blue("Hello") + " World" + chalk.red("!"));
print(chalk.blue.bgRed.bold("Hello World!"));
```


![e6c9d24egy1h167fnjba1j20o202kglu.jpg](https://raw.githubusercontent.com/dongyuanxin/static/main/blog/imgs/2022-04-11-nodejs-shell/e6c9d24egy1h167fnjba1j20o202kglu.jpg)


## 进度控制


### 实现原理


「回车」不等于「换行」：

- `\\r` 回车，回到当前行的行首，而不会换到下一行，如果接着输出的话，本行以前的内容会被逐一覆盖
- `\\n` 换行，换到当前位置的下一行，而不会回到行首

所以命令行的进度条，就是定时输出，每次输出的时候，最前面都带上`\\r`。这样就达到了清空当前行输出的目的。


这个就是 ora.js 的原理。


### ora.js 使用


```javascript
import ora from 'ora';
const spinner = ora({
    text: "链接网络中"
}).start(); // 开始状态 => 加载状态
setTimeout(() => {
    spinner.color = "yellow";
    spinner.text = "网速有点慢";
}, 1000); // 还是 加载状态, 更新文案和颜色
setTimeout(() => {
    spinner.succeed("下载成功"); // 加载状态 => 成功状态
}, 2000);
```


## 参数读取


### 实现原理


通过 `process.argv` 可以读取命令行参数。默认第一个是执行脚本，第二个是当前JS脚本文件的绝对路径。


比如 chalk.js 实现如下：


```javascript
process.argv.forEach((val, index) => {
    console.log(index, val)
})
```


通过命令行执行 `node chalk.js -L abc --list=abc list=abc`，输出如下：


```shell
0 /usr/local/bin/node
1 /Users/0x98k/work/outside/stack-hugo/tmp/2022-04-11-nodejs-shell/chalk.js
2 -L
3 abc
4 --list=abc
5 list=abc
```


先读取命令后面的参数，然后依次解析，就是 commander.js 的实现原理。


### commander.js 使用


方法一：注入 Options 以及对应的执行逻辑


```typescript
import program from 'commander'
program
    .version("0.0.1")
    .option("-t, --types [type]", "test options")
    // option这句话必须加
    .parse(process.argv);

```


方法二：注入 Command 以及对应的执行逻辑


```javascript
import program from 'commander'
// Commands 操作
program
    // 命令与参数: <> 必填; [] 选填
    .command("exec <cmd> [env]")
    // 别名
    .alias("ex")
    // 帮助信息
    .description("execute the given remote cmd")
    // 没用，option和command是冲突的
    // .option("-e, --exec_mode <mode>", "Which exec mode to use")
    // 执行的操作
    .action((cmd, env, options) => {
        // 参数可以拿到
        console.log(`env is ${env}`);
        console.log('exec "%s" using %s mode', cmd, options.exec_mode);
    })
    // 自定义help信息
    .on("--help", function() {
        console.log("自定义help信息");
    });
// 参数长度不够, 打印帮助信息
if (!process.argv.slice(2).length) {
    program.outputHelp();
}
// 解析命令行参数
program.parse(process.argv);

```


## 交互验证


### 实现原理

1. 利用前面的 chalk.js 实现不同样式字体的输出。
2. 监听用户键盘输入，比如上/下/左/右
3. 利用 ora.js 的原理，可以在用户输入上下左右的时候，重新刷新输出，从而实现命令行下的列表上下选择。

### inquirer.js 使用


可以快速地实现选择列表、输入框等逻辑。


```javascript
import inquirer from 'inquirer'

t
// type: input
promps.push({
    type: "input",
    name: "moduleName",
    message: "请输入模块名称",
    validate: function(input) {
        if (!input) {
            return "输入不能为空";
        }
        return true;
    }
});
// type: list
promps.push({
    type: "list",
    name: "cssPretreatment",
    message: "想用什么css预处理器呢",
    choices: [
        {
            name: "Sass",
            value: "sass"
        },
        {
            name: "Less",
            value: "less"
        }
    ]
});

inquirer.prompt(promps).then(function(answers) {
    console.log("answers are", answers);
});

```


交互效果：


![e6c9d24egy1h17w15axj1j20pc03g3z2.jpg](https://raw.githubusercontent.com/dongyuanxin/static/main/blog/imgs/2022-04-11-nodejs-shell/e6c9d24egy1h17w15axj1j20pc03g3z2.jpg)


## 并行命令


为了利用多核处理器，经常需要在脚本中并行启动多条shell命令/nodejs脚本。


### [**concurrently.js**](https://www.npmjs.com/package/concurrently)

- 基于JavaScript编写，可以作为SDK在代码中使用，也可以直接作为shell命令使用
- 利用多核处理器，并行执行命令
- 多条命令执行输出不会混乱

```javascript
const inquirer = require('inquirer');
const chalk = require('chalk');
const concurrently = require('concurrently');
const print = console.log;

inquirer.prompt(prompts).then(answers => {
  const choosenApps = answers?.choosenApps || [];
  if (!choosenApps?.length) {
    print(chalk.bold.red('Choose at least one app'));
    return;
  }
  // 输出并发启动的信息
  let output = '';
  output += chalk.bold('Start apps concurrently:');
  choosenApps.forEach(app => {
    output += ` ${chalk.greenBright(APP_NAME[app])},`;
  });
  print(output.slice(0, output.length - 1));
  // 并发启动
  const commands = choosenApps.map(app => ({
    name: APP_NAME[app], // 进程名字。例如：app1
    command: APP_START_CMD[app], // 启动脚本。例如：pnpm run start --filter app1
  }));
  concurrently(commands);
});
```


启动后的效果（output都跟在对应进程名字后）：


![Untitled.png](https://raw.githubusercontent.com/dongyuanxin/static/main/blog/imgs/2022-04-11-nodejs-shell/b468b0640be116c9adfabcb5ff0131d8.png)


### [TurboRepo](https://turbo.build/repo/docs/reference/configuration#dependson)

- 基于Golang编写，性能更高，专门用于monorepo的编译解决方案
- 支持npm、yarn、pnpm等包管理工具
- 利用多核处理器，并行执行命令
- 根据大仓的包依赖关系智能生成「依赖拓扑结构」，按照用户配置，智能编排脚本，保证顺序不出错。

假设apps/docs和apps/web的构建（build）均依赖packages/shared，如果直接自己编写shell脚本，顺序和耗时如下：


![Untitled.png](https://raw.githubusercontent.com/dongyuanxin/static/main/blog/imgs/2022-04-11-nodejs-shell/5a9ea0b3cf392f1cd08d378db458fcab.png)


如果使用turborepo进行配置后，效果如下：

- lint 和 test 命令没有依赖关系，可以直接并行执行
- build命令，apps/web和apps/docs依赖packages/shared，因此先执行packages/shared的build，再并行执行apps/web和apps/docs

![Untitled.png](https://raw.githubusercontent.com/dongyuanxin/static/main/blog/imgs/2022-04-11-nodejs-shell/1e1b6e20f45f1f6afa6d9aacc05a4550.png)


turborepo配置如下：


```javascript
{
  "$schema": "https://turbo.build/schema.json",
  "pipeline": {
    "build": {
      // ^build means build must be run in dependencies
      // before it can be run in this workspace
			// 翻译：^build  检查依赖包的build命令是否执行完成
      "outputs": [".next/**", "!.next/cache/**",".svelte-kit/**"],
      "dependsOn": ["^build"]
    },
    "test": {},
    "lint": {}
  }
}
```


## 参考链接

- [https://juejin.cn/post/6844904006981173256](https://juejin.cn/post/6844904006981173256)
- [https://learnku.com/articles/15124/the-principle-of-output-progress-bar-on-the-command-line](https://learnku.com/articles/15124/the-principle-of-output-progress-bar-on-the-command-line)
- [https://www.zhihu.com/question/505956571/answer/2290454921](https://www.zhihu.com/question/505956571/answer/2290454921)

