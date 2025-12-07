---
title: "玩转 Nodejs 命令行"
date: "2019-05-07"
permalink: /2019-05-07-play-node-shell/
categories: ["开发实战", "命令行编程"]
---

## 背景

在做 cli 工具的时候，非常需要命令行相关的第三方库。一个比较稳健成熟的命令行应该考虑以下 4 种需求：

1. 读取传入的各种参数，例如： --help, -v=123
1. 逻辑处理和友好的 UI 交互，例如：提供列表选择
1. 细致控制字体颜色和背景颜色
1. 状态显示，例如：等待过程前面是转圈圈，完成过程前面自动换成对号

代码仓库地址：[play-node-command](https://github.com/dongyuanxin/play-node-command) 。可以直接 clone 到本地，依次跑一下目录下的 4 个以 play 开头的 js 文件，就能直观看到效果。

## 读取参数: commander

这里用到的是 commander 这个库。它的文档地址是：[https://www.npmjs.com/package/commander](https://www.npmjs.com/package/commander)

请先看下面代码：

```javascript
const program = require("commander");

// 分为2种操作, 2种操作互相冲突

// Options 操作
program
  .version("0.0.1")
  .option("-t, --types [type]", "test options")
  // option这句话必须加
  .parse(process.argv);

// Commands 操作
program
  // 命令与参数: <> 必填; [] 选填
  .command("exec <cmd> [env]")
  // 别名
  .alias("ex")
  // 帮助信息
  .description("execute the given remote cmd")
  // 没用，option和command是冲突的
  .option("-e, --exec_mode <mode>", "Which exec mode to use")
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

if (program.types) {
  console.log(program.types);
}

// 解析命令行参数
program.parse(process.argv);
```

文档上基本都写明白了，但是有几个需要注意的点：

1. 它主要提供 options 和 commands 两种操作，option 就是形如“-t，--types”这样的传参，commands 就是形如“exec”这样的传参。 **不要混用两者** 。
1. 读取 commands 中传入的参数，写在 `.action`  中；读取 options 传入的参数，是通过访问 `program`  上的变量。除此之外，**options 操作需要执行  .parse(process.argv) 解析命令行参数**
1. `-V`  和 `-h`  默认也是提供的，但是也可以通过自定义覆盖
1. 一般都把 options 写在前面， **顺便标识版本号** ；把 commands 写在后面；最后会判断一下参数长度，不够会自动输出打印信息

## 交互验证：inquirer

深入交互并且提供基于命令行的选择列表、弹框等 UI 视图，我们借助：inquirer 库。它的文档地址是：[https://www.npmjs.com/package/inquirer](https://www.npmjs.com/package/inquirer)

请看下面这段代码：

```javascript
const inquirer = require("inquirer");
const program = require("commander");

program
  .version("1.0.0")
  .option("--sass [sass]", "启用sass")
  .option("--less", "启用less")
  .parse(process.argv);

program
  .command("module [moduleName]")
  .alias("m")
  .description("创建新模块")
  .action(option => {
    console.log(`option is ${option}`);
    console.log(`program.sass is ${program.sass}`);
    const config = {
      moduleName: null,
      des: "",
      sass: false,
      less: false
    };

    const promps = [];

    // type: input
    // 问答框类型
    if (config.moduleName !== "string") {
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
    }

    // type: list
    // 列表选择器类型
    if (!program.sass && !program.less) {
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
    }

    inquirer.prompt(promps).then(function(answers) {
      console.log(answers);
    });
  });

program.parse(process.argv);
```

除去 commader 库的应用，inquirer 库的应用在 15~64 行。它首先会验证是否传入 module 参数，如果没有，那么以问答的形式引导用户输入；紧接着检查是否指定了 scss / less，如果没有指定，弹出列表选择器供用户选择。

整个过程中的交互体验还是非常好的，尤其是针对多个选项的时候的列表选择器，一目了然。

## 颜色控制：chalk

这个比较简单，写过 c 的同学应该知道控制命令行颜色，只需要 颜色宏定义 + 字体内容 拼接即可。所以这个库也是，提供更语义化的 api 将文本处理成拼接后的结果，然后交给控制台输出。

```javascript
const chalk = require("chalk");
const print = console.log;
print(chalk.blue("Hello") + " World" + chalk.red("!"));
print(chalk.blue.bgRed.bold("Hello World!"));
```

## 过程控制：ora

它实现的核心功能是控制台刷新，我可以用它来做“下载进度条”（一直更新 text 属性即可）。当然，项目中用它来做状态提示，它会在语句前面给个转圈圈的 icon，还会有对号、错误等终止状态 icon。

看下面这段代码，假想现在是在下载**\*。**可以跑一下下面代码，mac 下比 windows 下好太多\*\*。

```javascript
const ora = require("ora");

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
