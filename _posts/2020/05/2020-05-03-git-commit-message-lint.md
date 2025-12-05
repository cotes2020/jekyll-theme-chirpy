---
title: "如何规范commit message记录"
url: "2020-05-03-git-commit-message-lint"
date: 2020-05-03
---

## 概述


随着项目体积的增加，参与到项目中的同学越来越多，每个人都有自己的打 git log 的习惯：

- 格式 1: `add: 添加...`
- 格式 2: `[add]: 添加...`
- 格式 3: `Add 添加...`

为了形成统一的规范，达成共识，从而降低协作开发成本，需要对 git commit 记录进行规范。


## 规范 git commit 记录


规范 git commit 记录，需要做两件事情：

- 通过交互式命令行，自动生成符合指定规范的 commit 记录
- 提交记录后，在 git hooks 中进行 commit 记录格式检查

问：既然已经交互式生成了规范记录，为什么需要在 hooks 进行检查？


交互式生成 commit 记录，需要用户调用自定义的 npm scripts，例如`npm run commit`。但还是可以直接调用原生 git 命令 `git commit` 来提交记录。而检查是在正式提交前进行的，因此不符合要求的记录不会生效，需要重新 commit。


## 调研：交互式 commit log 规范方案


前期调研结果，关于 commit 提示有两种做法：

1. 直接使用 commitizen 中常用的 adapter
2. 根据团队的需要，自定义 adapter

**方法 1 的优缺点：**


优点 1: 直接安装对应的 adapter 即可


优点 2: 无开发成本


缺点 1: 无法定制，不一定满足团队需要


**方法 2 的优缺点：**


优点 1: 可定制，满足开发需求


优点 2: 单独成库，发布 tnpm，作为技术建设


缺点 1: 需要单独一个仓库（但开发成本不高）


## 代码实现


在实际工作中，发现方法 1 中的常用规范，足够覆盖团队日常开发场景。所以，选择了方法 1.


step1: 安装 npm 包


```shell
npm i --save-dev commitizen cz-conventional-changelog @commitlint/cli @commitlint/config-conventional husky
```


添加 package.json 的配置：


```json
"scripts": {
    "commit": "git-cz"
},
"husky": {
    "hooks": {
        "commit-msg": "commitlint -E HUSKY_GIT_PARAMS"
    }
},
"config": {
    "commitizen": {
      "path": "./node_modules/cz-conventional-changelog"
    }
}
```


在项目根目录下创建`commitlint.config.js`：


```javascript
module.exports = {
    extends: ["@commitlint/config-conventional"]
};
```


使用方法：不再使用`git commit -m ...`，而是调用`npm run commit`。


