---
title: Package Management - NPM
date: 2020-07-16 11:11:11 -0400
categories: [30System, Basic]
tags: [Linux, Sysadmin]
math: true
image:
---

- [Package Management - NPM](#package-management---npm)
  - [package.json](#packagejson)
  - [command](#command)
    - [npm init](#npm-init)
    - [npm install](#npm-install)
      - [install globally or locally](#install-globally-or-locally)

---

# Package Management - NPM

Using npm effectively is a cornerstone of modern web development, no matter if it's exclusively with Node.js, as a package manager or build tool for the front-end, or even as a piece of workflows in other languages and on other platforms.

---

## package.json

- As a general rule, any project that's using Node.js will need to have a `package.json` file.
- a manifest of your project that includes the packages and applications it depends on, information about its unique source control, and specific metadata like the project's name, description, and author.


```json
{

  "name": "metaverse", // The name of your project
  "version": "0.92.12", // The version of your project
  "description": "The Metaverse virtual reality. The final outcome of all virtual worlds, augmented reality, and the Internet.", // The description of your project
  "main": "index.js",
  "license": "MIT", // The license of your project


  "devDependencies": {
    "mocha": "~3.1",
    "native-hello-world": "^1.0.0",
    "should": "~3.3",
    "sinon": "~1.9"
  },
  "dependencies": {
    "fill-keys": "^1.0.2",
    "module-not-found-error": "^1.0.0",
    "resolve": "~1.1.7"
  }
}
```

Having dependencies in your project's package.json allows the project to install the versions of the modules it depends on.
- By running `npm install` inside of a project, you can install all of the dependencies that are listed in the project's `package.json` - meaning they don't have to be (and almost never should be) bundled with the project itself.

---

## command

---

### npm init

```bash
npm init # This will trigger the initialization
npm init --yes # This will trigger automatically populated initialization.
```

- Once you run through the npm init steps above, a package.json file will be generated and placed in the current directory.
- If you run it in a directory that's not exclusively for your project, don't worry! Generating a package.json doesn't really do anything, other than create a package.json file.
- You can either move the package.json file to a directory that's dedicated to your project, or you can create an entirely new one in such a directory.

---

### npm install

- Installing modules from npm is one of the most basic things you should learn to do when getting started with npm.
- install a standalone module into the current directory:

```bash
# install the express module into /node_modules in the current directory
npm install <module>
npm i <module>
npm install <module> --save

# trigger the installation of all modules that are
# listed as dependencies and devDependencies in the package.json in the current directory.
npm install


```

- add the optional flag `--save`: add the module as a dependency of your project to the project's package.json as an entry in dependencies.

- add the optional flag `--save-dev`.
  - instead of saving the module being installed and added to package.json as an entry in dependencies, it will save it as an entry in the devDependencies.
  - The semantic difference here is that dependencies are for use in production - whatever that would entail for your project.
  - On the other hand, devDependencies are a collection of the dependencies that are used in development of your application - the modules that you use to build it, but don't need to use when it's running.
  - This could include things like testing tools, a local server to speed up your development, and more.




#### install globally or locally

- locally
  - Installing it locally means the module will be available only for a specific project (the directory you were in when you ran npm install), as it installs to the local node_modules folder.

- global
  - A global install will instead put the module into your global package folder (OS dependent), and allows you to run the included executable commands from anywhere. Note that by default you can only require local packages in your code.

- Generally speaking, you should install most modules locally, unless they provide a CLI command that you want to use anywhere.



.
