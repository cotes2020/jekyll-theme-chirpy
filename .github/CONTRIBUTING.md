# How to Contribute

We'd like to thank you for sparing time to improve this project! Here are some guidelines for contributing：

To ensure that the blog design is not confused, this project does not accept suggestions for design changes, such as color scheme, fonts, typography, etc. If your request is about an enhancement, it is recommended to first submit a [_Feature Request_](https://github.com/cotes2020/jekyll-theme-chirpy/issues/new?labels=enhancement&template=feature_request.md) issue to discuss whether your idea fits the project.

## Basic Process

Generally, contribute to the project by:

1. Fork this project on GitHub and clone it locally.
2. Create a new branch from the default branch and give it a descriptive name (format: `feature/<add-new-feat>` / `fix/<fix-a-bug>`).
3. After completing the development, submit a new _Pull Request_. Note that the commit message must follow the [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/), otherwise it will fail the PR check.

## Modifying JavaScript

If your contribution involves JavaScript modification, please read the following sections.

### Inline Scripts

If you need to add comments to the inline JavaScript (the code between the HTML tags `<script>` and `</script>`), please use `/* */` instead of two slashes `//`. Because the HTML will be compressed by [jekyll-compress-html](https://github.com/penibelst/jekyll-compress-html) during deployment, but it cannot handle the `//` properly, which will disrupt the structure of the compressed HTML.

### External Scripts

If you need to add/change/delete the JavaScript in the directory `_javascript/`, setting up [`Node.js`](https://nodejs.org/) and [`npx`](https://www.npmjs.com/package/npx) is a requirement. And then install the development dependencies:

```console
$ npm i
```

During JavaScript development, real-time debugging can be performed through the following commands:

Firstly, start a Jekyll server:

```console
$ bash tools/run.sh
```

And then open a new terminal tab and run:

```console
# Type 'Ctrl + C' to stop
$ npx gulp dev
```

After debugging, run the command `npx gulp` (without any argument) will automatically output the compressed files to the directory `assets/js/dist/`.

## Verify the commit messages

If you want to make sure your commits pass the CI check, you can refer to the following steps.

Install `commitlint` & `husky`:

```console
$ npm i -g @commitlint/{cli,config-conventional} husky
```

And then enable `husky`:

```console
$ husky install
```

---

:tada: Your volunteering will make the open-source world more beautiful, thanks again! :tada:
