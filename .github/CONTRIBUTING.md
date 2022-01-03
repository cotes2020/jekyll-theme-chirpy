# How to Contribute

We want to thank you for sparing time to improve this project! Here are some guidelines for contributing：

To ensure that the blog design is not confused, this project does not accept suggestions for design changes, such as color scheme, fonts, typography, etc. If your request is about an enhancement, it is recommended to first submit a [_Feature Request_](https://github.com/cotes2020/jekyll-theme-chirpy/issues/new?labels=enhancement&template=feature_request.md) issue to discuss whether your idea fits the project.

## Basic Process

Generally, contribute to the project by:

1. Fork this project on GitHub and clone it locally.
2. Create a new branch from the default branch and give it a descriptive name (e.g., `my-new-feature`, `fix-a-bug`).
3. After completing the development, submit a new _Pull Request_.

## Modifying JavaScript

If your contribution involves JS modification, please read the following sections.

### Inline Scripts

If you need to add comments to the inline JS (the JS code between the tags `<script>` and `</script>`), please use `/**/` instead of two slashes `//`. Because the HTML will be compressed by [jekyll-compress-html](https://github.com/penibelst/jekyll-compress-html) during deployment, but it cannot handle the `//` properly. And this will disrupt the structure of the compressed HTML.

### External Scripts

If you need to add or modify JavaScripts in the directory `_javascript`, you need to install [Gulp.js](https://gulpjs.com/docs/en/getting-started/quick-start).

During development, real-time debugging can be performed through the following commands:

```console
$ bash tools/run.sh
```

Open another terminal tab and run:

```console
$ gulp dev # Type 'Ctrl + C' to stop
```

After debugging, run the command `gulp` (without any argument) will automatically output the compressed files to the directory `assests/js/dist/`.

---

:tada: Your volunteering will make the open-source world more beautiful, thanks again! :tada:
