# How to Contribute

:tada: We really appreciate you taking the time to improve this project! :tada:

To ensure that the blog design is not confusing, this project does not accept
suggestions for design changes, such as color scheme, fonts, typography, etc.
If your request is about an enhancement, it is recommended to first submit a
[Feature Request][pr-issue] issue to discuss whether your idea fits the project.

Basically, you can follow these steps to complete the contribution.

1. Fork this project on GitHub and clone it locally.
2. Create a new branch from the default branch and give it a descriptive name
   (format: `feature/<add-new-feat>` or `fix/<fix-a-bug>`).
3. After completing development, create a [Conventional Commit][cc] with git.
  (See also: ["Verify the commits"](#verify-the-commits))
4. Create a [Pull Request][gh-pr].

## Make sure you can pass the CI tests

This project has [CI][ci] turned on. In order for your [PR][gh-pr] to pass the test,
please read the following.

### Check the core functionality
  
```console
bash ./tools/test
```

### Check the SASS syntax style

```console
npm test
```

### Verify the commits

Before you create a git commit, please complete the following setup.

Install `commitlint` & `husky`:

```console
npm i -g @commitlint/{cli,config-conventional} husky
```

And then enable `husky`:

```console
husky install
```

[pr-issue]: https://github.com/cotes2020/jekyll-theme-chirpy/issues/new?labels=enhancement&template=feature_request.md
[gh-pr]: https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/about-pull-requests
[cc]: https://www.conventionalcommits.org/
[ci]: https://en.wikipedia.org/wiki/Continuous_integration
