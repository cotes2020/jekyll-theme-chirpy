# Contributing Guidelines

:tada: First of all, thank you for considering contributing to this project! :tada:

There are many ways to contribute, from writing tutorials or blog posts, improving the documentation, submitting bug reports and feature requests or writing code which can be incorporated into Chirpy itself.

As a consumer of the open source community, you should make sure that you have gone through the [Wiki][wiki] to understand the project features and how to use it properly. This is to respect the time of the project's developers and maintainers, and to save their energy for other problems that really need to be solved. Please DO NOT email or tweet the project maintainers directly, everything about Chirpy should be left in GitHub issues/PRs.

**Tips**: If you are new to open source community, here is a very useful article for you — "[How To Ask Questions The Smart Way][ext-reading]".

## In This Article

- [Questions and Requests for Help](#questions-and-requests-for-help)
- [File a Bug Report](#file-a-bug-report)
- [Suggest a New Feature](#suggest-a-new-feature)
- [Submitting a Pull Request](#submitting-a-pull-request)
  - [Contributing Code and Documentation Changes](#contributing-code-and-documentation-changes)
  - [How To Pass the CI Tests](#how-to-pass-the-ci-tests)

## Questions and Requests for Help

We want to make sure that every reasonable question you have is answered appropriately. In most cases, you can get an answer by checking the Wiki and existing issues. Alternatively, you can ask a question in the [Jekyll Forum][forum] and [StackOverflow][stack_overflow], where there are tons of enthusiastic geeks eager to answer your interesting questions.

If you can't get an answer in any of the above ways, then open a new issue as your last option. As long as it is not a duplicate / [RTFM][rtfm] / [STFW][stfw] issue, we will respond as soon as possible.

## File a Bug Report

A great way to contribute to the project is to send a detailed issue when you encounter a problem. We always appreciate a well-written, thorough bug report.

- If the issue is caused by you modifying the project code or some configuration of Jekyll, then please **DO NOT** report such "bugs".
This project is open source, but it doesn't mean that we will maintain other specific projects (such as yours).
You can learn about Jekyll and modern Web development to solve problems caused by custom modifications.

- Update to the latest version and see if that solves the problem.

- Make good use of your browser's incognito mode to troubleshoot if the problem is caused by caching.

- Search for similar issues, but don't leave unhelpful comments such as "I had the same problem". Prefer using [reactions][gh-reactions] if you simply want to "+1" an existing issue.

- Once you've gone through the above, you can use the bug-report template to create a new issue, filling in the description of the bug according to the template. If you can, provide a small example of the bug that can be reproduced for faster troubleshooting.

## Suggest a New Feature

Feature requests are welcome! While we will consider all requests, we cannot guarantee your request will be accepted.

We want to avoid chaos in the UI design, so we won't accept requests for changes like color schemes, font families, typography, and so on. **Do not open a duplicate feature request.** Search for existing feature requests first. If you find your feature (or one very similar) previously requested, comment on that issue.

If accepted, we cannot make any commitments regarding the timeline for implementation and release. However, you are welcome to submit a pull request to help!

## Submitting a Pull Request

### Contributing Code and Documentation Changes

In short, you can follow these steps to complete the contribution.

1. Fork this project on GitHub and clone your repository locally.
2. Setting up the [development environment][dev-env].
3. Create a new branch from the default branch and give it a descriptive name (e.g. `add-a-new-feat` or `fix-a-bug`). When development is complete, create a [Conventional Commit][cc] with Git. (See also: "[Verify the commits](#verify-the-commits)")
4. Create a [Pull Request][gh-pr].

### How To Pass the CI Tests

This project has [CI][ci] turned on. In order for your pull request to pass the test,
please read the following.

#### Verify the Commits

Before you create a git commit, please complete the following setup.

Install `commitlint` & `husky`:

```console
npm i -g @commitlint/{cli,config-conventional} husky
```

And then enable `husky`:

```console
husky install
```

#### Check the Core Functionality
  
```console
bash ./tools/test
```

#### Check the SASS Code Style

```console
npm test
```

[wiki]: https://github.com/cotes2020/jekyll-theme-chirpy/wiki
[ext-reading]: http://www.catb.org/~esr/faqs/smart-questions.html
[forum]: https://talk.jekyllrb.com/
[stack_overflow]: https://stackoverflow.com/questions/tagged/jekyll
[rtfm]: https://en.wikipedia.org/wiki/RTFM
[stfw]: https://www.webster-dictionary.org/definition/STFW
[gh-reactions]: https://github.blog/2016-03-10-add-reactions-to-pull-requests-issues-and-comments/
[dev-env]: https://github.com/cotes2020/jekyll-theme-chirpy/wiki/Development
[cc]: https://www.conventionalcommits.org/
[gh-pr]: https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/about-pull-requests
[ci]: https://en.wikipedia.org/wiki/Continuous_integration
