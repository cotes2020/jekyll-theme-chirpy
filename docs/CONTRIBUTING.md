# Contributing Guidelines

_First of all, thank you for considering contributing to this project_ ! :tada:

There are many ways to contribute, from writing tutorials or blog posts, improving the documentation, submitting bug
reports and feature requests, or writing code that can be incorporated into the project. In order to make a good
experience for both contributors and maintainers, please start with the "[General Rules](#general-rules)"
before taking further action.

## Table of Contents

- [General Rules](#general-rules)
- [Questions and Requests for Help](#questions-and-requests-for-help)
- [Reporting a Bug](#reporting-a-bug)
- [Suggesting a New Feature](#suggesting-a-new-feature)
- [Contributing Code/Documentation](#contributing-codedocumentation)
- [Helpful Resources](#helpful-resources)

## General Rules

All types of contributions (_pull requests_, _issues_, and _discussions_) should follow these rules:

- You should read through the [Wiki][wiki] to understand the project features and how to use it properly. This is to
respect the time of the project's developers and
maintainers and to save their energy for other problems that really need to be resolved.

- Use the [latest release version][latest-ver]. If your contribution involves code/documentation changes, update to the
latest version of the default (`master`) branch.

- Avoid making duplicate contributions by searching for existing [issues][issues] / [discussions][discus] /
[pull requests][pr], but don't leave any unhelpful comments such as "I have the same problem". Prefer using
[reactions][gh-reactions] if you simply want to "+1" an existing issue.

- DO NOT email or tweet the
project developers and maintainers directly, everything about the project should be left on GitHub.

**Tip**: If you are new to the open-source community, then please read through
"[How To Ask Questions The Smart Way][ext-reading]" before contributing.

## Questions and Requests for Help

We expect every reasonable question you ask to be answered appropriately. If you want a quick and timely response,
please ask questions at [Jekyll Talk][jekyll-talk] and [StackOverflow][stack-overflow], where there are tons of
enthusiastic geeks who will positively answer your challenging questions.

If you can't get an answer in any of the above ways, then create a new [discussion][discus]. As long as it is not a
duplicate and [RTFM][rtfm] / [STFW][stfw] issue, we will respond as soon as possible.

## Reporting a Bug

A great way to contribute to the project is to send a detailed issue when you encounter a problem. We always appreciate
a well-written, thorough bug report.

1. Please figure out why the bug occurred, or locate the module in the project that caused this bug. Otherwise, there is
a high probability that you are using/setting it incorrectly.

2. If the issue is caused by you modifying the project code or some configuration of Jekyll, then please DO NOT
report such "bugs".
Chirpy is an open-source project, but that doesn't mean we will maintain other specific forks (such as yours).
You can learn about Jekyll and modern Web development to solve problems caused by custom modifications.

3. Make good use of your browser's incognito mode to troubleshoot if the problem is caused by caching.

4. As a last option, you can create a new [Bug Report][bug-report] following the template to describe the details.
If possible, providing a demo that reproduces the error will help us troubleshoot faster.

## Suggesting a New Feature

Feature requests are welcome! While we will consider all requests, we cannot guarantee your request will be accepted.  
We want to avoid chaos in the UI design and therefore do not accept requests for changes like color schemes,
fontfamilies, typography, and so on. We want to avoid [feature creep][feat-creep] and focus only on the core features.
If accepted, we cannot make any commitments regarding the timeline for implementation and release. However, you are
welcome to submit a pull request to help!

## Contributing Code/Documentation

If your request is about an enhancement, it is recommended to first submit a
[Feature Request][feat-request] to discuss whether your idea fits the project.
See also: "[Suggesting a New Feature](#suggesting-a-new-feature)". Other than that, you can start the PR process.

1. Fork this project on GitHub and clone your repository locally.
2. Setting up the [development & test environments][dev-env].
3. Creating a new branch from the default branch and give it a descriptive name (e.g. `add-a-new-feat` or `fix-a-bug`).
When development is complete, create a [Conventional Commit][cc] with Git.
4. Submitting a [Pull Request][gh-pr].

## Helpful Resources

- [Code of conduct](https://github.com/cotes2020/jekyll-theme-chirpy/blob/master/docs/CODE_OF_CONDUCT.md)
- [Security policy](https://github.com/cotes2020/jekyll-theme-chirpy/blob/master/docs/SECURITY.md)
- [How To Ask Questions The Smart Way][ext-reading]

[latest-ver]: https://github.com/cotes2020/jekyll-theme-chirpy/releases/latest
[wiki]: https://github.com/cotes2020/jekyll-theme-chirpy/wiki
[issues]: https://github.com/cotes2020/jekyll-theme-chirpy/issues?q=is%3Aissue
[pr]: https://github.com/cotes2020/jekyll-theme-chirpy/pulls
[discus]: https://github.com/cotes2020/jekyll-theme-chirpy/discussions
[ext-reading]: http://www.catb.org/~esr/faqs/smart-questions.html
[jekyll-talk]: https://talk.jekyllrb.com/
[stack-overflow]: https://stackoverflow.com/questions/tagged/jekyll
[rtfm]: https://en.wikipedia.org/wiki/RTFM
[stfw]: https://www.webster-dictionary.org/definition/STFW
[gh-reactions]: https://github.blog/2016-03-10-add-reactions-to-pull-requests-issues-and-comments/
[bug-report]: https://github.com/cotes2020/jekyll-theme-chirpy/issues/new?assignees=&labels=&projects=&template=bug_report.yml
[feat-request]: https://github.com/cotes2020/jekyll-theme-chirpy/issues/new?assignees=&labels=enhancement&projects=&template=feature_request.yml
[feat-creep]: https://en.wikipedia.org/wiki/Feature_creep
[dev-env]: https://github.com/cotes2020/jekyll-theme-chirpy/wiki/Development-&-Test-Environments
[cc]: https://www.conventionalcommits.org/
[gh-pr]: https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/about-pull-requests
