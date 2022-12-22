---
title: Benefits of CI - Decrease code review time
description: A thought on how well thought through CI pipeline can help in code reviews. Article to share thoughts on benefits for continuous integration. CI pipeline can help developers to review the code more effectively.
tags: ["ci", "releases"]
category: ["CI"]
date: 2017-01-17
permalink: '/ciBlogs/benefits-of-ci-decrease-code-review-time/'
---

Sometimes, we work in teams where code churn is high. Multiple developers working on features in the same code base, and there is a pressure of delivery. In these high-pressure situations reviewing code becomes critical. However, the pressure of delivery pushes the team to merge the code as soon as possible.

A well thought through CI pipeline can make a developer's life easier. Teams can quickly hook the CI pipeline updates to slack channels. These updates can include:
* A merge request that was raised
* Any failed test scenarios on the merge request
* CI to run code coverage tools and report it back to the team
* CI to run static analysis job to provide immediate feedback to developers

Thus, when a peer reviews the code, it has already passed a few quality gates.


