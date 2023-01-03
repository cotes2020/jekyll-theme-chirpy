---
title: CI/CD - Github Action Template
date: 2020-11-11 11:11:11 -0400
categories: [10SecConcept]
tags: [SecConcept]
toc: true
image:
---

- [CI/CD - Github Action Template](#cicd---github-action-template)
  - [example 1](#example-1)
  - [example 2: build a website](#example-2-build-a-website)

---


# CI/CD - Github Action Template

---


## example 1

```yml
# This is a basic workflow to help you get started with Actions
name: CI

# Controls when the action will run.
on:
  # Triggers the workflow on push or pull request events but only for the main branch
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]

  # Allows you to run this workflow manually from the Actions tab
  workflow_dispatch:

# A workflow run is made up of one or more jobs that can run sequentially or in parallel
jobs:
  # This workflow contains a single job called "build"
  build:
    # The type of runner that the job will run on
    runs-on: ubuntu-latest

    # Steps represent a sequence of tasks that will be executed as part of the job
    steps:
      # Checks-out your repository under $GITHUB_WORKSPACE, so your job can access it
      - uses: actions/checkout@v2

      # Runs a single command using the runners shell
      - name: Run a one-line script
        run: echo Hello, world!

      # Runs a set of commands using the runners shell
      - name: Run a multi-line script
        run: |
          echo Add other actions to build,
          echo test, and deploy your project.
```


---



## example 2: build a website

```yml
name: 'Automatic build'

# Controls when the action will run.
on:
  # Triggers the workflow on push or pull request events but only for the main branch
  push:                 # when push
    branches:           # which brances will trigger
      - master
    paths-ignore:       # which path will not trigger
      - '.gitignore'
      - 'README.md'
      - 'LICENSE'
  pull_request:
    branches: [ main ]

# A workflow run is made up of one or more jobs that can run sequentially or in parallel
jobs:
   # job 1 name： This workflow contains a single job called "build-n-test"
  build-n-test:
    # The type of runner that the job will run on
    runs-on: ubuntu-latest

    # Steps represent a sequence of tasks that will be executed as part of the job
    steps:
      - uses: actions/setup-ruby@v1     # run official "actions/checkout@v2" code to setup the environment
        with:
          ruby-version: '2.6.x'

      # Checks-out your repository under $GITHUB_WORKSPACE, so your job can access it
      - name: Checkout                  # name the shell
        uses: actions/checkout@v2       # run official "actions/checkout@v2" code to copy the current code
        with:
          fetch-depth: 0

      - name: Bundle Caching
        id: bundle-cache
        uses: actions/cache@v1
        with:
          path: vendor/bundle
          key: ${{ runner.os }}-gems-${{ hashFiles('**/Gemfile') }}
          restore-keys: |
            ${{ runner.os }}-gems-

      - name: Bundle config                    # name the shell
        run: |                                 # the cmd to run script
          bundle config path vendor/bundle

      - name: Bundle Install
        if: steps.bundle-cache.outputs.cache-hit != 'true'
        run: |
          bundle install

      - name: Bundle Install locally
        if: steps.bundle-cache.outputs.cache-hit == 'true'
        run: |
          bundle install --local

      - name: Build Site
        run: |
          bash tools/build.sh -b ""

      - name: Test Site
        run: |
          bash tools/test.sh

  deploy:                       # job 2 name
    needs: build-n-test         # needs job 1 to be run first for deploy to be run
    runs-on: ubuntu-latest

    steps:
      - uses: actions/setup-ruby@v1
        with:
          ruby-version: '2.6.x'

      - name: Checkout
        uses: actions/checkout@v2
        with:
          fetch-depth: 0

      - name: Bundle Caching
        id: bundle-cache
        uses: actions/cache@v1
        with:
          path: vendor/bundle
          key: ${{ runner.os }}-gems-${{ hashFiles('**/Gemfile') }}
          restore-keys: |
            ${{ runner.os }}-gems-

      - name: Bundle config
        run: |
          bundle config path vendor/bundle

      - name: Bundle Install
        if: steps.bundle-cache.outputs.cache-hit != 'true'
        run: |
          bundle install

      - name: Bundle Install locally
        if: steps.bundle-cache.outputs.cache-hit == 'true'
        run: |
          bundle install --local

      - name: Build site
        run: |
          bash tools/build.sh

      - name: Deploy
        run: |
          bash tools/deploy.sh
```











.
