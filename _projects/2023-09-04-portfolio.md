---
layout: project
title: Altered Chirpy theme
author: cabiste
date: 2023-09-04 12:00:00 +0100
tags: [portfolio, jekyll, chirpy]
pin: true
image:
  path: /assets/img/posts/text-and-typography/devices-mockup.png
  alt: Responsive rendering of Chirpy theme on multiple devices.
---

## Introduction

This is a modified fork of the popular jekyll theme [chirpy](https://chirpy.cotes.page/)

### what changed?

Added:

- A projects list page
- A project layout (no comments, or recommendations)

Removed:

- Archive page (can be restored)
- Tags page (can be restored)
- image LQIP (kinda buggy outside of home page)

## How to use

modify `_config.yml` and set your own configs.

To create a new post or project use the provided templates.

Add yourself as an author in `_data/authors.yml` following the template (you can remove the other authors).

If you want to add or remove some / All the contact buttons on the side bard you can do so by modifying `_data/contact.yml`

If you want to add or remove some / All the share buttons in the post page you can do so by modifying `_data/share.yml`

## tools

Use `tools/init` when you first setting up the project.

Use `tools/run` to execute the project.

Use `tools/test` to run the tests and make sure your website is deployable by github.
