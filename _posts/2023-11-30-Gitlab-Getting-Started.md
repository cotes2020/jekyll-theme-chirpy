---
title: Getting Started with GitLab CI/CD
author: Sarath
date: 2023-11-30
categories: [DevOps, Gitlab]
tags: [Gilab, CI/CD]
pin: true
---

# Getting Started with GitLab CI/CD

## Setting Up a GitLab Account and Project

### Step 1: Creating a GitLab Account
- Visit [GitLab's sign up page](https://gitlab.com/users/sign_in#register-pane).
- Fill in your details and follow the prompts to create a new account.

### Step 2: Setting Up a New Project
- Once logged in, click on the "New project" button.
- Choose a project name, visibility level, and initialize with a README if desired.
- Click "Create project" to set up your new project.

## Understanding GitLab CI/CD Pipelines and .gitlab-ci.yml File

GitLab CI/CD pipelines are configured using a YAML file called `.gitlab-ci.yml` placed in the root of your repository. This file defines the structure and order of the pipeline and determines:

- What to execute using GitLab Runner.
- When to run the jobs.
- How to run the jobs.

You can learn more about the `.gitlab-ci.yml` file in the [GitLab CI/CD Pipeline Configuration Reference](https://docs.gitlab.com/ee/ci/yaml/).

## Creating Your First Simple CI/CD Pipeline

### Step 1: Writing a Basic `.gitlab-ci.yml` File
- Create a `.gitlab-ci.yml` file in the root of your repository.
- Define a simple job, for example:

```yaml
job1:
  script:
    - echo "Hello, World!"
```

### Step 2: Commit and Push Your Changes
- Commit the `.gitlab-ci.yml` file to your repository.
- Push the commit to GitLab.

### Step 3: Viewing the Pipeline in Action
- Go to the CI/CD section of your project in GitLab.
- You'll see the pipeline running the job you defined.

Congratulations! You have created your first simple CI/CD pipeline in GitLab.

---

Stay tuned for more insightful posts in this GitLab CI/CD series!
