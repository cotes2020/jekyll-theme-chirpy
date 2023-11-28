---
title: Workflows
author: Sarath
date: 2023-11-28
categories: [DevOps, Github Actions]
tags: [Github Actions, CI/CD]
pin: true
---

# Advanced Workflows in GitHub Actions

**Objective**: Delve into more complex workflows in GitHub Actions to enhance automation capabilities.

## Setting up Workflows for Different Branches
Creating branch-specific workflows allows you to automate different processes for development, staging, and production environments.

### Example: Branch-specific Workflows
```yaml
name: Branch-Specific Workflow

on:
  push:
    branches:
      - main
      - develop

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v2
    - name: Run different scripts based on branch
      run: |
        if [ "\${{ github.ref }}" == "refs/heads/main" ]; then
          echo "Deploying to production"
          # Add production deployment scripts here
        elif [ "\${{ github.ref }}" == "refs/heads/develop" ]; then
          echo "Deploying to staging"
          # Add staging deployment scripts here
        fi
```
This example demonstrates how to set up different actions for the `main` and `develop` branches.

## Managing Workflow Dependencies
Sequential and parallel execution of jobs can help manage dependencies within your workflow.

### Example: Job Dependencies
```yaml
jobs:
  job1:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v2
    - name: First job step
      run: echo "This is the first job."

  job2:
    needs: job1
    runs-on: ubuntu-latest
    steps:
    - name: Second job step
      run: echo "This runs after job1 completes."
```
In this setup, `job2` will only run after `job1` has successfully completed.

## Using Environment Variables and Secrets
Environment variables and secrets are essential for managing sensitive data and customizing workflows.

### Example: Using Secrets
```yaml
jobs:
  build:
    runs-on: ubuntu-latest
    env:
      CUSTOM_VAR: "Sample Value"
    steps:
    - uses: actions/checkout@v2
    - name: Use secret
      run: echo "Secret value is ${{ secrets.MY_SECRET }}"
    - name: Use environment variable
      run: echo "Environment variable value is $CUSTOM_VAR"
```
This example shows how to use secrets (stored in GitHub repository settings) and environment variables in your workflow.

**Expected Outcome**: By understanding these advanced concepts, you can create more nuanced and efficient workflows to suit the specific needs of your project.

---