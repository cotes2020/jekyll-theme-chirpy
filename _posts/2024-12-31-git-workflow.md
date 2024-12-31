---
title: git workflow
date: "2024-12-31T21:41:41+09:00"
categories: [Knowledge, IT]
tags: [git, development]
description: 로그인 기능 개발을 위한 Git과 GitHub 워크플로우입니다.
author: hoon
---

가상의 시나리오입니다.

| Step | Example Action                                                | Performed By | Description                                                      |
| :--- | :------------------------------------------------------------ | :----------- | :--------------------------------------------------------------- |
| 1    | `git clone https://github.com/your-username/project-name.git` | Developer    | Clone the repository to the local machine to start working.      |
| 2    | `git checkout -b feature/login-function`                      | Developer    | Create a new branch for the feature development.                 |
| 3    | `Implement login functionality`                               | Developer    | Implement the login functionality in the code.                   |
| 4    | `git add .`                                                   | Developer    | Stage the changes to prepare for commit.                         |
| 5    | `git commit -m "Implement login functionality"`               | Developer    | Commit the staged changes.                                       |
| 6    | `git push origin feature/login-function`                      | Developer    | Push the committed changes to the remote repository.             |
| 7    | `Create PR`                                                   | Developer    | Create a pull request to request a code review.                  |
| 8    | `Request changes after review`                                | Team Lead    | The pull request is rejected, and changes are requested.         |
| 9    | `Apply requested changes and push again`                      | Developer    | Apply the feedback, modify the code, and push the changes again. |
| 10   | `git pull origin main`                                        | Developer    | Pull the latest updates from the main branch.                    |
| 11   | `git push origin feature/login-function`                      | Developer    | Resolve any conflicts and push the changes.                      |
| 12   | `Request final review`                                        | Developer    | Notify the reviewer to perform the final code review.            |
| 13   | `Merge PR`                                                    | Team Lead    | Merge the pull request into the main branch.                     |
| 14   | `git push origin main`                                        | Team Lead    | Push the merged changes to the main branch.                      |