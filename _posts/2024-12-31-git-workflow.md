---
title: git workflow
date: "2024-12-31T21:41:41+09:00"
categories: [Knowledge, IT]
tags: [git, development]
description: 로그인 기능 개발을 위한 Git과 GitHub 워크플로우입니다.
author: hoon
---
test
가상의 시나리오입니다.

기존의 프로젝트 리포지토리를 클론 후 로컬에서 개발 한 뒤 머지하는 과정입니다.

| Step                                     | Example Code                                                  | Performed By | Description                                                      |
| :--------------------------------------- | :------------------------------------------------------------ | :----------- | :--------------------------------------------------------------- |
| 1. Clone the Repository                  | `git clone https://github.com/your-username/project-name.git` | Developer    | Clone the repository to the local machine to start working.      |
| 2. Create Feature Branch                 | `git checkout -b feature/login-function`                      | Developer    | Create a new branch for the feature development.                 |
| 3. Implement Functionality               |                                                               | Developer    | Implement the login functionality in the code.                   |
| 4. Stage Changes                         | `git add .`                                                   | Developer    | Stage the changes to prepare for commit.                         |
| 5. Commit Changes                        | `git commit -m "Implement login functionality"`               | Developer    | Commit the staged changes.                                       |
| 6. Push Changes                          | `git push origin feature/login-function`                      | Developer    | Push the committed changes to the remote repository.             |
| 7. Create Pull Request                   |                                                               | Developer    | Create a pull request to request a code review.                  |
| 8. Code Review Rejected                  |                                                               | Team Lead    | The pull request is rejected, and changes are requested.         |
| 9. Apply Requested Changes and Re-review |                                                               | Developer    | Apply the feedback, modify the code, and push the changes again. |
| 10. Sync with Main                       | `git pull origin main`                                        | Developer    | Pull the latest updates from the main branch.                    |
| 11. Resolve Conflicts and Push           | `git push origin feature/login-function`                      | Developer    | Resolve any conflicts and push the changes.                      |
| 12. Notify for Final Review              |                                                               | Developer    | Notify the reviewer to perform the final code review.            |
| 13. Merge Pull Request                   |                                                               | Team Lead    | Merge the pull request into the main branch.                     |
| 14. Push After Merge                     | `git push origin main`                                        | Team Lead    | Push the merged changes to the main branch.                      |