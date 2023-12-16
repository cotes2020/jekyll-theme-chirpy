---
title: AWS - CodeDevelop - CodeCommit
date: 2020-07-18 11:11:11 -0400
categories: [01AWS, CodeDevelop]
tags: [AWS]
toc: true
image:
---

- [CodeCommit](#codecommit)
  - [basic](#basic)
  - [work flow](#work-flow)
  - [setup](#setup)

---

# CodeCommit

![CodeCommit](https://i.imgur.com/ENCpW6N.png)

- a <font color=red> version control service </font> hosted by Amazon Web Services
  - tracks and manages code changes
  - Maintains version history
- <font color=red> Centralized Code Repository </font>
  - use to privately store and manage assets (such as documents, source code, and binary files) in the cloud.
  - place to store source code, binaries, libraries, images, HTML files ...
  - based on Git
- <font color=red> Enables Collaboration </font>
  - manages updates from multiple users.

---

## basic
- secure, highly scalable, managed source control service that hosts private Git repositories.
- eliminates the need to manage your own source control system or scaling the infrastructure.
- use CodeCommit to store anything from code to binaries.
- It supports the standard functionality of Git, works seamlessly with your existing Git-based tools.

With CodeCommit, you can:

1. <font color=red> fully managed service </font> hosted by AWS.
   - high service availability and durability and eliminates the administrative overhead of managing your own hardware and software.
   - no hardware to provision and scale and no server software to install, configure, and update.

2. <font color=red> Store code securely </font>
   - Encryption
     - CodeCommit repositories are auto encrypted at rest and in transit through AWS Key Management Service (AWS KMS) using customer-specific keys.
     - You can transfer your files to and from AWS CodeCommit using HTTPS or SSH, as you prefer.
   - Access Control
     - AWS CodeCommit uses AWS Identity and Access Management to control and monitor who can access the data and how, when, and where they can access it.
     - CodeCommit also helps you monitor your repositories via `AWS CloudTrail and AWS CloudWatch`.


3. <font color=red> Work collaboratively on code </font>
   - CodeCommit repositories <font color=blue> support pull requests </font>
     - provide a mechanism to request code reviews and discuss code with collaborators.
   - users can review and comment on each other's code changes before merging them to branches;
      - easily <font color=blue> commit, branch, and merge </font> the code to easily maintain control of team’s projects.
   - notifications that automatically send emails to users about pull requests and comments; and more.

4. <font color=red> Easily scale the version control projects </font>
   - CodeCommit repositories can scale up to meet your development needs.
   - The service can handle repositories with large numbers of files or branches, large file sizes, and lengthy revision histories.

5. <font color=red> Store anything, anytime </font>
   - no limit on the size of repositories or the file types to store.

6. Integrate with other AWS and third-party services.
   - CodeCommit keeps your repositories close to your other production resources in the AWS Cloud, which helps increase the speed and frequency of your development lifecycle.
   - It is integrated with IAM and can be used with other AWS services and in parallel with other repositories.

7. <font color=red> Easy Access and Integration </font>
   - Easily migrate files from other remote repositories.
     - migrate to CodeCommit from any Git-based repository.
   - use the AWS Management Console, AWS CLI, and AWS SDKs to manage your repositories.
   - can also use Git commands or Git graphical tools to interact with your repository source files.
     - AWS CodeCommit supports all Git commands and works with your existing Git tools.
     - You can integrate with your development environment plugins or continuous integration/continuous delivery systems.


8. <font color=red> High Availability and Durability </font>
   - AWS CodeCommit stores your repositories in Amazon S3 and Amazon DynamoDB.
   - encrypted data is redundantly stored across multiple facilities.
   - increases the availability and durability of the repository data.
   - Unlimited Repositories
     - create as many repositories as you need
     - up to 1,000 repositories by default and no limits upon request.
     - You can store and version any kind of file, including application assets such as images and libraries alongside your code.


9. Notifications and Custom Scripts
   - receive notifications for events impacting your repositories.
   - Notifications will come in the form of Amazon SNS notifications.
   - Each notification will include a status message as well as a link to the resources whose event generated that notification.
   - Additionally, using AWS CodeCommit repository triggers, you can send notifications and create HTTP webhooks with Amazon SNS or invoke AWS Lambda functions in response to the repository events you choose.

---


## work flow

- similar to Git-based repositories
- provides a console for easy creation of repositories and the listing of existing repositories and branches.
- find information about a repository and clone it to their computer, creating a local repo where they can make changes and then push them to the CodeCommit repository.
- Users can work from
  - the command line on local machines
  - or use a GUI-based editor.

![arc-workflow](https://i.imgur.com/dCtKMc0.png)


![Screen Shot 2020-12-27 at 03.33.55](https://i.imgur.com/NDqsZwC.png)


![Screen Shot 2020-12-27 at 03.34.24](https://i.imgur.com/a99WuOp.png)


![Screen Shot 2020-12-27 at 03.34.47](https://i.imgur.com/Tb6tPsF.png)


---


## setup

1. Create a Server from an Amazon Linux 2 AMI
2. Create IAM user for CodeCommit
   1. user:`cloud_user`
   2. User ARN: `arn:aws:iam::183169071737:user/cloud_user`
   3. Add permissions > Attach existing policies directly.
      1. `AWSCodeCommitFullAccess` policy
   4. user credentials:
      1. create access Key: <font color=red> for aws configure </font>
         - `AcessKeyID` and `cloud_user_accessKeys.csv`
      2. create HTTPS Git credentials for AWS CodeCommit: <font color=red> for git clone </font>
         - Username: `cloud_user-at-183169071737`
         - credentials: `cloud_user_codecommit_credentials.csv`


```bash
aws configure
<<<<<<< HEAD
# AWS Access Key ID [None]: AKIASVJN2HJ4ULJBBUEH
# AWS Secret Access Key [None]: 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
=======
# AWS Access Key ID [None]: abcd
# AWS Secret Access Key [None]: abcd
>>>>>>> 1a148b47672b35d180699fc905d033785c8bbe28
# Default region name [None]: us-east-1
# Default output format [None]: json

aws codecommit create-repository --repository-name RepoFromCLI --repository-description "My 1st repository"
# {
#     "repositoryMetadata": {
#         "accountId": "183169071737",
#         "repositoryId": "bfc04a5a-833f-4d33-b2aa-6b50f91db4ee",
#         "repositoryName": "RepoFromCLI",
#         "repositoryDescription": "My 1st repository",
#         "lastModifiedDate": "2021-01-18T16:53:00.294000-05:00",
#         "creationDate": "2021-01-18T16:53:00.294000-05:00",
#         "cloneUrlHttp": "https://git-codecommit.us-east-1.amazonaws.com/v1/repos/RepoFromCLI",
#         "cloneUrlSsh": "ssh://git-codecommit.us-east-1.amazonaws.com/v1/repos/RepoFromCLI",
#         "Arn": "arn:aws:codecommit:us-east-1:183169071737:RepoFromCLI"
#     }
# }

git clone https://git-codecommit.us-east-1.amazonaws.com/v1/repos/RepoFromCLI
cd RepoFromCLI/
cat "hello" > test.text
git add test.text

git commit -m "added test.txt"
# [master 16ac6f1] added test.txt
#  1 file changed, 1 insertion(+)
#  create mode 100644 test.text

git log
# commit 16ac6f1cd97caef191132db56095efc126960753 (HEAD -> master)
# Author: L.desk <54053176+ocholuo@users.noreply.github.com>
# Date:   Mon Jan 18 16:59:52 2021 -0500
#     added test.txt
# commit 7e0c3566758489d4afb3394e95b48ffcbba524d6 (origin/master, origin/HEAD)
# Author: 1 <1111@aws.com>
# Date:   Mon Jan 18 21:54:10 2021 +0000
#     Added Screen Shot 2021-01-18 at 16.28.28.png

git push -u origin master
# Enumerating objects: 4, done.
# Counting objects: 100% (4/4), done.
# Delta compression using up to 8 threads
# Compressing objects: 100% (2/2), done.
# Writing objects: 100% (3/3), 327 bytes | 327.00 KiB/s, done.
# Total 3 (delta 0), reused 0 (delta 0), pack-reused 0
# To https://git-codecommit.us-east-1.amazonaws.com/v1/repos/RepoFromCLI
#    7e0c356..16ac6f1  master -> master
# Branch 'master' set up to track remote branch 'master' from 'origin'.

git log
# commit 16ac6f1cd97caef191132db56095efc126960753 (HEAD -> master, origin/master, origin/HEAD)
# Author: L.desk <54053176+ocholuo@users.noreply.github.com>
# Date:   Mon Jan 18 16:59:52 2021 -0500
#     added test.txt
# commit 7e0c3566758489d4afb3394e95b48ffcbba524d6
# Author: 1 <1111@aws.com>
# Date:   Mon Jan 18 21:54:10 2021 +0000
#     Added Screen Shot 2021-01-18 at 16.28.28.png
```








.
