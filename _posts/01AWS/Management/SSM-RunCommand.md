---
title: AWS - Management - SSM - Run Command
date: 2020-07-18 11:11:11 -0400
categories: [01AWS, Management]
tags: [AWS]
toc: true
image:
---

[toc]


- ref
  - [aws doc](https://docs.aws.amazon.com/systems-manager/latest/userguide/setup-instance-profile.html)
  - [如何有效操控管理多台 EC2](https://www.ecloudture.com/how-to-remotely-run-command-on-ec2-instance/)

---


# Systems Manager - Run Command

---

## basic


AWS Systems Manager (SSM) 是一個集中管理 AWS 資源的服務，幫助使用者清楚掌握資源運作的狀況，且能執行 AWS 資源自動化處理，對於使用者來說，在管理上面擁有非常大的效益，可滿足合規及安全的要求。
- 在 AWS SSM 服務下有分成 **營運管理**、**應用程式管理**、**動作與變更**、**Instance 和節點**及**共享資源** 等五個面向的功能，
- 在 **Instance 和節點** 的操作下，有以下重點功能：

  - Managed Instance: 可以看到透過 AWS SSM 管理的 EC2 Instance（需要在 Instance 上安裝 SSM Agent）。

  - Session Manager: 不需要透過 SSH、Bastion Server 連線至 EC2，透過 HTTPS 協定連線至 EC2，幫助企業有效控管 Instance 存取權限。

  - State Manager: 透過自動化程序讓 Instance 維持在定義的狀態，可定期去追蹤 Instance 是否有依照設置狀態下執行。

  - Patch Manager: 自動修補 Instance 系統及應用程式的更新。

  - Run Command: 以遠端的方式，透過 Command 安全地管理 Instance 的設定。


## run commmand

- 以遠端的方式，透過 Command 安全地管理 Instance 的設定。
- remotely and securely manage the configuration of the managed instances at scale.
- perform on-demand changes like updating applications or running Linux shell scripts and Windows PowerShell commands on a target set of dozens or hundreds of instances.


## setup


### 前置作業

EC2 必須要具有 **AWS SSM 權限**及安裝 **SSM Agent**
- 如何在開機前事先安裝 SSM Agent，可以在創建 EC2 時，透過 User Data 達到此目的。

### 設置

> 透過 Command 一次對多台的 EC2 做軟體升級或是更改配置，所以這個時候就可以透過 **Run Command** 來達到此目的，不須 ssh 連線到每一台機器裡面去做操作。


1. AWS SSM 控制台 > `Managed Instances`，即可看到有安裝 EC2 Agent 的 EC2 狀態
2. 在左側導覽格中選擇 **Run Command**。
   - 然後選擇 **Run a command** 中，會看到許多官方預設提供的 **Command document**
   - 在此範例使用 **AWS-RunShellScript** Command document.

3. 選擇要對哪幾台 EC2 去做操作
   - **Specify instance tags**
     - EC2 有設置 Tag， 使用者可以選擇對有這個 Tag 的 EC2 做操作。
   - **Choose instances manually**
     - 目前 **Running** EC2 做操作。
     - 這些 Running 的 EC2 都必須要安裝 SSM Agent 及擁有 SSM Role 才能操作。
   - **Choose a resource group**，
     - 可以選取是先建立好的 resource group。

4. 接下來要在 **Commands** 的部分，貼上要下的指令：

5. 可以選擇將 **Command Output** 推到 **S3 Bucket** 或是 **CloudWatch log**。

6. 也能將 **Command Status** 透過 SNS 發送通知。
7. 設定完成後即可按下 **Run**。
   - 這時候就在控制台畫面看到 **Run Command** 的結果，像是執行成功、失敗或 Timeout 以及 Instance 數量。










---
