---
title: VS Code Extensions For Power BI Development and Administration
description: VS Code extension for Power BI Development and Administration
author: duddy
date: 2024-05-16 00:00:00 +0000
categories: [VS Code Extension]
tags: [vscode extension]
pin: false
image:
  path: /assets/img/005-VSCodeExtensions/VS%20Code.png
  alt: VS Code
---

[VS Code](https://code.visualstudio.com/) is a simple, powerful and highly customizable IDE. Features include syntax highlighting, intellisense, git version control, integrated terminal, and more. The Extension Marketplace offers additional languages and editor features. You can make your environment your own, by applying themes and setting up editor shortcuts and preferences.

Because of this, VS Code is the IDE most used by developers; as per the [Stack Overflow 2023 Developer Survey](https://survey.stackoverflow.co/2023/#section-most-popular-technologies-integrated-development-environment).

# Extensions

Here are subset of extensions I use for Power BI development, administration and automation.

## Languages

| Language                                                                                                      | VS Code Extension ID         | Syntax Highlighting | Intellisense |
| ------------------------------------------------------------------------------------------------------------- | ---------------------------- | :-----------------: | :----------: |
| [TMDL](https://learn.microsoft.com/en-us/analysis-services/tmdl/tmdl-overview?view=asallproducts-allversions) | analysis-services.TMDL       |         ✔️          |      ❌      |
| [DAX](https://learn.microsoft.com/en-us/dax/)                                                                 | jianfajun.dax-language       |         ✔️          |      ❌      |
| [M](https://learn.microsoft.com/en-us/powerquery-m/)                                                          | PowerQuery.vscode-powerquery |         ✔️          |      ✔️      |
| [C#](https://learn.microsoft.com/en-us/dotnet/csharp/language-reference/)                                     | ms-dotnettools.csharp        |         ✔️          |      ✔️      |
| [Powershell](https://learn.microsoft.com/en-us/powershell/)                                                   | ms-vscode.powershell         |         ✔️          |      ✔️      |
| [Python](https://www.python.org/)                                                                             | ms-python.python             |         ✔️          |      ✔️      |
| [YAML](https://yaml.org/)                                                                                     | redhat.vscode-yaml           |         ✔️          |      ❌      |

TMDL, DAX and M are used to review and update tabular models that have been decompiled to files (.pbip, pbitools, tabular editor). Powershell and C# for scripting using the analysis services libraries and cmdlets. Python and Powershell for scripting using the [Power BI](https://learn.microsoft.com/en-us/rest/api/power-bi/) and [Fabric](https://learn.microsoft.com/en-us/rest/api/fabric/articles/) REST APIs. YAML is used for developing [GitHub](https://github.com/)/[ADO](https://azure.microsoft.com/en-us/products/devops) pipelines.

## Version Control

| Extension | VS Code Extension ID |
| --------- | -------------------- |
| Gitlens   | eamodio.gitlens      |
| Git Graph | mhutchie.git-graph   |

These extensions extend VS Code's inbuilt git version control functionality. The features I use most are Inline blame annotation on rows, and On hover pop-up to show the details of commit.

![Blame](/assets/img/005-VSCodeExtensions/Blame.png)
_GitLens: In-line Blame_

![Blame Hover](/assets/img/005-VSCodeExtensions/Blame%20Hover.png)
_GitLens: On Hover Blame Details_

Commit graph and GitGraph. Commit graph, gives you a graph of commits, branches and merges over time, although I prefer Git Graph for this as it is simpler and more compact.

![Commit Graph](/assets/img/005-VSCodeExtensions/Commit%20Graph.png)
_GitLens: Commit Graph_

![Commit Graph](/assets/img/005-VSCodeExtensions/Git%20Graph.png)
_Git Graph: Git Graph_
