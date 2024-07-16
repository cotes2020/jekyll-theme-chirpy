---
title: Using Git Submodules to Distribute a Common Theme to Power BI Reports
description: Defining a Theme in a Git Submodule that can be injected into Reports in other repos upon deployment
author: duddy
date: 2024-07-16 16:00:00 +0000
categories: [Power BI, PBIR, PBIP, DevOps, Git]
tags: [pbir, bpir, theme, git, git submodule, power bi, devops]
pin: false
image:
  path: /assets/img/0007-PBIRTemplateDonation/beforeAfter.png
  alt: Updating a Report theme from a Donor Report in Git Submodule
---

As of a couple of months ago [PBIR](https://learn.microsoft.com/en-us/power-bi/developer/projects/projects-report#pbir-format) has been added to [PBIP](https://learn.microsoft.com/en-us/power-bi/developer/projects/projects-report). This new format brings a bunch of benefits. As a chance to explore the format more I've explored the concept for injecting a report Theme from a Donor Report, defined in a Git Submodule, into Recipient Reports.

## Recipient 
Lets start by creating a Recipient folder. We create our report and save the report in the PBIP format with PBIR enabled. We enable git and commit our changes.

```powershell
cd Recipient
git init
git add .
git commit -m "init"
```

```diff
+📁 Recipient
+├── 📁 recipient.Report
+│    ├── 📁 .pbi
+│    ├── 📁 definition
+│    │   ├── 📁 pages
+│    │   ├── 📄report.json
+│    │   └── 📄version.json
+│    ├── 📁 StaticResources
+│    │   └── 📁 SharedResources
+│    │       └── 📁 BaseThemes
+│    │           └── 📄CY24SU06.json
+│    ├── 📄 .platform
+│    └── 📄 definition.pbir
+├── 📁 recipient.SemanticModel
+└ .gitignore
```

## Donor
Now we'll create a Donor folder to host our donor report. We create a blank report, define a custom theme, and save the report in the PBIP format with PBIR enabled. I defined a full PBIP here rather than individual files to allow for easy updates via PBI Desktop.

```powershell
cd Donor
git init
git add .
git commit -m "init"
```

```diff
+📁 Donor
+├── 📁 donor.Report
+│    ├── 📁 .pbi
+│    ├── 📁 definition
+│    │   ├── 📁 pages
+│    │   ├── 📄report.json
+│    │   └── 📄version.json
+│    ├── 📁 StaticResources
+│    │   └── 📁 RegisteredResources
+│    │   │   └── 📄 donorTheme.json
+│    │   └── 📁 SharedResources
+│    │       ├── 📁 BaseThemes
+│    │       │   └── 📄 CY24SU06.json
+│    │       └── 📁 BaseThemes
+│    ├── 📄 .platform
+│    └── 📄 definition.pbir
+├── 📁 donor.SemanticModel
+└ .gitignore
```

I then pushed this repo to GitHub.

```powershell
git remote add origin https://github.com/EvaluationContext/Donor.git
git branch -M main
git push -u origin main
```

## Git Submodule
We now need to navigate back to our local Recipient folder and add register our remote Donor repo as a submodule. 

```powershell
cd Recipient
git submodule add https://github.com/EvaluationContext/Donor
```

```diff
 📁 Recipient
 ├── 📁 recipient.Report
 ├── 📁 recipient.SemanticModel
+├── 📁 Donor
+│   ├── 📁 donor.Report
+│   ├── 📁 donor.SemanticModel
+│   └ .gitignore
+├ .gitmodules
 └ .gitignore
```

Above you can see the Donor repo is nested within Recipient repo, plus a new .gitmodules file. This means the Recipient repo now has access to files in the Donor Repo. The point being any arbitrary number of Recipient repos can access the files defined once in Donor repo.

## Script to Donate Theme
We now need to add and update files in the Recipient Report, so that the Donor theme is applied.

### Required Changes
In order for the theme to be applied we need to:
- Copy `Donor/Recipient/recipient.Report/StaticResources/RegisteredResources/donorTheme.json` to `Donor/donor.Report/StaticResources/RegisteredResources/`

``` json
{
    "name": "donorTheme",
    "textClasses": {
        "label": {
            "color": "#0D9BDD",
            "fontFace": "'Segoe UI Light', wf_segoe-ui_light, helvetica, arial, sans-serif"
        }
    },
    "dataColors": [
        "#BF1212",
        "#B34545",
        "#4B1818",
        "#6B007B",
        "#E044A7",
        "#D9B300",
        "#D63550"
    ]
}
```

- Register the custom theme in `Donor/donor.Report/definition/report.json`

```diff
{
    "$schema": "https://developer.microsoft.com/json-schemas/fabric/item/report/definition/report/1.0.0/schema.json"
    ,"ThemeCollection": {
        "baseTheme": {},
+        "customTheme": {
+            "name": "donorTheme",
+            "reportVersionAtImport": "5.55",
+            "type": "RegisteredResources"
+        }
    },
    ...
    "resourcePackages": [
        {
            "name": "SharedResources",
            ...
        },
+        {
+            "name": "RegisteredResources",
+            "type": "RegisteredResources",
+            "items": [
+                {
+                    "name": "donorTheme.json",
+                    "path": "donorTheme.json",
+                    "type": "CustomTheme"
+                }
+            ]
+        }
    ]
}
```

### Manifest
We have hosted the entire Donor Report, and we might want to define the donation of other visuals assets in the Recipient repo. Therefore we want to create a file to specifies what assets we want to donate. I have a manifest file (`Recipient/.deploymentManifest.json`) that I am using for deployments, I extended it to add allow configuration of the required operation.

```json
{
    "repo": {},
    "items": {
        "semanticModels" : {},
        "reports": {
            "recipient.report": {
                "path": "recipient.report",
                "addItems": {
                    "path": "Donor/donor.report",
                    "visuals": {},
                    "images": {},
                    "theme": "Recipient/recipient.Report/StaticResources/RegisteredResources/donorTheme.json"
                }
            }
        }
    }
}
```

We can save this to the repo.

```diff
 📁 Recipient
 ├── 📁 recipient.Report
 ├── 📁 recipient.SemanticModel
 ├── 📁 Donor
+├ .deploymentManifest.json
 ├ .gitmodules
 └ .gitignore
```

### Script
We now need to read `.deploymentManifest.json` detect if a custom theme is specified and update the definition of the Recipient Report. As a proof of concept I'll assume there is no custom theme currently applied in the Recipient Report.

> I apologize in advance for this Powershell script, I'm sure there is a nicer way of writing this
{: .prompt-warning }

```powershell
$deploymentManifest = Get-Content '.deploymentManifest.json' | Out-String | ConvertFrom-Json -AsHashtable

foreach ($recipientReport in $deploymentManifest.items.reports.GetEnumerator()) {
    foreach($donorReport in $recipientReport.Value.addItems.GetEnumerator()) {

        $recipientPath = $recipientReport.Value.path
        $donorPath = $donorReport.Value.path

        Write-Host "Donating Files"
        $theme = $donorReport.Value.theme
        $donorPath = "$pwd/$donorPath/StaticResources/RegisteredResources/$theme"
        $recipientFolderPath = "$pwd/$recipientPath/StaticResources/RegisteredResources"
        $recipientPath = "$recipientFolderPath/$theme"
        if(-Not (Test-Path $recipientFolderPath)) {New-Item -ItemType "directory" -Path $recipientFolderPath}
        Copy-Item -Path $donorPath -Destination $recipientPath

        Write-Host "Registering Files"
        $recipientReportjson = Get-Content -Path "$pwd/$recipientPath/definition/report.json" | ConvertFrom-Json -AsHashtable
        $themeCollection = @{
            name = $theme;
            reportVersionAtImport = "5.55";
            type = "RegisteredResources"
        }

        $resourcePackages = @{
            name  = "RegisteredResources";
            type  = "RegisteredResources";
            items = @(
                @{
                    name = $theme;
                    path = $theme;
                    type = "CustomTheme"
                }
            )
        }

        $recipientReportjson.themeCollection["customTheme"] = $themeCollection
        $recipientReportjson.resourcePackages += $resourcePackages
        $updatedFile = $recipientReportjson | ConvertTo-Json -Depth 10
        Set-Content -Path "$pwd/$recipientPath/definition/report.json" -Value $updatedFile
```

Running the script results in the following results in the following.

```diff
 📁 Recipient
 ├── 📁 recipient.Report
 │    ├── 📁 .pbi
 │    ├── 📁 definition
 │    │   ├── 📁 pages
-│    │   ├── 📄report.json
+│    │   ├── 📄report.json
 │    │   └── 📄version.json
 │    ├── 📁 StaticResources
+│    │   └── 📁 RegisteredResources
+│    │   │   └── 📄 donorTheme.json
 │    │   └── 📁 SharedResources
 │    ├── 📄 .platform
 │    └── 📄 definition.pbir
 ├── 📁 recipient.SemanticModel
 ├── 📁 Donor
 │   ├── 📁 donor.Report
 │   ├── 📁 donor.SemanticModel
 │   └ .gitignore
 ├ .deploymentManifest.json
 ├ .gitmodules
 └ .gitignore
```

When we open the file we can see the theme has changed.

![Theme application](/assets/img/0007-PBIRTemplateDonation/beforeAfter.png)

## Conclusion
In regards to resources it would be nice if their presence would register them as to use to avoid having to register them in report.json. Regardless, this pattern could be quite useful in defining a theme, allow propagation of a standard from a single repo to many reports. This version while rough introduces the concept.