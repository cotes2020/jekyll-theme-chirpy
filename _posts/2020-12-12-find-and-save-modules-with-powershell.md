---
layout: post
title: Find And Save Modules with PowerShell
date: 2020-12-12 15:37
comments: true
category: powershell
author: ebmarquez
tags: [windows, powershell]
summary: Using PowerShell Find and Save Modules.
---

Today I wanted to discuss a couple of PowerShell tools I needed to utilize this week, Find-Module and Save-Module.  These are powerful tools if you’re working with a restricted environment and don’t have direct access to PSGallery.  Find-Module and Save-Module are helpful tools and can identify required modules and save them to a temporary location.  These modules are built-in PowerShell method that allows you to call into PSGallery or a specific repository.  In my case, I only need access to PSGallery and the system I’m working with has internet access.  Let’s get started.

**Find-Module**

Starting with Find-Module, it can call into a repository like PSGallery to query modules that have a matching name.  The names don’t need to be exact when using a wildcard with the name. 

```
❯ Find-Module -Name *pest*

Version              Name                                Repository           Description
-------              ----                                ----------           -----------
5.1.0                Pester                              PSGallery            Pester provides a framework for running BDD style Tests to e…
0.3.1                PesterMatchArray                    PSGallery            Array assertions for Pester
1.6.0                Format-Pester                       PSGallery            Powershell module for documenting Pester s results.
0.0.4                PestWatch                           PSGallery            Adds File watcher support to Pester
1.0.0                PSPesterTest                        PSGallery            Provides pre-defined pester tests for PowerShell scripts and…
0.3.0                PesterMatchHashtable                PSGallery            Hashtable assertions for Pester
0.0.4                PesterInfrastructureTests           PSGallery            Small module that helps with quickly testing Active Director…
0.0.6                PesterHelpers                       PSGallery            PesterHelpers contains helper functions to help users move a…
0.2.0                pspestertesthelpers                 PSGallery            A collection of helper functions to use with Pester tests.
```

Once the module is discovered, additional parameters can be used to identify more details of the module in question.  -AllVersions will expose all the published versions of a module that are in the repository.

```
❯ Find-Module -Name Pester -AllVersions

Version              Name                                Repository           Description
-------              ----                                ----------           -----------
5.1.0                Pester                              PSGallery            Pester provides a framework for running BDD style Tests to e…
5.0.4                Pester                              PSGallery            Pester provides a framework for running BDD style Tests to e…
5.0.3                Pester                              PSGallery            Pester provides a framework for running BDD style Tests to e…
5.0.2                Pester                              PSGallery            Pester provides a framework for running BDD style Tests to e…
5.0.1                Pester                              PSGallery            Pester provides a framework for running BDD style Tests to e…
5.0.0                Pester                              PSGallery            Pester provides a framework for running BDD style Tests to e…
4.10.1               Pester                              PSGallery            Pester provides a framework for running BDD style Tests to e…
4.10.0               Pester                              PSGallery            Pester provides a framework for running BDD style Tests to e…
```

From the full list, several details can be teased out from the list of available modules. In this case some of these modules are considered pre-release.  Find-Module does offer a way to discover versions that the publisher has listed as prerelease.  -AllowPrerelease will bubble up those versions from the list.

```
❯ Find-Module -Name Pester -AllowPrerelease

Version              Name                                Repository           Description
-------              ----                                ----------           -----------
5.1.0                Pester                              PSGallery            Pester provides a framework for running BDD style Tests to e…
```

In my case, I’m looking for a specific version that was required on my system.  Using -RequiredVersion plus the version number will show the details of the module.  In this example, the module details are listed when it’s piped to Format-List.

```
❯ Find-Module -Name pester -RequiredVersion 4.9.0 |Format-List

Name                       : Pester
Version                    : 4.9.0
Type                       : Module
Description                : Pester provides a framework for running BDD style Tests to execute and validate PowerShell commands inside of
                             PowerShell and offers a powerful set of Mocking Functions that allow tests to mimic and mock the
                             functionality of any command inside of a piece of PowerShell code being tested. Pester tests can execute any
                             command or script that is accessible to a pester test file. This can include functions, Cmdlets, Modules and
                             scripts. Pester can be run in ad hoc style in a console or it can be integrated into the Build scripts of a
                             Continuous Integration system.
Author                     : Pester Team
CompanyName                : {dlwyatt, nohwnd}
Copyright                  : Copyright (c) 2019 by Pester Team, licensed under Apache 2.0 License.
PublishedDate              : 9/8/2019 8:56:49 AM
InstalledDate              :
UpdatedDate                :
LicenseUri                 : https://www.apache.org/licenses/LICENSE-2.0.html
ProjectUri                 : https://github.com/Pester/Pester
IconUri                    : https://raw.githubusercontent.com/pester/Pester/master/images/pester.PNG
Tags                       : {powershell, unit_testing, bdd, tdd…}
Includes                   : {DscResource, Workflow, Function, RoleCapability…}
PowerShellGetFormatVersion :
ReleaseNotes               : https://github.com/pester/Pester/releases/tag/4.9.0
Dependencies               : {}
RepositorySourceLocation   : https://www.powershellgallery.com/api/v2/
Repository                 : PSGallery
PackageManagementProvider  : NuGet
```

Now that the Module has been discovered, how to download it for easy transfer to my system?

**Save-Module**

The Save-Module was created take module package and place it in a specific location so it can be transferred to its ultimate destination.   Like Find-Module this method can narrowly download a specific version it that is needed.  Is has similar parameters like MaximumVersion, MinimumVersion and RequiredVersion.  One of the required parameters is the -Path.  This will tell the method where to deposit the downloaded package. Both examples will accomplish the same job.

```
Find-Module -Name pester -RequiredVersion 4.9.0 | Save-Module -Path C:\temp\foo

Save-Module -Path c:\temp\foo -RequiredVersion 4.9.0 -Name pester
```

Now that the package is downloaded, what to do with it?  The c:\temp\foo\Pester can now be transferred as is to the system or systems that require the version.  

```
    Directory: C:\temp\foo\Pester\4.9.0

Mode                 LastWriteTime         Length Name
----                 -------------         ------ ----
d----           12/8/2020 11:18 PM                bin
d----           12/8/2020 11:18 PM                Dependencies
d----           12/8/2020 11:18 PM                en-US
d----           12/8/2020 11:18 PM                Functions
d----           12/8/2020 11:18 PM                lib
d----           12/8/2020 11:18 PM                Snippets
-a---            9/8/2019  8:19 AM           4475 junit_schema_4.xsd
-a---          12/11/2018  1:29 PM            611 LICENSE
-a---            5/1/2019  9:36 AM           5933 nunit_schema_2.5.xsd
-a---            9/8/2019  8:53 AM          15364 Pester.psd1
-a---            9/8/2019  8:53 AM          79290 Pester.psm1
```

If the tool is needed as a system wide package, the module should be placed in the PSModulePath.  $ENV:PSModulePath will show all the available path locations where modules are installed from. There are two different options where the package can be placed, `~\Documents\WindowsPowerShell\Modules` or in `c:\Program Files\WindowsPowerShell\Modules directories`.  The user Documents folder is typically reserved for user defined modules and if a module is needed for wider system access module are typically placed in the Program Files.  There is a third option, `c:\windows\System32\WindowsPowerShell\v1.0\Modules`.   This location is reserved for windows, it’s not advised to utilize this location.

This post walked through a couple of helpful tools to discover some modules with Find-Module and save the Module using Save-Module. A few examples were provided to show various ways to use the PowerShell methods and where to place the package after it has been downloaded.  Happy Power Shelling. 
