---
layout: post
title: Zip Large Files With PowerShell
date: 2023-01-25 23:35
comments: false
category: powershell
author: ebmarquez
tags:
- PowerShell
- dotnet
- code
summary: Notes on how to use powershell to zip files over 2G
---
<br><br><br>
This is a quick note for anybody looking for a solution to zip large files over 2G.  In PowerShell, you can typically use the `Expand-Archive` or `Compress-Archive` to zip files. This method typically works except when the file or files you need to zip is zip happens to be larger than 2 Gigs. To get around this problem, the dotnet library `System.IO.Compression` needs to be used.

[ZipFile.Open Method (System.IO.Compression) \| Microsoft Learn](https://learn.microsoft.com/en-us/dotnet/api/system.io.compression.zipfile.open?view=net-7.0#system-io-compression-zipfile-open)

**Compress a Zip package**

```powershell
Add-Type -Assembly System.IO.Compression.FileSystem
$ZIPFileName = "c:\ZipTarget\master.zip"
$folderName = "C:\ExpandLocation\largeFile.bin"

[Reflection.Assembly]::LoadWithPartialName('System.IO.Compression.FileSystem') | Out-Null
$zip = [System.IO.Compression.ZipFile]::Open($ZIPFileName,'Create')
$fileName = [System.IO.Path]::GetFileName($FolderName)
[System.IO.Compression.ZipFileExtensions]::CreateEntryFromFile($zip,$folderName,$fileName,'Optimal') | Out-Null
$zip.Dispose()
```

**Expand a zip package**

```powershell
Add-Type -Assembly System.IO.Compression.FileSystem
$ZIPFileName = "c:\ZipTarget\master.zip"
$FolderName = "c:\ExpandLocation\"
$zip = [System.IO.Compression.ZipFile]::Open($ZIPFileName,'Read')
[Reflection.Assembly]::LoadWithPartialName('System.IO.Compression.FileSystem') | Out-Null
[System.IO.Compression.ZipFileExtensions]::ExtractToDirectory($zip,$FolderName) | Out-Null
```

In this article it describes diffent modes that can be run while defalting a zip package.

[ZipArchiveMode Enum (System.IO.Compression) | Microsoft Learn](https://learn.microsoft.com/en-us/dotnet/api/system.io.compression.ziparchivemode?view=net-7.0)<br>
Multiple Mode are available when calling the library.

- Create
  Create used to generate the zip package and does not have a 2G limit.
- Read
  Read will allow the extraction process to bypass the 2G buffer size and write the stream to disk.
- Update
  Update has a limited buffer size of 2G this option should not be utilized.
