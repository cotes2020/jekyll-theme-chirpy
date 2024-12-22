---
title: Keeping Power BI Incremental Refresh Up To Date
description: A alternate solution to a solution by Nikolaos Christoforidis for keeping local Incremental Refresh Reports up to date
author: duddy
date: 2024-12-22 18:00:00 +0000
categories: [Powerquery, Incremental Refresh]
tags: [powerquery, incrementalrefresh]
pin: false
image:
  path: /assets/img/0015-IncRefresh/post.png
  alt: post
---
 
I was inspired to write this blog post after seeing a linkedin [post](https://www.linkedin.com/feed/update/urn:li:activity:7276526605563252736/) from [Nikolaos Christoforidis](https://www.linkedin.com/in/nikolaos-christoforidis-2678111b4/). He was trying to address a problem when developing models with Incremental Refresh, whereby the parameter required for Incremental Refresh can be a headache when developing locally, as you have to keep redefining `RangeStart` and `RangeEnd` to get up-to-date data. He suggested using dynamic M parameters, [Pavel A.](https://www.linkedin.com/in/paveladamcr/) mentioned these may fail in some scenarios, e.g. Dataflows GEN1. This post proposes another method to solve this problem.

## Power BI Incremental Refresh

A SSAS Tabular table is made up of one or more [partitions](https://learn.microsoft.com/en-us/analysis-services/tabular-models/partitions-ssas-tabular?view=asallproducts-allversions). When you first create a table in Power BI it will have one partition. You can create additional partitions with [TMSL](https://learn.microsoft.com/en-us/analysis-services/tmsl/tabular-model-scripting-language-tmsl-reference?view=asallproducts-allversions) and the [XMLA endpoint](https://learn.microsoft.com/en-us/power-bi/enterprise/service-premium-connect-tools), or [Microsoft.AnalysisServices.Tabular](https://learn.microsoft.com/en-us/dotnet/api/microsoft.analysisservices.tabular?view=analysisservices-dotnet) client library, or tool such as [Tabular Editor](https://github.com/TabularEditor/TabularEditor) and [SQL Server Management Studio (SSMS)](https://learn.microsoft.com/en-us/sql/ssms/sql-server-management-studio-ssms?view=sql-server-ver16) that leverage these. These additional partitions will have the same schema, but be filtered to include different data (i.e. 2020, 2021, 2022). Each partition can then be processed (Refreshed) independent of other partitions and in [parallel](https://blog.crossjoin.co.uk/2021/06/27/increasing-refresh-parallelism-and-performance-in-power-bi-premium/).

Normally with partitions you are responsible for creating and managing them. [Incremental Refresh](https://learn.microsoft.com/en-us/power-bi/connect-data/incremental-refresh-overview) is a feature in the Power BI service that automates partition creation and management. It is configured by the creation of a Refresh Policy for a table, defining rolling time windows for hot and cold data, and data eviction. Tabular Editor has some very good [docs](https://docs.tabulareditor.com/te3/tutorials/incremental-refresh/incremental-refresh-about.html?tabs=filterstep%2Cimport) on configuration and gotchas for this feature.

## Problem/Solution

When you are working on a Power BI file locally you will have static `RangeStart` and `RangeEnd` parameters filtering a table to a given data range.

```fs
//RangeStart
#datetime(2022, 12, 01, 0, 0, 0) meta [IsParameterQuery = true, IsParameterQueryRequired = true, Type = type datetime]
```

```fs
//RangeEnd
#datetime(2022, 12, 31, 0, 0, 0) meta [IsParameterQuery = true, IsParameterQueryRequired = true, Type = type datetime]
```

```fs
// table
let
    source = .....,
    incrementalRefresh = Table.SelectRows(source, each [date] >= #"RangeStart" and [date] < #"RangeEnd")
in 
    incrementalRefresh
```

If you haven't worked on the Report in a while, these dates might be from months ago, but you want to look at new data. You can update `RangeStart` and `RangeEnd` with new dates, but there is another solution. [Nikolaos Christoforidis's](https://www.linkedin.com/in/nikolaos-christoforidis-2678111b4/) solution is to have dynamic parameters that provides the previous 3 month of data. 

```fs
//RangeStart
DateTime.From(Date.StartOfMonth(Date.AddMonths(DateTime.LocalNow(), -3))) meta [IsParameterQuery = true, IsParameterQueryRequired = true, Type = type datetime]
```

```fs
//RangeEnd
DateTime.From(Date.EndOfMonth(DateTime.LocalNow())) meta [IsParameterQuery = true, IsParameterQueryRequired = true, Type = type datetime]
```

But this has the issues with dataflow Dataflows GEN1 mentioned before. Instead we are able keep `RangeStart` and `RangeEnd` static, and move the logic to return the most recent 3 month to the table. We can leverage the fact that the Power BI service will hijack the `RangeStart` and `RangeEnd` parameters to filter the data for a data range for each partition, and that we only want the most recent 3 months locally. Firstly we want to identify a date that precedes the window of our Incremental Refresh window. For example if we are archiving 2 years of data we could pick `01-01-2020`. In the table definition we can look out for this date and filter to the last 3 month, else enact the regular incremental refresh pattern.

```fs
// table
let
    source = .....,
    threeMonthsAgo = DateTime.From(Date.StartOfMonth(Date.AddMonths(DateTime.LocalNow(), -3))),
    now = DateTime.From(Date.EndOfMonth(DateTime.LocalNow())),
    lastThreemonths = Table.SelectRows(data, each [date] >= threeMonthsAgo and [date] < now),
    incrementalRefresh = Table.SelectRows(#"Changed Type", each [Date] >= #"RangeStart" and [Date] < #"RangeEnd"),
    selectPath = if #"RangeStart" = DateTime.FromText("01/01/2020 00:00:00") then lastThreemonths else incrementalRefresh
in 
    selectPath
```

Now if we set RangeStart to `01/01/2020 00:00:00`, this will mean we will return lastThreemonths. In the service based on our Refresh Policy, this value will not be injected by the service and the incrementalRefresh path will be taken instead.

## Conclusion

This pattern of using a value in parameter to provide a behavior quite a useful pattern in the DevOps space as well. For example, we can define a `TopN` parameter, this could be set locally to 1000, this means locally locally or in a deployment to dev environment we to test the schema and connectivity without performing a large refresh. As part of the DevOps process when deploying to to a UAT or Prod environment we can use the [Update Parameters In Group](https://learn.microsoft.com/en-us/rest/api/power-bi/datasets/update-parameters-in-group) REST API, or script the update the PBIP files in the deployment to overwrite the parameters to include all the data.