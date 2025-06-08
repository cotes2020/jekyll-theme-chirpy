---
title: "Building Medallion Architecture in Azure Databricks"
date: 2025-06-08
categories: [Azure, Data Engineering]
tags: [Databricks, ADF, Delta Lake, PySpark, Unity Catalog, Power BI]
---

In this blog post, I share how we designed and implemented a scalable, governed **Medallion Architecture (Bronze, Silver, Gold)** using **Azure Databricks** for a modern retail analytics platform.

---

### 🚀 Project Overview

The client had legacy systems with on-prem SQL Server and Excel-based reports. The goal was to modernize the stack using a **cloud-native Lakehouse platform** with automated ingestion, governed data layers, and Power BI dashboards.

---

### 🏗️ Architecture Summary

We adopted a **Medallion Architecture** using Delta Lake:

Source Systems → ADF → Bronze → Silver → Gold → Power BI

Each layer served a specific purpose:
- **Bronze:** Raw ingestion from source
- **Silver:** Cleaned, transformed, enriched data
- **Gold:** Aggregated, business-consumable models

---

### 🔄 Ingestion & Raw Layer (Bronze)

- Used **Azure Data Factory (ADF)** to ingest data from CSVs, databases to **ADLS Gen2**.
- Loaded raw data into **Bronze Delta tables** using `multiLine=True` for JSON and schema inference.
- Stored with partitioning and metadata registered in Unity Catalog.

---

### ✨ Transformation Layer (Silver)

- Cleaned nulls, fixed schema mismatches using **PySpark**.
- Joined lookup tables and normalized data types.
- Applied **Z-Ordering** and `OPTIMIZE` for performance.
- Registered tables with Unity Catalog for auditing.

---

### 📊 Aggregation Layer (Gold)

- Built fact/dimension models:  
  `store_sales_fact`, `online_sales_fact`, `customer`, `product`, `store`, `time`
- Calculated KPIs like:
  ```python
  total_amount = quantity * unit_price
  ```
## Designed for Power BI to directly consume with low-latency queries.

### 🔐 Governance with Unity Catalog
  - Created catalogs/schemas per domain (e.g., retail.bronze, retail.gold)

  - Applied row-level and column-level ACLs

  - Enabled data lineage and audit via built-in logging

### ⚙️ Automation with ADF & CI/CD
  - Orchestrated workflows using ADF pipelines

  - Each layer ran in isolation with checkpoints and retry logic

  - Setup Azure DevOps CI/CD to deploy:

  - Notebooks

  - Job configs

  - Catalog/table metadata

### 📈 Reporting with Power BI
  - Published Power BI dashboards on top of Gold tables

  - Used Azure SQL Endpoint for live connections

  - Dashboards included:

  - Product performance

  - Store-wise revenue

  - Online vs offline trends

### ✅ Impact & Results
  - ⏱ Reduced processing time by 85%

  - 🧠 Improved data quality and traceability

  - 🔐 Centralized access controls via Unity Catalog

  - 📊 Enabled self-serve BI with secured Gold layer

  - 🧠 What We Learned
    - Delta Lake with Z-Ordering = Huge performance gain

    - Unity Catalog simplifies security and audit at scale

    - CI/CD saves time when promoting notebooks and configs

    - Modular layers make debugging and lineage easier

### 📌 What's Next?
  - Event-driven ingestion with Event Hubs

  - Real-time ML pipelines for fraud/anomaly detection

  - Data contracts between data producers and consumers

#### Thanks for reading!
  - Got questions? Reach out on [LinkedIn](https://www.linkedin.com/in/lokesh-manne)
.