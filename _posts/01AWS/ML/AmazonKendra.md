---
title: AWS - ML - Amazon Kendra
date: 2021-04-04 11:11:11 -0400
categories: [01AWS, ML]
tags: [AWS, ML]
toc: true
image:
---

- [ML - Amazon Kendra](#ml---amazon-kendra)
  - [overview](#overview)
  - [Benefits of Amazon Kendra](#benefits-of-amazon-kendra)
  - [setup](#setup)

- ref
  - [Amazon Kendra: Transform the Way You Search and Interact with Enterprise Data Using AI](https://www.youtube.com/watch?v=eWg9xaC4vNw)


---

# ML - Amazon Kendra

![Screen Shot 2021-06-30 at 3.15.21 PM](https://i.imgur.com/30o3OQV.jpg)

![Screen Shot 2021-06-30 at 3.17.40 PM](https://i.imgur.com/LLtYwpy.png)

![Screen Shot 2021-06-30 at 3.20.56 PM](https://i.imgur.com/zxYMBeg.png)

![Screen Shot 2021-06-30 at 3.25.39 PM](https://i.imgur.com/1Lxb6SG.jpg)


## overview

- a highly accurate intelligent search service
- enables the users to search unstructured data using natural language.
- It returns specific answers to questions, giving users an experience that's close to interacting with a human expert.
- highly scalable and capable of meeting performance demands,
- tightly integrated with other AWS services such as Amazon S3 and Amazon Lex
- offers enterprise-grade security.

Amazon Kendra users can ask the following types of questions, or queries:
- **Factoid questions**
  - Simple `who, what, when, or where questions`
  - such as Who is on duty today? or Where is the nearest service center to me? Factoid questions have fact-based answers that can be returned in the form of a single word or phrase. The precise answer, however, must be explicitly stated in the ingested text content.
- **Descriptive questions**
  - Questions whose answer could be a sentence, passage, or an entire document.
  - For example, How do I connect my Echo Plus to my network? or How do I get tax benefits for lower income families?.
- **Keyword searches**
  - Questions where the intent and scope are not clear.
  - For example, keynote address. As 'address' can often have several meanings, Amazon Kendra can infer the user's intent behind the search query to return relevant information aligned with the user's intended meaning. Amazon Kendra uses deep learning models to handle this kind of query.

---


## Benefits of Amazon Kendra

Amazon Kendra has the following benefits:

- Accuracy
  - Unlike traditional search services that use keyword searches where results are based on basic keyword matching and ranking,
  - Amazon Kendra attempts to understand the content, the user context, and the question. Amazon Kendra searches across your data and goes beyond traditional search to return the most relevant word, snippet, or document for your query.
  - Amazon Kendra uses machine learning to improve search results over time.
- Simplicity
  - provides a console and API for managing the document to search.
  - can use a simple search API to integrate Amazon Kendra into lient applications, such as websites or mobile applications.
- Connectivity
  - Amazon Kendra can connect to third-party data sources to provide search across documents managed in different environments.
- User Access Control
  - Amazon Kendra delivers highly secure enterprise search for your search applications.
  - Your search results reflect the security model of your organization.
  - Customers are responsible for authenticating and authorizing users to gain access to their search application.


---


## setup

Amazon Kendra provides an interface for indexing and searching documents.

- use Amazon Kendra to create an **updatable index** of documents of a variety of types
  - including plain text, HTML files, Microsoft Word documents, Microsoft PowerPoint presentations, and PDF files.
- It has a search API that you can use from a variety of client applications, such as websites or mobile applications.

Amazon Kendra integrates with other services.
- For example, you can power Amazon Lex chat bots with Amazon Kendra search to provide answers to users’ questions.
- You can use Amazon S3 bucket as a data source for your Amazon Kendra index.
- set up AWS Identity and Access Management to control access to Amazon Kendra resources.

Amazon Kendra has the following components:
- The **index**
  - provides a search API for client queries.
  - create the index from source documents.
- A **source repository**
  - contains the documents to index.
- A **data source**
  - syncs the documents in your source repositories to an Amazon Kendra index. You can automatically synchronize a data source with an Amazon Kendra index so that new, updated, and deleted files in the source repository are updated in the index.
- A document addition API
  - adds documents directly to the index.


To manage indexes and data sources, you can use the Amazon Kendra console or the API. You can create, update, and delete indexes. Deleting an index deletes all data sources and permanently deletes all of your document information from Amazon Kendra.













.
