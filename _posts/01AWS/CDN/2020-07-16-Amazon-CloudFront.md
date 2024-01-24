---
title: AWS Lab - AWS CloudFront
date: 2020-07-16 11:11:11 -0400
categories: [01AWS, CDN]
tags: [AWS, Lab, CloudFront]
math: true
image:
---


# CloudFront

- [CloudFront](#cloudfront)
  - [create an Amazon CloudFront distribution](#create-an-amazon-cloudfront-distribution)
  - [Task 1: Store a Publicly Accessible Image File in an Amazon S3 Bucket](#task-1-store-a-publicly-accessible-image-file-in-an-amazon-s3-bucket)
  - [Task 2: Create an Amazon CloudFront Web Distribution](#task-2-create-an-amazon-cloudfront-web-distribution)
  - [Task 3: Create a Link to Your Object](#task-3-create-a-link-to-your-object)
  - [Task 4: Delete Your Amazon CloudFront Distribution](#task-4-delete-your-amazon-cloudfront-distribution)

---

## create an Amazon CloudFront distribution

create an Amazon CloudFront distribution that will use a CloudFront domain name in the url to distribute a publicly accessible image file stored in an Amazon S3 bucket.
- Create a new Amazon CloudFront distribution
- Use Amazon CloudFront distribution to serve an image file
- Delete Amazon CloudFront distribution when it is no longer required


## Task 1: Store a Publicly Accessible Image File in an Amazon S3 Bucket

1. AWS Management Console > Services > S3 > Create bucket
2. Permissions tab > Block public access > Uncheck the Block all public access. All five boxes should now be unchecked
3. Upload file > Manage public permissions > Grant public read access to this object(s)
4. file > Object URL


## Task 2: Create an Amazon CloudFront Web Distribution

1. AWS Management Console > Services > CloudFront.
2. Create Distribution
   - delivery method: Web section
   - Origin Domain Name: `the S3 bucket create`
   - Scroll to the bottom of the page, then click Create Distribution


## Task 3: Create a Link to Your Object

`myimage.html`
- DOMAIN: Amazon CloudFront Domain Name
- OBJECT: name of the file uploaded to Amazon S3 bucket

```html
<html>
<head>My CloudFront Test</head>
<body>
<p>My text content goes here.</p>
<p><img src="https://DOMAIN/OBJECT" alt="my test image" /></p>
</body>
</html>
```

## Task 4: Delete Your Amazon CloudFront Distribution

1. Disable > delete













.
