---
title: Uploading Files to S3 with Presigned URL
author: Pig
date: 2022-05-15 14:18:00 +0800
categories: [Backend]
tags: [Catty]
layout: post
current: post
class: post-template
subclass: 'post'
navigation: True
cover: assets/img/post_images/catty_cover2.png
---

AWS S3 is a great place to keep online service assets such as images or large files. The AWS SDK provides the [putObject](https://docs.aws.amazon.com/AWSJavaScriptSDK/latest/AWS/S3.html#putObject-property) function to upload binaries to S3. This comes in handy for uploading files from the server, but to do this on frontend application, the S3 bucket either needs to have complete public access(obviously bad for security), or the frontend application needs to some secret aws credentials that allows access to S3, which comes with a risk of the secret keys being exposed.

&nbsp;
## Presigned URL
&nbsp;

To grant temporary access to the frontend application, AWS SDK provides the [getSignedUrl](https://docs.aws.amazon.com/AWSJavaScriptSDK/latest/AWS/S3.html#getSignedUrl-property) function which generates a temporary URL to allow operations on the bucket. This can be used to create an API for performing put or get operations from the frontend application without concerning about AWS credentials.

The handler for generating presigned url request was implemented like the following

```jsx
import { S3 } from "aws-sdk";
const s3 = S3({endpoint: process.env.BUCKET_URL});

// NOTE: Change this value to adjust the signed URL's expiration
const URL_EXPIRATION_SECONDS = 10800;
// API response type
interface S3UploadResponse {
    filename: string;
    viewUrl: string;
    uploadUrl: string;
}
// API request type
interface S3UploadRequest {
    contentType: string;
    key: string;
}

export class S3Handler {
    filename: string;
    viewUrl: string;
    uploadUrl: string;

    static async getUploadUrl(_data: any): Promise<S3UploadResponse> {
        const request = JSON.parse(_data) as S3UploadRequest;

        const s3Params = {
            Bucket: process.env.UPLOAD_BUCKET_PATH, //name of bucket to upload
            Key: request.key, //name of asset
            Expires: URL_EXPIRATION_SECONDS, //url lifetime
            ContentType: request.contentType, //set content type to allow previewing
            ACL: 'public-read' //change this to suit the needs
        }
        console.log('S3 Params: ', s3Params);
        const uploadURL = await s3.getSignedUrlPromise('putObject', s3Params);
        console.log('uploadURL: ', uploadURL);
        console.log({uploadURL: uploadURL});
        return {
            filename: request.key,
            viewUrl: process.env.BUCKET_URL + "upload/" + request.key,
            uploadUrl: uploadURL
        } as S3UploadResponse;
    }
}
```

Only the url for uploading assets was needed for my use case, so generating url for viewing was not implemented, but this can be done trivially by changing putObject to getObject in the parameters.

&nbsp;
## Creating the API
&nbsp;

Now that the s3 handler class is ready, the function to handle the request can be written.

```jsx
export async function handleS3Event(event, context) {
    console.log("event received", event);
    try{
        switch (event.httpMethod) {
            case 'POST':
                return response (await S3Handler.getUploadUrl(event.body), 200);
        }
    } catch(e) {
        return response(e, 400);
    }
}
```

And update serverless.yml file as well to reflect the addition of a new endpoint.

```jsx
iam:
    role: arn:aws:iam::of::role::to:access::s3

s3Event:
    handler: handler.handleS3Event
    events:
      - http:
          path: /uploads
          method: post
          cors: true
```

&nbsp;
## Using the URL to upload from the frontend
&nbsp;