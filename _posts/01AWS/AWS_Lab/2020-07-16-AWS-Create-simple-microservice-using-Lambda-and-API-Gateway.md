---
title: AWS Lab - Create a simple microservice using Lambda and API Gateway
date: 2020-07-16 11:11:11 -0400
categories: [01AWS, AWSLab]
tags: [AWS, Lab]
math: true
image:
---



# Create a simple microservice using Lambda and API Gateway

- [Create a simple microservice using Lambda and API Gateway](#create-a-simple-microservice-using-lambda-and-api-gateway)
  - [overall](#overall)
  - [create an API using Amazon API Gateway](#create-an-api-using-amazon-api-gateway)

---

## overall

- create a Lambda function, and an Amazon API Gateway endpoint to trigger that function.
- call the endpoint with any method (GET, POST, PATCH, etc.) to trigger Lambda function.
- When the endpoint is called, the entire request will be passed through to your Lambda function.
- function action depend on the method you call your endpoint with:
   - DELETE: delete an item from a DynamoDB table
   - GET: scan table and return all items
   - POST: Create an item
   - PUT: Update an item

## create an API using Amazon API Gateway

1. create an API
   - open the AWS Lambda console > Create Lambda function > Blueprint.
   - Enter `microservice` in the search bar.
   - Choose the `microservice-http-endpoint` blueprint and then choose `Configure`
   - Configure the following settings.
     - Name – lambda-microservice.
     - Role – Create a new role from one or more templates.
       - Lambda will create an execution role named lambda-microservice-role-d9awhq6v
       - with permission to upload logs to Amazon CloudWatch Logs.
     - Role name – lambda-apigateway-role.
     - Policy templates – Simple microservice permissions.
     - API – Create a new API.
     - Security – Open.
   - Create function.

Lambda creates a proxy resource named lambda-microservice under the API name you selected.
A proxy resource has an AWS_PROXY integration type and a catch-all method ANY. The AWS_PROXY integration type applies a default mapping template to pass through the entire request to the Lambda function and transforms the output from the Lambda function to HTTP responses. The ANY method defines the same integration setup for all the supported methods, including GET, POST, PATCH, DELETE and others.
Test sending an HTTPS request

1. use the console to test the Lambda function.

   1. In addition, you can run a curl command to test the end-to-end experience. That is, send an HTTPS request to your API method and have Amazon API Gateway invoke your Lambda function. In order to complete the steps, make sure you have created a DynamoDB table and named it "MyTable".
   2. To test the API
      1. choose Configure test event.
      2. Replace the existing text with the following
      3. Save and test.

```YAML
{
	"httpMethod": "GET",
	"queryStringParameters": {
	"TableName": "MyTable"
    }
}

result:
{
  "statusCode": "200",
  "body": "{\"Items\":[{\"name\":\"bob\",\"age\":\"26\"}],\"Count\":1,\"ScannedCount\":1}",
  "headers": {
    "Content-Type": "application/json"
  }
}

Request ID:
"a7631331-3bef-4e13-86d0-ec8900f6cbb3"
Function logs:
START RequestId: a7631331-3bef-4e13-86d0-ec8900f6cbb3 Version: $LATEST
END RequestId: a7631331-3bef-4e13-86d0-ec8900f6cbb3
REPORT RequestId: a7631331-3bef-4e13-86d0-ec8900f6cbb3	Duration: 121.95 ms	Billed Duration: 200 ms	Memory Size: 512 MB	Max Memory Used: 89 MB
```
