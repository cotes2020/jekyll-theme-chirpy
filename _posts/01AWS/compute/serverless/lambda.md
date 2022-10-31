

- [the Different Ways to Invoke Lambda Functions](#the-different-ways-to-invoke-lambda-functions)
  - [Synchronous Invokes](#synchronous-invokes)
  - [Asynchronous Invokes](#asynchronous-invokes)
  - [Poll-Based Invokes](#poll-based-invokes)

In our first post, we talked about general design patterns to enable massive scale with serverless applications.




## the Different Ways to Invoke Lambda Functions

https://aws.amazon.com/blogs/architecture/understanding-the-different-ways-to-invoke-lambda-functions/

![Screen-Shot-2019-06-27-at-2.23.51-PM-1024x510](https://i.imgur.com/AoCgNOQ.png)


### Synchronous Invokes

- the most straight forward way to invoke the Lambda functions.
- the functions execute immediately whe perform the Lambda Invoke API call.
- This can be accomplished through a variety of options, including using the CLI or any of the supported SDKs.

exampl:

synchronous invoke using the CLI:

```bash
aws lambda invoke \
    —function-name MyLambdaFunction \
    —invocation-type RequestResponse \
    —payload  “[JSON string here]”
```

> The Invocation-type flag specifies a value of “RequestResponse”.
> This instructs AWS to execute the Lambda function and wait for the function to complete.

- When perform a synchronous invoke, you are responsible for checking the response and determining if there was an error and if you should retry the invoke.

list of services that invoke Lambda functions synchronously:
- Elastic Load Balancing (Application Load Balancer)
- Amazon Cognito
- Amazon Lex
- Amazon Alexa
- Amazon API Gateway
- Amazon CloudFront (Lambda@Edge)
- Amazon Kinesis Data Firehose



### Asynchronous Invokes

Here is an example of an asynchronous invoke using the CLI:

```bash
aws lambda invoke \
    —function-name MyLambdaFunction \
    —invocation-type Event \
    —payload  “[JSON string here]”
```

> Notice, the Invocation-type flag specifies “Event.” If the function returns an error, AWS will automatically retry the invoke twice, for a total of three invocations.

Here is a list of services that invoke Lambda functions asynchronously:
- Amazon S3
- Amazon SNS
- Amazon Simple Email Service
- AWS CloudFormation
- Amazon CloudWatch Logs
- Amazon CloudWatch Events
- AWS CodeCommit
- AWS Config

- Asynchronous invokes place the invoke request in Lambda service queue and we process the requests as they arrive. You should use AWS X-Ray to review how long the request spent in the service queue by checking the “dwell time” segment.



### Poll-Based Invokes
- This invocation model is designed to allow you to integrate with AWS Stream and Queue based services with no code or server management.
- Lambda will poll the following services on the behalf, retrieve records, and invoke the functions.

The following are supported services:
- Amazon Kinesis
- Amazon SQS
- Amazon DynamoDB Streams


- AWS will manage the poller on the behalf and perform Synchronous invokes of the function with this type of integration.
- The retry behavior for this model is based on data expiration in the data source.
- For example, Kinesis Data streams store records for 24 hours by default (up to 168 hours). The specific details of each integration are linked above.
