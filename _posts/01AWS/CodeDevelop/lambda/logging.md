


- ref
  - [logging](https://docs.aws.amazon.com/lambda/latest/dg/python-logging.html)

---


# AWS Lambda function logging in Python


AWS Lambda automatically monitors Lambda functions on your behalf and sends function metrics to Amazon CloudWatch.
- Lambda function **comes with a CloudWatch Logs log group and a log stream** for each instance of your function.
- The Lambda runtime environment sends details about each invocation to the log stream, and relays logs and other output from your function's code.

This page describes how to produce log output from your Lambda function's code, or access logs using the AWS Command Line Interface, the Lambda console, or the CloudWatch console.


## Creating function that returns logs
To output logs from your function code, you can use the print method, or any logging library that writes to stdout or stderr. The following example logs the values of environment variables and the event object.

```py
import os
def lambda_handler(event, context):
    print('## ENVIRONMENT VARIABLES')
    print(os.environ)
    print('## EVENT')
    print(event)

# START RequestId: 8f507cfc-xmpl-4697-b07a-ac58fc914c95 Version: $LATEST
# ## ENVIRONMENT VARIABLES
# environ({'AWS_LAMBDA_LOG_GROUP_NAME': '/aws/lambda/my-function', 'AWS_LAMBDA_LOG_STREAM_NAME': '2020/01/31/[$LATEST]3893xmpl7fac4485b47bb75b671a283c', 'AWS_LAMBDA_FUNCTION_NAME': 'my-function', ...})
# ## EVENT
# {'key': 'value'}
# END RequestId: 8f507cfc-xmpl-4697-b07a-ac58fc914c95
# REPORT RequestId: 8f507cfc-xmpl-4697-b07a-ac58fc914c95  Duration: 15.74 ms  Billed Duration: 16 ms Memory Size: 128 MB Max Memory Used: 56 MB  Init Duration: 130.49 ms
# XRAY TraceId: 1-5e34a614-10bdxmplf1fb44f07bc535a1   SegmentId: 07f5xmpl2d1f6f85 Sampled: true
```

The Python runtime logs the `START`, `END`, and `REPORT` lines for each invocation.
- The report line provides the following details.
- Report Log
  - RequestId – The unique request ID for the invocation.
  - Duration – The amount of time that your function's handler method spent processing the event.
  - Billed Duration – The amount of time billed for the invocation.
  - Memory Size – The amount of memory allocated to the function.
  - Max Memory Used – The amount of memory used by the function.
  - Init Duration – For the first request served, the amount of time it took the runtime to load the function and run code outside of the handler method.
  - XRAY TraceId – For traced requests, the AWS X-Ray trace ID.
  - SegmentId – For traced requests, the X-Ray segment ID.
  - Sampled – For traced requests, the sampling result.


```bash
aws lambda invoke \
    --function-name my-function out \
    --log-type Tail

aws lambda invoke \
    --function-name my-function out \
    --log-type Tail \
    --query 'LogResult' \
    --output text |  base64 -d

# START RequestId: 57f231fb-1730-4395-85cb-4f71bd2b87b8 Version: $LATEST
# "AWS_SESSION_TOKEN": "AgoJb3JpZ2luX2VjELj...", "_X_AMZN_TRACE_ID": "Root=1-5d02e5ca-f5792818b6fe8368e5b51d50;Parent=191db58857df8395;Sampled=0"",ask/lib:/opt/lib",
# END RequestId: 57f231fb-1730-4395-85cb-4f71bd2b87b8
# REPORT RequestId: 57f231fb-1730-4395-85cb-4f71bd2b87b8  Duration: 79.67 ms      Billed Duration: 80 ms         Memory Size: 128 MB     Max Memory Used: 73 MB
```



















































.
