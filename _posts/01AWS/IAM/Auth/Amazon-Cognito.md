
[toc]

# Building fine-grained authorization for API using Amazon Cognito, API Gateway, and IAM

- ref
  - [Building fine-grained authorization using Amazon Cognito, API Gateway, and IAM](https://aws.amazon.com/blogs/security/building-fine-grained-authorization-using-amazon-cognito-api-gateway-and-iam/)


---

Authorizing functionality of an application based on group membership is a best practice.

If you’re building APIs with [Amazon API Gateway](https://aws.amazon.com/api-gateway/) and you need fine-grained access control for the users, use [Amazon Cognito](https://aws.amazon.com/cognito/).

- Amazon Cognito allows you to use groups to create a collection of users, which is often done to set the permissions for those users.
- IAM and Amazon Cognito can be used to provide fine-grained access control for the API behind API Gateway.
- use this approach to transparently apply fine-grained control to the API, without having to modify the code in the API, and create advanced policies by using IAM condition keys.


To build fine-grained authorization to protect the APIs using Amazon Cognito, API Gateway, and IAM.

a customer-facing application where the users are going to log into the web or mobile application
- as such you will be exposing the APIs through **API Gateway** with upstream services.
- The APIs could be deployed on [Amazon Elastic Container Service (Amazon ECS)](https://aws.amazon.com/ecs), [Amazon Elastic Kubernetes Service (Amazon EKS)](https://aws.amazon.com/eks/), [AWS Lambda](https://aws.amazon.com/lambda/), or [Elastic Load Balancing](https://aws.amazon.com/elasticloadbalancing/) where each of these options will forward the request to the [Amazon Elastic Compute Cloud (Amazon EC2)](https://aws.amazon.com/ec2/) instances.
- Additionally, you can use on-premises services that are connected to the AWS environment over an **AWS VPN** or [AWS Direct Connect](https://aws.amazon.com/directconnect/).
- It’s important to have fine-grained controls for each API endpoint and [HTTP method](https://docs.aws.amazon.com/apigateway/latest/developerguide/api-gateway-method-settings-method-request.html#setup-method-add-http-method).
- For instance
  - the user should be allowed to make a `GET` request to an endpoint
  - but should not be allowed to make a `POST` request to the same endpoint.
  - as a best practice, assign users to **groups** and use group membership to allow/deny access to the API services.

---

## Solution overview

Use an [Amazon Cognito user pool](https://docs.aws.amazon.com/cognito/latest/developerguide/cognito-user-identity-pools.html) as a user directory and let users authenticate and acquire the [JSON Web Token (JWT)](https://tools.ietf.org/html/rfc7519) to pass to the API Gateway.
- The JWT is used to identify what group the user belongs to, as mapping a group to an IAM policy will display the access rights the group is granted.

> **Note:**
> The solution works similarly if Amazon Cognito would be **federating users with an external identity provider (IdP)**
> - such as Ping, Active Directory, or Okta
> - instead of being an IdP itself.
> [Adding User Pool Sign-in Through a Third Party](https://docs.aws.amazon.com/cognito/latest/developerguide/cognito-user-pools-identity-federation.html).
> to use groups from an external IdP to grant access: [Role-based access control using Amazon Cognito and an external identity provider](https://aws.amazon.com/blogs/security/role-based-access-control-using-amazon-cognito-and-an-external-identity-provider/)



The following figure shows the basic architecture and information flow for user requests.

![Figure 1: User request flow](https://d2908q01vomqb2.cloudfront.net/22d200f8670dbdb3e253a90eee5098477c95c23d/2021/05/19/Building-fine-grained-authorization-1.png)

1. A user logs in and acquires an Amazon Cognito JWT ID token, access token, and refresh token.
   - [using tokens with user pools](https://docs.aws.amazon.com/cognito/latest/developerguide/amazon-cognito-user-pools-using-tokens-with-identity-providers.html).
2. A **RestAPI request** is made and a bearer token (an access token) is passed in the headers.
3. **API Gateway** forwards the request to a Lambda authorizer (custom authorizer).
4. The **Lambda authorizer** verifies the Amazon Cognito JWT using the **Amazon Cognito public key**.
   - On initial Lambda invocation, the public key is downloaded from Amazon Cognito and cached.
   - Subsequent invocations will use the public key from the cache.
5. The Lambda authorizer
   - looks up the Amazon Cognito group that the user belongs to in the JWT
   - and does a lookup in [Amazon DynamoDB](https://aws.amazon.com/dynamodb/) to get the policy that’s mapped to the group.
6. Lambda returns the policy and context (optionally) to API Gateway.
   - The context is a map containing key-value pairs that you can pass to the upstream service.
   - It can be additional information about the user, the service, or anything that provides additional information to the upstream service.
7. The **API Gateway policy engine** evaluates the policy.
   - Lambda isn’t responsible for understanding and evaluating the policy. That responsibility falls on the native capabilities of API Gateway.

8. The request is forwarded to the service.

> **Note:**
> To further optimize Lambda authorizer, the authorization policy can be cached or disabled, depending on the needs.
> By enabling cache, you could improve the performance as the authorization policy will be returned from the cache whenever there is a cache key match. [Configure a Lambda authorizer using the API Gateway console](https://docs.aws.amazon.com/apigateway/latest/developerguide/configure-api-gateway-lambda-authorization-with-console.html).

Example policy that is stored as part of an item in DynamoDB.

```json
    {
       "Version":"2012-10-17",
       "Statement":[
          {
             "Sid":"PetStore-API",
             "Effect":"Allow",
             "Action":"execute-api:Invoke",
             "Resource":[
                "arn:aws:execute-api:*:*:*/*/*/petstore/v1/*",
                "arn:aws:execute-api:*:*:*/*/GET/petstore/v2/status"
             ],
             "Condition":{
                "IpAddress":{
                   "aws:SourceIp":[
                      "192.0.2.0/24",
                      "198.51.100.0/24"
                   ]
                }
             }
          }
       ]
    }
```

Based on this example policy
- the user is allowed to make calls to the petstore API.
- For version v1, the user can make requests to any verb and any path, which is expressed by an asterisk `*`.
- For v2, the user is only allowed to make a `GET` request for path `/status`.
- [Output from an Amazon API Gateway Lambda authorizer](https://docs.aws.amazon.com/apigateway/latest/developerguide/api-gateway-lambda-authorizer-output.html).

---

## Getting started

**For this solution, need the following prerequisites**:

- The [AWS Command Line Interface (CLI)](https://aws.amazon.com/cli/) installed and [configured for use](https://docs.aws.amazon.com/cli/latest/userguide/cli-chap-configure.html).
- Python 3.6 or later, to package Python code for Lambda
  - recommend use a [virtual environment](https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/) or [virtualenvwrapper](https://virtualenvwrapper.readthedocs.io/en/latest/) to isolate the solution from the rest of the Python environment.
- An [IAM](https://aws.amazon.com/iam/) role or user with enough permissions to create `Amazon Cognito User Pool, IAM Role, Lambda, IAM Policy, API Gateway and DynamoDB table`.
- The GitHub repository for the solution.
  - [download it](https://github.com/aws-samples/amazon-cognito-api-gateway/archive/refs/heads/main.zip),
  - or [Git](https://git-scm.com/) command to download it from the terminal.


**To implement this reference architecture, utilizing the following services**:

[Amazon Cognito](https://aws.amazon.com/cognito) to support a **user pool** for the user base.
- A user pool is a user directory in Amazon Cognito.
- With a user pool, the users can log in to the web or mobile app through Amazon Cognito.
- use the Amazon Cognito user directory directly, as this sample solution creates an Amazon Cognito user.
- However, the users can also log in through `social IdPs, OpenID Connect (OIDC), and SAML IdPs`


[Lambda](https://aws.amazon.com/lambda) to serve the APIs.
- Lambda as backing API service
- Initially, create a Lambda function that serves the APIs.
- API Gateway forwards all requests to the Lambda function to serve up the requests.

[API Gateway](https://aws.amazon.com/api-gateway) to secure and publish the APIs.
- creates an Amazon Cognito user pool, a Lambda function, and an API Gateway instance.
- Next integrate the API Gateway instance with the Lambda function created.
- This API Gateway instance serves as an entry point for the upstream service.
- configures [proxy integration](https://docs.aws.amazon.com/apigateway/latest/developerguide/set-up-lambda-proxy-integrations.html) with Lambda and deploys an API Gateway stage.

---

## Deploy the sample solution

```bash
$ git clone https://github.com/aws-samples/amazon-cognito-api-gateway.git
$ cd amazon-cognito-api-gateway

# package the Python code for deployment to Lambda.
$ bash ./helper.sh package-lambda-functions
…
Successfully completed packaging files.

# generate a random Amazon Cognito user password and create the resources described in the previous section.
$ bash ./helper.sh cf-create-stack-gen-password
...
Successfully created CloudFormation stack.
```

### Validate Amazon Cognito user creation

To validate that an Amazon Cognito user has been created successfully, run the following command to open the Amazon Cognito UI in the browser and then log in with the credentials.

> **Note:**
> When you run this command, it returns the user name and password that you should use to log in.

```bash
$ bash ./helper.sh open-cognito-ui
Opening Cognito UI. Please use following credentials to login:
Username: cognitouser
Password: xxxxxxxx
```

Alternatively, you can open the CloudFormation stack and get the [Amazon Cognito hosted UI](https://docs.aws.amazon.com/cognito/latest/developerguide/cognito-user-pools-app-integration.html) URL from the stack outputs.
- The URL is the value assigned to the `CognitoHostedUiUrl` variable.

![Figure 2: CloudFormation Outputs - CognitoHostedUiUrl](https://d2908q01vomqb2.cloudfront.net/22d200f8670dbdb3e253a90eee5098477c95c23d/2021/05/19/Building-fine-grained-authorization-2.png)


### Validate Amazon Cognito JWT upon login

Since we haven’t installed a web application that would respond to the redirect request, Amazon Cognito will redirect to localhost, which might look like an error.
- The key aspect is that after a successful log in, there is a URL similar to the following in the navigation bar of the browser:
- https://localhost/#id_token=eyJraWQiOiJicVhMYWFlaTl4aUhzTnY3W


---

## Test the API configuration

To protect the API with Amazon Cognito so that only authorized users can access it
- verify that the configuration is correct and the API is served by API Gateway.

```bash
# makes a curl request to API Gateway to retrieve data from the API service.
$ bash ./helper.sh curl-api
{
    "pets":[
        {"id":1,"name":"Birds"},
        {"id":2,"name":"Cats"},
        {"id":3,"name":"Dogs"},
        {"id":4,"name":"Fish"}
    ]
}
# The expected result is that the response will be a list of pets.
# In this case, the setup is correct: API Gateway is serving the API.
```


---

## Protect the API

To protect the API, the following is required:

1. **DynamoDB** to store the policy that will be evaluated by the API Gateway to make an authorization decision.
2. A **Lambda function** to verify the user’s access token and look up the policy in DynamoDB.


### Lambda authorizer

- an API Gateway feature that uses a Lambda function to control access to an API.
- use a [Lambda authorizer](https://docs.aws.amazon.com/apigateway/latest/developerguide/apigateway-use-lambda-authorizer.html) to implement a custom authorization scheme that uses a bearer token authentication strategy.
- When a client makes a request to one of the API operations, the API Gateway calls the Lambda authorizer.
- The Lambda authorizer takes the identity of the caller as input and returns an IAM policy as the output.
- The output is the policy that is returned in DynamoDB and evaluated by the API Gateway.
- If there is no policy mapped to the caller identity, Lambda will generate a deny policy and request will be denied.



### DynamoDB table

- a key-value and document database that delivers single-digit millisecond performance at any scale.
- ideal for this use case to ensure that the Lambda authorizer can quickly process the bearer token, look up the policy, and return it to API Gateway.
- [Control access for invoking an API](https://docs.aws.amazon.com/apigateway/latest/developerguide/api-gateway-control-access-using-iam-policies-to-invoke-api.html).

create the DynamoDB table for the Lambda authorizer to look up the policy, which is mapped to an Amazon Cognito group.
- an item in DynamoDB. Key attributes are:
  - Group, which is used to look up the policy.
  - Policy, which is returned to API Gateway to evaluate the policy.

![Figure 3: DynamoDB item](https://d2908q01vomqb2.cloudfront.net/22d200f8670dbdb3e253a90eee5098477c95c23d/2021/05/19/Building-fine-grained-authorization-3.png)

Based on this policy
- the user that is part of the Amazon Cognito group `pet-veterinarian` is allowed to make API requests to endpoints
  - `https://_<domain>_/_<api-gateway-stage>_/petstore/v1/\*`
  - and `https://_<domain>_/_<api-gateway-stage>_/petstore/v2/status` for `GET` requests only.

---

### Update and create resources

```bash
# update existing resources and create a Lambda authorizer and DynamoDB table.
$ bash ./helper.sh cf-update-stack
Successfully updated CloudFormation stack.
```

---

## Test the custom authorizer setup

Begin the testing with the following request, which doesn’t include an access token.
- The request is denied with the message **Unauthorized**.
- the Amazon API Gateway expects a header named _Authorization_ (case sensitive) in the request.
- If there’s no authorization header, the request is denied before it reaches the lambda authorizer. This is a way to filter out requests that don’t include required information.

```bash
$ bash ./helper.sh curl-api
{"message":"Unauthorized"}
```

pass the required header
- but the token is invalid, it wasn’t issued by Amazon Cognito but is a simple JWT-format token stored in `./helper.sh`. [decode and verify an Amazon Cognito JSON token](https://aws.amazon.com/premiumsupport/knowledge-center/decode-verify-cognito-json-token/).
- This time the message is different. The Lambda authorizer received the request and identified the token as invalid and responded with the message **User is not authorized to access this resource**.

```bash
$ bash ./helper.sh curl-api-invalid-token
{"Message":"User is not authorized to access this resource"}
```

To make a successful request to the protected API, the code will need to perform the following steps:
1. Use a user name and password to authenticate against the Amazon Cognito user pool.
2. Acquire the tokens (id token, access token, and refresh token).
3. Make an `HTTPS (TLS) request` to **API Gateway** and pass the access token in the headers.
   - Before the request is forwarded to the API service, API Gateway receives the request and passes it to the Lambda authorizer.
   - The authorizer performs the following steps.
   - If any of the steps fail, the request is denied.
     1. Retrieve the public keys from Amazon Cognito.
     2. Cache the public keys so the Lambda authorizer doesn’t have to make additional calls to Amazon Cognito as long as the Lambda execution environment isn’t shut down.
     3. Use public keys to [verify the access token](https://docs.aws.amazon.com/cognito/latest/developerguide/amazon-cognito-user-pools-using-tokens-verifying-a-jwt.html).
     4. Look up the policy in DynamoDB.
     5. Return the policy to API Gateway.

The access token has claims such as Amazon Cognito assigned groups, user name, token use, and others, as shown in the following example (some fields removed).

```json
    {
        "sub": "00000000-0000-0000-0000-0000000000000000",
        "cognito:groups": [
            "pet-veterinarian"
        ],
    ...
        "token_use": "access",
        "scope": "openid email",
        "username": "cognitouser"
    }
```

Finally, programmatically log in to Amazon Cognito UI, acquire a valid access token, and make a request to API Gateway.
- call the protected API.
- receive a response with data from the API service.

```bash
$ bash ./helper.sh curl-protected-api
{"pets":[{"id":1,"name":"Birds"},{"id":2,"name":"Cats"},{"id":3,"name":"Dogs"},{"id":4,"name":"Fish"}]}
```


Steps that the example code performed:

1. Lambda authorizer validates the access token.
2. Lambda authorizer looks up the policy in DynamoDB based on the group name that was retrieved from the access token.
3. Lambda authorizer passes the IAM policy back to API Gateway.
4. API Gateway evaluates the IAM policy and the final effect is an allow.
5. API Gateway forwards the request to Lambda.
6. Lambda returns the response.

Let’s continue to test our policy from Figure 3. In the policy document, `arn:aws:execute-api:\*:\*:\*/\*/GET/petstore/v2/status` is the only endpoint for version V2, which means requests to endpoint `/GET/petstore/v2/pets` should be denied. Run the following command to test this.

```
$ bash ./helper.sh curl-protected-api-not-allowed-endpoint
{"Message":"User is not authorized to access this resource"}
```

### clean up

clean up all the resources associated with this solution:

```bash
$ bash ./helper.sh cf-delete-stack
```

### Advanced IAM policies to further control the API

With IAM, you can create advanced policies to further refine access to the APIs.
- [condition keys that can be used in API Gateway](https://docs.aws.amazon.com/apigateway/latest/developerguide/apigateway-resource-policies-aws-condition-keys.html), their use in an [IAM policy with conditions](https://docs.aws.amazon.com/IAM/latest/UserGuide/reference_policies_condition-keys.html), and how [policy evaluation logic](https://docs.aws.amazon.com/IAM/latest/UserGuide/reference_policies_evaluation-logic.html) determines whether to allow or deny a request.
