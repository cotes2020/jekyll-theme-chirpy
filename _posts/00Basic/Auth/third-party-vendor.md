

- [Access Flow](#access-flow)



---



## Access Flow

Org company's Cloud infrastructure is accessed using following information.

- `Programmatic Access` to Org company's infrastructure, through 3-party’s Vendor trustee account, using Access Key and Secret Key.
- At the time of credentialing, an auto-generated External-ID is used, which is unique to each Org company's account.
  - External ID is regenerated when the existing credential in Vendor product is deleted and re-credentialing of Org company's account is done in Vendor.


Questions:

- Manually rotate the Vendor IAM credentials? If so, the engineer operates the rotation can have a valid key and access Org company's data (even if they quit a day after)?
  - We do manually rotate, every 90 days, the Access-Key and Secret Key of 3-party’s Vendor trustee account.
  - The rotation is done with a policy and change management process with approvals.
  - Upon an engineer exiting the company, rotation will be done immediately as per the policy and process.

- An IAM user is used by Vendor to access roles in AWS@Company. How is access to the IAM user monitored? How are the API creds (access key, secret key) protected and monitored?
  - `Programmatic Access` to Org company's infrastructure, through 3-party’s Vendor trustee account, using Access Key and Secret Key. Access Key and Secret Key are rotated every 90 days.
  - Vendor leverages AWS best practice monitoring framework through CloudTrail, CloudWatch.

- Are we in a multitenant AWS account? Can Vendor assume into AWS environments for other customers from the same Vendor AWS account used to assume into AWS@Company?
  - As a multitenant solution, Vendor follows the AWS recommended approach of having a single AWS trustee account. Vendor also leverages AWS best practices for IAM roles and permission policies to access customer data.
  - We use `programmatic access` to Org company's infrastructure, through 3-party’s Vendor trustee account, using Access Key and Secret Key. Additional protection is given with an auto-generated External-ID to distinguish the tenants.
  - At the time of credentialing, an auto-generated External-ID is used, which is unique to each Org company's account. External ID is regenerated when the existing credential in Vendor product is deleted and re-credentialing of Org company's account is done in Vendor.
  - Vendor has **one AWS Trustee account and IAM user** to access all customers data via assuming an IAM role (created by customer using CF template).
  - Vendor uses per AWS account external IDs to prevent accidental cross customers access from that Trustee account, also known as “confuse deputy attack”
    - https://docs.aws.amazon.com/IAM/latest/UserGuide/id_roles_create_for-user_externalid.html ;
    - https://docs.aws.amazon.com/IAM/latest/UserGuide/confused-deputy.html
  - This trustee account is a single isolated AWS account that follows a least privilege permission model.
    - E.g. It can only access billing or usage data that is permitted by the customer.
  - Customer controls the cloud infrastructure and data in their AWS environment.
  - In the event of any suspicious activity, Org company's can remove the access given immediately.

- For the S3 bucket we pull Org company's data, it is the single bucket for all customers, do we have any object level security?
  - Customer data is encryption at rest, SSE-S3, managed by AWS.
  - Common S3 bucket across all customers, with object level encryption.
  - Vendor follows AWS best practices of secured access to S3 buckets controlled through access policies.
  - All accesses to S3 buckets is monitored by AWS Cloudtrail.
