---
title: AWS - Balancing - CloudFront
date: 2020-07-18 11:11:11 -0400
categories: [01AWS, Balancing]
tags: [AWS, Balancing, CloudFront]
toc: true
image:
---

- [CloudFront](#cloudfront)
  - [cloudFront](#cloudfront-1)
    - [edge locations](#edge-locations)
    - [Regional Edge caches](#regional-edge-caches)
  - [Origin Access Identity (OAI)](#origin-access-identity-oai)
  - [benefits](#benefits)
- [enable CloudFront](#enable-cloudfront)
- [CloudFront distributions](#cloudfront-distributions)
  - [cache control headers](#cache-control-headers)
  - [presigned URLs](#presigned-urls)
- [Configuring Secure Access and Restricting Access to Content](#configuring-secure-access-and-restricting-access-to-content)
  - [CloudFront Custom SSL Support](#cloudfront-custom-ssl-support)
  - [Using HTTPS with CloudFront](#using-https-with-cloudfront)
    - [Requiring HTTPS for Communication Between Viewers and CloudFront](#requiring-https-for-communication-between-viewers-and-cloudfront)
    - [Requiring HTTPS for Communication Between CloudFront and Custom Origin](#requiring-https-for-communication-between-cloudfront-and-custom-origin)
      - [Changing CloudFront Settings](#changing-cloudfront-settings)
      - [Installing an SSL/TLS Certificate on Your Custom Origin Server](#installing-an-ssltls-certificate-on-your-custom-origin-server)
      - [About RSA and ECDSA Ciphers](#about-rsa-and-ecdsa-ciphers)
    - [Requiring HTTPS for Communication Between CloudFront and S3 Origin](#requiring-https-for-communication-between-cloudfront-and-s3-origin)
  - [Using Alternate Domain Names and HTTPS](#using-alternate-domain-names-and-https)
  - [Restricting content with signed URLs and signed cookies](#restricting-content-with-signed-urls-and-signed-cookies)
  - [Restricting Access to S3 Content by signed URLs](#restricting-access-to-s3-content-by-signed-urls)
  - [Restricting Access to S3 Content by OAI Origin Access Identity](#restricting-access-to-s3-content-by-oai-origin-access-identity)
    - [Audit](#audit)
    - [OAI Setup](#oai-setup)
    - [create OAI and update CloudFront distribution](#create-oai-and-update-cloudfront-distribution)
      - [Creating an OAI by CloudFront console](#creating-an-oai-by-cloudfront-console)
      - [Creating an OAI by CloudFront API](#creating-an-oai-by-cloudfront-api)
    - [Granting the OAI Permission to Read Files in S3 Bucket](#granting-the-oai-permission-to-read-files-in-s3-bucket)
    - [Using S3 Bucket Policies](#using-s3-bucket-policies)
      - [Specify an OAI as the `Principal`](#specify-an-oai-as-the-principal)
    - [Updating S3 Object ACLs](#updating-s3-object-acls)
    - [Using an OAI in S3 Regions that Support Only Signature Version 4 Authentication](#using-an-oai-in-s3-regions-that-support-only-signature-version-4-authentication)

---

# CloudFront

---

![Screen Shot 2020-06-09 at 22.21.14](https://i.imgur.com/xIybRHQ.png)

![Screen Shot 2020-06-09 at 22.32.02](https://i.imgur.com/3CatMDS.png)


---

## cloudFront


- a web service
  - a fast CDN service
  - securely delivers data, videos, applications, and application programming interfaces (APIs) to customers globally with low latency and high transfer speeds.

- speeds the <font color=red> distribution of static and dynamic web content to users </font>
  - such as HTML; CSS; JavaScript; and image.

- <font color=red> provide additional security (especially against DDos) </font>
  - Build in DDoS attack protection.

- <font color=red> integrated with many AWS services </font>
  - such as S3, EC2, ELB, Route53, Lambda
  - both physical locations that are directly connected to the AWS global infrastructure
  - and software that works seamlessly with services including AWS shield for DDoS mitigation,
  - S3, Elastic Load Balancing, or EC2 as origins for your applications,
  - AWS Lambda to run custom code close to your viewer.

- provides a developer-friendly environment.

- <font color=red> different from traditional CDN </font>
  - quickly obtain the benefits of high-performance content delivery
  - without negotiated contracts, high prices, or minimum fees.

---


![Screen Shot 2020-05-06 at 01.39.15](https://i.imgur.com/cB7oVyh.png)



### edge locations
- locations are designed to serve popular content quickly to viewers.
- cache content at edge location (data centers) for fast distribution to customers.
- When user requests content that serve with CloudFront,
  - the user is routed to the edge location that provides the lowest latency (time delay)
  - content is delivered with the best possible performance.
  - If the content is already in the edge location with the lowest latency,
    - CloudFront delivers it immediately
  - If the content is not currently in that edge location,
    - CloudFront retrieves it from an S3 bucket / HTTP web server for the definitive version of your content.
- As objects become less popular
  - individual edge locations might remove those objects to make room for more popular content.

---

### Regional Edge caches
- help reduce the load on your origin resources
  - minimizes the operational burden
  - minimizes the costs of scaling the origin resources.

- Regional edge caches
  - are turned on by default for your CloudFront distributions, and do not need to make any changes to your distributions to take advantage of this feature.

- are CloudFront locations that deployed globally and close to viewers.
  - located between the origin server and the global edge locations
  - serve content directly to viewers.

- Regional edge cache has larger cache than individual edge location
  - so objects remain in the Regional edge cache longer.


---
## Origin Access Identity (OAI)
- A virtual identity
  - created OAI and associate it with your distribution.
  - configure permissions
  - CloudFront use the OAI to access and serve files to your users
  - but users can't use a direct URL to the S3 bucket to access a file there.

- restricting users to the CloudFront distribution,
  - enhance S3 distribution performance and create a better user experience
  - not used to configure private viewing within CloudFront
  - is not associated with an on-premise server; rather tied to S3 service.

- <font color=red> Prevent users from bypassing CloudFront security restrictions to access the S3 origin bucket </font>
  - OAIs are applied to a CloudFront distribution,
  - this allows access to end-users while protecting the direct URL to the S3 bucket.

- <font color=red> OAIs append code to a bucket policy </font>
  - allowing CloudFront users access to a particular bucket.
  - restrict users to use a presigned URL instead of directly accessing the origin S3 bucket.

- S3 hosted website
  - The endpoint URL can be altered.
  - Security errors in S3 may require the enabling of Cross-Origin Resource Sharing (CORS)
  - security measure
  - When needing object backups
  - When needing a resilient object
  - When needing performance replication
  - allowing a web application running in one domain to reference resources in another.
  - configuring S3 Cross-Region Replication (CRR)
  - A replication rule is needed, to set the source, destination, and IAM role.
  - Versioning needs to be enabled on both buckets (source and the destination)
  - Buckets need to be in different regions
  - Object ownership and its associated storage tiers can be altered when going to a new region.
  - CloudFront can be used to speed up the S3 service.
  - Server access logs can be enabled for buckets that host websites.
  - Apache-like server access logs can be enabled.
  - index.html file
  - default documented loaded when a visiting a bucket's URL endpoint.
- domain name
  - The domain name within CloudFront can be visited by HTTP and HTTPS.
  - A domain name is used to view distributed content in a browser.
  - The domain name is generated when a distribution is created.

---

## benefits

Key features for CloudFront include:
- TCP/IP optimizations for the network path.
- Keep-alive connections to reduce round-trip time.
- SSL/TLS termination that is close to viewers.
- Latency-based routing.
- And regional edge caches.


benefits
- <font color=red> Fast and global </font>
  - CloudFront is massively scaled and globally distributed.
  - To deliver content to end users with low latency, CloudFront uses a global network that consists of <font color=blue> edge locations </font> and <font color=blue> regional caches </font>

- <font color=red> Security at the edge </font>
  - provides both <font color=blue> network-level and application-level protection </font>
  - Your traffic and applications benefit through various built-in protections,
    - such as <font color=red> AWS Shield Standard </font> with no additional cost.
  - use configurable features
    - such as <font color=red> AWS Certificate Manager (ACM) </font> to create and manage custom Secure Sockets Layer (SSL) certificates with no extra cost.

- <font color=red> Highly programmable </font>
  - CloudFront features can be customized for specific application requirements.
  - integrates with <font color=red> Lambda@Edge </font> to run <font color=blue> custom code across AWS locations worldwide </font> to move complex application logic closer to users to improve responsiveness.
  - The CDN also supports integrations with other tools and automation interfaces for DevOps. It offers continuous integration and continuous delivery (CI/CD) environments.

- <font color=red> Deeply integrated with AWS </font>
  - integrated with AWS, with both physical locations that are directly connected to the AWS Global Infrastructure and other AWS services.
  - use APIs or the AWS Management Console to programmatically configure all features in the CDN.

- <font color=red> Cost-effective </font>
  - no minimum commitmente, charges only for what you use.
  - Compared to self-hosting
    - avoids the expense and complexity of operating a network of cache servers in multiple sites across the internet.
    - eliminates the need to overprovision capacity to serve potential spikes in traffic.
  - CloudFront also uses techniques like collapsing simultaneous viewer requests at an edge location for the same file into a single request to your origin server.
  - The result is
    - reduced load on the origin servers and reduced need to scale the origin infrastructure
    - result in further cost savings.
    - If the AWS origins is such as S3 or Elastic Load Balancing, you pay only for storage costs, not for any data transferred between these services and CloudFront.

CloudFront charges are based on actual usage of the service in four areas:
- <font color=red> Data transfer out </font>
  - charged for the volume of data that is transferred out from CloudFront edge locations, measured in GB, to the internet or to your origin (both AWS origins and other origin servers).
  - Data transfer usage is totaled separately for specific geographic regions, and then cost is calculated based on pricing tiers for each area.
  - If you use other AWS services as the origins of your files, you are charged separately for your use of those services, including storage and compute hours.
- <font color=red> HTTP(S) requests </font>
  - charged for the number of HTTP(S) requests that are made to CloudFront for the content.
- <font color=red> Invalidation requests </font>
  - charged per path in your invalidation request.
  - A path that is listed in your invalidation request represents the URL (or multiple URLs if the path contains a wildcard character) of the object that you want to invalidate from CloudFront cache.
  - can request up to 1,000 paths/month from CloudFront at no additional charge.
  - Beyond the first 1,000 paths, charged per path that is listed in your invalidation requests.
- <font color=red> Dedicated IP custom Secure Sockets Layer (SSL) </font>
  - pay $600 per month for each custom SSL certificate that is associated with one or more CloudFront distributions that use the Dedicated IP version of custom SSL certificate support.
  - This monthly fee is prorated by the hour.
  - For example,
  - custom SSL certificate was associated with at least one CloudFront distribution for just 24 hours (that is, 1 day) in a month, total charge for using the custom SSL certificate feature:
  - (1 day / 30 days) * $600 = $20

---

# enable CloudFront

![Screen Shot 2020-07-12 at 17.47.11](https://i.imgur.com/GBDeyRd.png)
￼
1. <font color=red> specify origin servers </font>
   - like an S3 bucket or your own HTTP server.
   - when CloudFront gets your files from the origin servers
   - the files will then be distributed from CloudFront edge locations worldwild.

2. <font color=red> upload files to origin servers </font>
   - Your files is objects
   - typically include webpages, images, and media files.

3. <font color=red> create a CloudFront distribution </font>
   - which tells CloudFront which origin servers to get the files from
     - when users request the files through your website or application.
   - At the same time, specify details, such as
     - whether you want CloudFront to log all requests,
     - whether you want the distribution to be enabled as soon as it's created.

4. <font color=red> CloudFront assigns a domain name to your new distribution </font>
   - can see the domain name in the CloudFront console.
   - The domain name can also be returned in the response to a programmatic request, like a request from an API.

5. CloudFront <font color=red> sends the distribution's configuration (not content) to all its edge locations </font>
   - collections of servers in geographically dispersed data centers where CloudFront caches copies of your objects.


CloudFront has several options for enablement.
- use a <font color=blue> separate Canonical Name Record / CNAME </font> for static content,
  - the static content is cached straight from the origin server.
  - most efficient
  - but takes more effort to set up and manage.
- point the <font color=blue> entire uniform resource locator / URL </font> to CloudFront
  - easier to manage.
  - can use URL patterns to stage dynamic content.
  - All of the content goes through edge locations.

---


# CloudFront distributions
￼
![Screen Shot 2020-07-12 at 17.55.36](https://i.imgur.com/HoysfPS.png)

- example
  - Route 53 resolves example.com on behalf of the client.
  - A request is made to the CloudFront distribution.
  - CloudFront looks at the S3 bucket for the content that it identifies as being stored statically.
  - For any content that has a time to live of zero,
    - CloudFront will go directly to the Elastic Load Balancing load balancer
    - to go back to the origin server to pull the content.
  - In this way, CloudFront delivers both static and dynamic content.

<font color=red> 2 distribution types </font>
- <font color=blue> Web distribution </font>
  - lets access to web content in any combination of up to 10 S3 buckets or custom origin servers.
- <font color=blue> Real Time Messaging Protocol / RTMP distribution </font>
  - always an S3 bucket.

To configure a cloudfront distribution to access a S3 private bucket:
- Add truested signers to you distribution
- Creat cloudfront key pairs for your trusted signers
- Write the code that generates <font color=red> signed uniform resource locators (URLs) </font>


```json
// creates a distribution for an S3 bucket named awsexamplebucket, and also specifies index.html as the default root object
aws cloudfront create-distribution \
    --origin-domain-name awsexamplebucket.s3.amazonaws.com \
    --default-root-object index.html

// Instead of using command line arguments, you can provide the distribution configuration in a JSON file
aws cloudfront create-distribution \
    --distribution-config file://dist-config.json

// The file dist-config.json:
{
    "CallerReference": "cli-example",
    "Aliases": { "Quantity": 0 },
    "DefaultRootObject": "index.html",
    "Origins": {
        "Quantity": 1,
        "Items": [
            {
                "Id": "awsexamplebucket.s3.amazonaws.com-cli-example",
                "DomainName": "awsexamplebucket.s3.amazonaws.com",
                "OriginPath": "",
                "CustomHeaders": { "Quantity": 0 },
                "S3OriginConfig": { "OriginAccessIdentity": "" }
            }
        ]
    },
    "OriginGroups": { "Quantity": 0 },
    "DefaultCacheBehavior": {
        "TargetOriginId": "awsexamplebucket.s3.amazonaws.com-cli-example",
        "ForwardedValues": {
            "QueryString": false,
            "Cookies": { "Forward": "none" },
            "Headers": { "Quantity": 0 },
            "QueryStringCacheKeys": { "Quantity": 0 }
        },
        "TrustedSigners": {
            "Enabled": false,
            "Quantity": 0
        },
        "ViewerProtocolPolicy": "allow-all",
        "MinTTL": 0,
        "AllowedMethods": {
            "Quantity": 2,
            "Items": [ "HEAD", "GET" ],
            "CachedMethods": {
                "Quantity": 2,
                "Items": [ "HEAD", "GET" ]
            }
        },
        "SmoothStreaming": false,
        "DefaultTTL": 86400,
        "MaxTTL": 31536000,
        "Compress": false,
        "LambdaFunctionAssociations": { "Quantity": 0 },
        "FieldLevelEncryptionId": ""
    },
    "CacheBehaviors": { "Quantity": 0 },
    "CustomErrorResponses": { "Quantity": 0 },
    "Comment": "",
    "Logging": {
        "Enabled": false,
        "IncludeCookies": false,
        "Bucket": "",
        "Prefix": ""
    },
    "PriceClass": "PriceClass_All",
    "Enabled": true,
    "ViewerCertificate": {
        "CloudFrontDefaultCertificate": true,
        "MinimumProtocolVersion": "TLSv1",
        "CertificateSource": "cloudfront"
    },
    "Restrictions": {
        "GeoRestriction": {
            "RestrictionType": "none",
            "Quantity": 0
        }
    },
    "WebACLId": "",
    "HttpVersion": "http2",
    "IsIPV6Enabled": true
}
```


---


## cache control headers

cache control headers
- CloudFront reads cache control headers
  - to determine how frequently to check the origin server for an updated version of that file.

- The cache control header set on your files
  - identifies static and dynamic content.
  - can even have custom headers within the CloudFront distribution graphical user interface / GUI within the console.

- Delivering all your content by using a single CloudFront distribution helps to ensure performance optimization on your entire website.

- expiration period
  - How long is a file kept at the edge location
    - set the expiration period by the cache control headers on files in origin server.
    - set one. If your files don’t change very often,
    - set to 0 seconds, CloudFront will revalidate every request with the origin server.
  - best practice：
    - set a long expiration period,
    - implement a versioning system to manage updates to your files.
- By default, if no cache control header is set
  - each edge location checks for an updated version of file
  - when it receives a request more than 24 hours after the previous time


3 ways to <font color=red> set content to expire / retire cached content </font>
- <font color=red> Use time to live, TTL (preferred) </font>
  - easiest but not immediate.
  - Fixed period of time
  - set the TTL for a particular origin server to 0,
    - CloudFront will still cache the content from that origin server.
  - CloudFront make a `GET` request to origin with an `If-Modified-Since` header.
    - This header allows the origin server to signal that CloudFront can continue to use the cached content if the content has not changed at the origin server.

- <font color=red> Change the object name (preferred) </font>
  - more effort but immediate.
  - No name force refresh
  - There might be some support for this option in some content management systems, or CMSs.
  - CloudFront distributes objects to edge locations
    - only when the objects are requested,
    - not when you put new or updated objects in your origin server.
  - Although you can update existing objects in a CloudFront distribution and use the same object names, it is not recommended.
    - If update an existing object in the origin server with a newer version that has the same name
    - an edge location won’t get that new version from your origin server until the object is updated and requested.

- <font color=red> Invalidating an object </font>
  - Inefficient and expensive
  - should be used sparingly for individual objects.
  - bad solution because the system must forcibly interact with all edge locations.


---

## presigned URLs
- owner can grant any user permissions to access an S3 object.
  - presigned URLs don't require users to have AWS security credentials or permissions.
  - A temporary URL that allows users to see assigned S3 objects using the creator's credentials.
  - Presigned URLs utilize the STS service.
- if get an error when access an S3 object.
  - presigned URL has expired
  - was created using an IAM role, and that role's temporary credentials have expired.
  - Permissions of the URL creator have changed


---


# Configuring Secure Access and Restricting Access to Content

- For web distributions, CloudFront provides several options for securing content that it delivers.
- The following are some ways you can use CloudFront to secure and restrict access to content:
  - <font color=red> Configure HTTPS connections </font>
  - Use <font color=red> geo restriction (geoblocking) </font>
    - Prevent users in specific geographic locations from accessing content
  - access content by <font color=red> CloudFront signed URLs or signed cookies </font>
    - do not want access by direct URL for the file.
    - Instead, access the files only by using the CloudFront URL, so that your protections work.
  - Set up <font color=red> field-level encryption </font> for specific content fields
  - Use <font color=red> AWS WAF to control access </font> to the content
  - create a web access control list (web ACL) to restrict access to your content.
    - such as the IP addresses that requests originate from or the values of query strings,
    - CloudFront responds to requests
      - with the requested content
      - or with an HTTP 403 status code (Forbidden).
  - Restrict access to content in S3 buckets
    - If use an S3 bucket as the origin for a CloudFront distribution
    - set up an <font color=red> origin access identity (OAI) </font> to manage direct access to your content.

---


## CloudFront Custom SSL Support
- By default, content is delivered to viewers over HTTPS by using a CloudFront distribution domain name. (https://xxx.cloudfront.net/image.jpg)

- CloudFronthas support <font color=red> custom Secure Sockets Layer, SSL </font>
  - can create your own certificate that will have the `CloudFront.net` domain,
  - or can bring your own certificate if you have a specific domain name that you want to use.

- <font color=red> Server Name Indication (SNI) Custom SSL </font>
  - relies on the SNI extension of the Transport Layer Security protocol
  - <font color=blue> allows multiple domains to serve SSL traffic over the same IP address </font>
  - CloudFront delivers content from each edge location, offers the same security as the Dedicated IP Custom SSL feature.
  - Some older browsers do not support SNI
    - some users might not be able to access your content by older browsers.
    - these browsers will not be able to establish a connection with CloudFrontto load the HTTPS version of your content.
  - no separate pricing for this feature.
    - use SNI Custom SSL with no upfront or monthly fees for certificate management.
    - Instead, pay normal CloudFrontrates for data transfer and HTTPS requests.

- <font color=red> Dedicated IP Custom SSL works for all clients </font>
  - to deliver content to browsers that don’t support SNI
  - CloudFront allocates dedicated IP addresses to serve your SSL content at each CloudFront edge location.
  - To learn more about Dedicated IP Custom SSL certificate support, select the link.http://aws.amazon.com/cloudfront/custom-ssl-domains/
  - When we approve your request, you can upload an SSL certificate and use the AWS Management Console to associate it with your CloudFront distributions.
  - If you need to associate more than one custom SSL certificate with your CloudFront distribution, include details about your use case and the number of custom SSL certificates that you intend to use in the “Use Case and # of SSL Certs You Intend to Use” section of the form.


---

## Using HTTPS with CloudFront

For web distributions, can configure CloudFront to

- require <font color=red> viewers use HTTPS to request the objects </font>
  - encrypt connections when CloudFront communicates with viewers.

- configure <font color=red> CloudFront to use HTTPS to get objects </font> from your origin,
  - encrypt connections when CloudFront communicates with your origin.

- configure CloudFront to <font color=red> require HTTPS to communicate with both viewers and origin </font>
  - when CloudFront receives a request for an object:
  1. A viewer submits an HTTPS request to CloudFront.
     - There's some SSL/TLS negotiation between the viewer and CloudFront.
     - In the end, the viewer submits the request in an encrypted format.
  2. CloudFront check the request
     - If the object is in the CloudFront edge cache
       - CloudFront encrypts the response and returns it to the viewer, and the viewer decrypts it.
     - If the object is not in the CloudFront cache
       - CloudFront performs SSL/TLS negotiation with your origin
       - when the negotiation is complete, forwards the request to your origin in an encrypted format.
       - Your origin decrypts the request, encrypts the requested object, and returns the object to CloudFront.
       - CloudFront decrypts the response, re-encrypts it, and forwards the object to the viewer.
       - CloudFront also saves the object in the edge cache so that the object is available the next time it's requested.
  4. The viewer decrypts the response.
  5. The process works similar whether the origin is an S3 bucket, MediaStore, or a custom origin such as an HTTP/S server.

> Note
> To help thwart SSL renegotiation-type attacks, CloudFront does not support renegotiation for viewer and origin requests.

---


### Requiring HTTPS for Communication Between Viewers and CloudFront

1. configure one or more cache behaviors in the CloudFront distribution
   - to require HTTPS for communication between viewers and CloudFront.
   - to allow both HTTP and HTTPS
     - so CloudFront requires HTTPS for some objects, but not for others

2. The configuration steps depend on <font color=red> the domain name using in object URLs </font>
   - using <font color=blue> the domain name that CloudFront assigned </font> to your distribution
     - like: `d111111abcdef8.cloudfront.net`
     - change the  <font color=red> Viewer Protocol Policy </font> setting for one or more cache behaviors to require HTTPS communication.
     - In that configuration, CloudFront provides the SSL/TLS certificate.
     - change the value of Viewer Protocol Policy 
       - by using the CloudFront console
       - use the CloudFront API to change the value of the `ViewerProtocolPolicy` element
         - UpdateDistribution in the CloudFront API.

   - using <font color=blue> your own domain name </font>
     - like: `example.com`
     - need to
       - change several CloudFront settings.
       - use an SSL/TLS certificate provided by AWS Certificate Manager (ACM), or import a certificate from a third-party certificate authority into ACM or the IAM certificate store.


to ensure the objects get from CloudFront were encrypted when CloudFront got them from the origin
- always use HTTPS between CloudFront and your origin.

If recently changed from HTTP to HTTPS between CloudFront and the origin
- recommend that invalidate objects in CloudFront edge locations.
- CloudFront will return an object to a viewer regardless of whether the protocol used by the viewer (HTTP or HTTPS) matches the protocol that CloudFront used to get the object.



To configure CloudFront to <font color=red> require HTTPS between viewers and CloudFront for one or more cache behaviors </font>
1. AWS Management Console > CloudFront console. `https://console.aws.amazon.com/cloudfront/`
2. CloudFront console > the ID for the distribution to update.
3. Behaviors tab > the cache behavior to update > choose Edit.
4. Specify one of the following values for <font color=red> Viewer Protocol Policy </font>
   - <font color=red> Redirect HTTP to HTTPS </font>
     - Viewers can use both protocols.
     - HTTP `GET` and `HEAD` requests are automatically redirected to HTTPS requests.
     - CloudFront returns HTTP status code 301 (Moved Permanently) along with the new HTTPS URL.
     - The viewer then resubmits the request to CloudFront using the HTTPS URL.
       - If you send `POST, PUT, DELETE, OPTIONS, or PATCH` over HTTP with an HTTP to HTTPS cache behavior and a request protocol version of HTTP
       - `protocol version of HTTP 1.1 or above`,
         - CloudFront redirects the request to a HTTPS location with a <font color=blue> HTTP status code 307 (Temporary Redirect) </font>
         - This guarantees that the request is sent again to the new location using the same method and body payload.
       - `protocol version below HTTP 1.1`
         - CloudFront returns a <font color=blue> HTTP status code 403 (Forbidden) </font>
     - When a viewer makes an HTTP request that is redirected to an HTTPS request, CloudFront charges for both requests.
       - For the HTTP request, the charge is only for the request and for the headers that CloudFront returns to the viewer.
       - For the HTTPS request, the charge is for the request, and for the headers and the object that are returned by your origin.
   - <font color=red> HTTPS Only </font>
     - Viewers can access your content only if they're using HTTPS.
     - If a viewer sends an HTTP request instead of an HTTPS request
       - CloudFront returns <font color=blue> HTTP status code 403 (Forbidden) </font> and does not return the object.

5. Choose Yes, Edit
6. Repeat steps 3 through 5 for each additional cache behavior that you want to require HTTPS for between viewers and CloudFront.
7. Confirm the following before you use the updated configuration in a production environment:
   - The path pattern in each cache behavior applies only to the requests that you want viewers to use HTTPS for.
   - The cache behaviors are listed in the order that you want CloudFront to evaluate them in.
   - The cache behaviors are routing requests to the correct origins.

---

### Requiring HTTPS for Communication Between CloudFront and Custom Origin

- require HTTPS for communication between CloudFront and your custom origin, the steps depend on
  - whether you're using the domain name that CloudFront assigned to your distribution (`d111111abcdef8.cloudfront.net`)
  - or using your own alternate domain name (`example.com`)

> Note
> If your custom origin is an S3 bucket that’s configured as a website endpoint, you can’t configure CloudFront to use HTTPS with your origin because S3 doesn’t support HTTPS for website endpoints.

1. <font color=red> Use the default CloudFront domain name </font>
   - use the domain name that CloudFront assigned to the distribution in the URLs for the objects
   - (https://d111111abcdef8.cloudfront.net/logo.jpg)
   - require HTTPS by following the procedures in this topic to do the following:
     - Change the <font color=red> Origin Protocol Policy </font> setting for specific origins in your distribution
     - Install an SSL/TLS certificate on your custom origin server (this isn't required when you use an S3 origin)

2. Use an alternate domain name
   - add an alternate domain name that's easier to work with, like `example.com`.
   - [steps and guidance in Using Alternate Domain Names and HTTPS](https://docs.aws.amazon.com/AmazonCloudFront/latest/DeveloperGuide/using-https-alternate-domain-names.html)


#### Changing CloudFront Settings
- configure CloudFront to use HTTPS to communicate with an Elastic Load Balancing load balancer, an EC2 instance, or another custom origin.
- to use the CloudFront API to update a web distribution, see `UpdateDistribution` in the CloudFront API Reference.

<font color=red> To configure CloudFront to require HTTPS between CloudFront and your custom origin </font>

1. AWS Management Console > CloudFront console at `https://console.aws.amazon.com/cloudfront/`
2. CloudFront console > choose the ID for the distribution that you want to update.
3. Origins tab > choose the origin to update > choose Edit.
4. Update the following settings:
   - <font color=red> Origin Protocol Policy </font>
     - Change the Origin Protocol Policy for the applicable origins in your distribution:
       - **HTTPS Only**
         - CloudFront uses only HTTPS to communicate with your custom origin.
       - **Match Viewer**
         - CloudFront communicates with your custom origin using HTTP or HTTPS
           - depending on the protocol of the viewer request.
         - Example
           - choose Match Viewer for Origin Protocol Policy
           - the viewer `uses HTTPS to request an object` from CloudFront,
           - CloudFront also `uses HTTPS to forward the request` to your origin.
         - Choose Match Viewer only if you specify Redirect HTTP to HTTPS or HTTPS Only for Viewer Protocol Policy.
         - CloudFront caches the object only once even if viewers make requests using both HTTP and HTTPS protocols.
   - <font color=red> Origin SSL Protocols </font>
     - Choose the Origin SSL Protocols for the applicable origins in your distribution.
     - The SSLv3 protocol is less secure, only if origin doesn’t support TLSv1 or later.
       - The TLSv1 handshake is both backwards and forwards compatible with SSLv3, but TLSv1.1 and TLSv1.2 are not.
       - When you choose SSLv3, CloudFront only sends SSLv3 handshake requests.
5. Choose Yes, Edit.
6. Repeat steps 3 through 5 for each additional origin that you want to require HTTPS for between CloudFront and your custom origin.
7. Confirm the following before you use the updated configuration in a production environment:
   - The path pattern in each cache behavior applies only to the requests that you want viewers to use HTTPS for.
   - The cache behaviors are listed in the order that you want CloudFront to evaluate them in. For more information, see Path Pattern.
   - The cache behaviors are routing requests to the origins that you changed the Origin Protocol Policy for.


#### Installing an SSL/TLS Certificate on Your Custom Origin Server

use an SSL/TLS certificate from the following sources on your custom origin:

- origin is an Elastic Load Balancing load balancer
  - use a certificate provided by AWS Certificate Manager (ACM).
  - use a certificate that is signed by a trusted third-party certificate authority and imported into ACM.

- origins other than ELB load balancers,
  - must use a certificate that is signed by a trusted third-party certificate authority (CA)
  - for example, Comodo, DigiCert, or Symantec.

When CloudFront uses HTTPS to communicate with your origin
- CloudFront verifies that the certificate was issued by a trusted certificate authority.
- CloudFront supports the same certificate authorities that Mozilla does.
- For the current list, see Mozilla Included CA Certificate List.
- You can't use a self-signed certificate for HTTPS communication between CloudFront and the origin.

Important
- If the origin server returns an expired / invalid / self-signed certificate, or returns the certificate chain in the wrong order,
  - CloudFront drops the TCP connection, r
  - eturns HTTP status code 502 (Bad Gateway),
  - and sets the X-Cache header to Error from cloudfront.
- Also, if the full chain of certificates, including the intermediate certificate, is not present,
  - CloudFront drops the TCP connection.

The certificate returned from the origin
- must cover the domain specified for Origin Domain Name for the corresponding origin in your distribution.
- In addition, if you configured CloudFront to forward the Host header to your origin, the origin must respond with a certificate matching the domain in the Host header.




#### About RSA and ECDSA Ciphers
The encryption strength of a communications connection depends on the key size and strength of the algorithm of origin server’s certificate.
- The two options that CloudFront supports for connections with a custom origin
  - RSA
  - Elliptic Curve Digital Signature Algorithm (ECDSA).
- For lists of the RSA and ECDSA ciphers supported by CloudFront, see Supported SSL/TLS protocols and ciphers for communication between CloudFront and your origin.


<font color=red> RSA </font>

- CloudFront and origin servers typically use RSA 2048-bit asymmetric keys for SSL/TLS termination.

- The strength of RSA
  - relies on the presumed difficulty of breaking a key that requires factoring the product of two large prime numbers.
    - RSA algorithms use the product of two large prime numbers, with another number added to it to create a public key.
    - The private key is a related number.
  - but faster computer calculations have weakened RSA algorithms
  - it’s now easier to break the encryption.
- to maintain encryption strength while continuing to use RSA
  - increase the size of the RSA keys.
  - However, this approach isn’t easily scalable because using larger keys increases the compute cost for cryptography.


<font color=red> ECDSA </font>

- use an ECDSA certificate.
- ECDSA bases its security on more complex mathematical problem than RSA
  - takes more computer processing time to break ECDSA encryption.
- ECDSA is built on the principle that it is difficult to solve for the discrete logarithm of a random elliptic curve when its base is known, also known as the Elliptic Curve Discrete Logarithm Problem (ECDLP).

- so use shorter key lengths to achieve the equivalent security of using RSA with much larger key sizes.
  - providing better security
  - ECDSA's smaller keys
  - enables faster computing of algorithms, smaller digital certificates, and fewer bits to transmit during the SSL/TLS handshake.
  - reduce the time it takes to create and sign digital certificates for SSL/TLS termination on origin servers.
  - can increase throughput by reducing the compute cycles needed for cryptography, freeing up server resources to process other work.


Choosing Between RSA and ECDSA Ciphers
- Sample tests
  - Example,
  - 2048-bit RSA to 256-bit ECDSA (nistp256)
  - nistp256 option was 95% faster than 2048-bit RSA
  - while providing the same security strength as 3072-bit RSA.
- CloudFront continues to support RSA for SSL/TLS connections.
  - However, for encryption for SSL/TLS authentication for origin servers, ECDSA could be a better option.
- stronger encryption, the reduction in computational cost of cryptography while using ECDSA at your origin servers is an added advantage.



<font color=red> Use ECDSA Ciphers for communications between CloudFront and the origin </font>

1. Generate a private key by using either of the supported curves (prime256v1 or secp384r1).
2. Generate an <font color=blue> ECDSA Digital Certificate </font> in the `X.509` PEM format with a trusted certificate authority.
3. Set up your origin to prefer the ECDSA certificate.
4. Using ECDSA doesn't require any settings changes in the CloudFront console or APIs, and there is no additional fee.


---


### Requiring HTTPS for Communication Between CloudFront and S3 Origin

When origin is an S3 bucket, options for using HTTPS for communications with CloudFront depend on how you're using the bucket.
- the S3 bucket is configured as a website endpoint
  - no HTTPS when communicate between origin and CloudFront
  - S3 doesn't support HTTPS connections in that configuration.

- When your origin is an S3 bucket that supports HTTPS communication
  - CloudFront always forwards requests to S3 by using the protocol that viewers used to submit the requests.
  - The default setting for the <font color=red> Origin Protocol Policy setting is Match Viewer </font> and can't be changed.

- to require HTTPS for communication between CloudFront and S3
  - must <font color=red> change Viewer Protocol Policy to Redirect HTTP to HTTPS or HTTPS Only </font>
  - The procedure later in this section explains how to use the CloudFront console to change Viewer Protocol Policy.
  - using the CloudFront API to update the `ViewerProtocolPolicy` element for a web distribution: `UpdateDistribution` in the CloudFront API

When you use HTTPS with an S3 bucket that supports HTTPS communication, S3 provides the SSL/TLS certificate, so you don't have to.

To configure CloudFront to require HTTPS to your S3 origin
1. AWS Management Console > CloudFront console
2. CloudFront console > choose the ID for the distribution to update.
3. Behaviors tab > choose the <font color=red> cache behavior </font> to update > choose Edit.
4. Specify one of the following values for <font color=red> Viewer Protocol Policy </font>
   - **Redirect HTTP to HTTPS**
     - Viewers can use both protocols
     - but HTTP requests are automatically redirected to HTTPS requests.
     - CloudFront returns HTTP status code 301 (Moved Permanently) along with the new HTTPS URL.
     - The viewer then resubmits the request to CloudFront using the HTTPS URL.
     - Important
       - CloudFront doesn't redirect `DELETE, OPTIONS, PATCH, POST, or PUT` requests from HTTP to HTTPS.
       - CloudFront responds to `HTTP DELETE, OPTIONS, PATCH, POST, or PUT` requests for that cache behavior with HTTP status code 403 (Forbidden).
     - When a viewer makes an HTTP request that is redirected to an HTTPS request, CloudFront charges for both requests.
       - For the HTTP request, the charge is only for the request and for the headers that CloudFront returns to the viewer.
       - For the HTTPS request, the charge is for the request, and for the headers and the object returned by your origin.
   - **HTTPS Only**
     - Viewers can access your content only if they're using HTTPS.
     - If a viewer sends an HTTP request instead of an HTTPS request,
       - CloudFront returns HTTP status code 403 (Forbidden) and does not return the object.
5. Choose Yes, Edit.
6. Repeat steps 3 through 5 for each additional cache behavior that you want to require HTTPS for between viewers and CloudFront, and between CloudFront and S3.
7. Confirm the following before you use the updated configuration in a production environment:
   - The path pattern in each cache behavior applies only to the requests that you want viewers to use HTTPS for.
   - The cache behaviors are listed in the order that you want CloudFront to evaluate them in. For more information, see Path Pattern.
   - The cache behaviors are routing requests to the correct origins.


---




## Using Alternate Domain Names and HTTPS
1. To use dedicated IP, request permissionfor your AWS account (not necessary for SNI)
  - By default, when request permission to use an alternate domain name with HTTPS, we update your account so that you can associate two custom SSL certificates with your CloudFront distributions.
  - Typically, you only use the second certificate temporarily, like when have more than one distribution and need to rotate certificates.
  - If you need to permanently associate two or more certificates with your distributions, indicate how many certificates you need, and describe your circumstances in your request.
1. upload your SSL certificate to the IAM certificate store by AWS Certificate Manager (ACM) or AWS CLI.
  - ACM (preferred tool): provisioning, managing, and deploying server certificates.
  - Certificates that are provided by ACM are free, and automatically renew.
  - With ACM, you can
  - To request a certificate.
  - to deploy an existing ACM or external certificate to AWS resources.
  - to manage server certificates from the console or programmatically.
  - use IAM as a certificate manager only when must support HTTPS connections in a Region that not supported by ACM.
  - IAM securely encrypts your private keys, and itstores the encrypted version in IAM SSL certificate storage.
  - IAM supports deploying server certificates in all Regions, but you must obtain your certificate from an external provider for use with AWS. You cannot upload an ACM certificate to IAM.
  - When you use the IAM CLI to upload your SSL certificate to the IAM certificate store, you must use the same AWS account that you used to create your CloudFront distribution.
  - When you upload your certificate to IAM, the value of the -path parameter, or certificate path, must start with /cloudfront/.
  - Examples of the -path parameter include
  - /cloudfront/production/
  - /cloudfront/test/
  - The path also must end with a /.
  - to use the CloudFront API to create or update your distribution,
  - make sure note the alphanumeric string that AWS CLI returns, such as AS1A2M3P4L5E67SIIXR3J.
  - This string is the value that you will specify in the IAMCertificateId element.
  - You do not need the IAM Amazon Resource Name—or ARN—which is also returned by the CLI.
1. You can update your distribution to include your domain names
  - so that you can specify which SSL certificate you want to use, specify a dedicated IP address or SNI, and add or update DNS records.
  - After you associate your SSL certificate with your CloudFront distribution, do not delete the certificate from the IAM certificate store until you remove the certificate from all distributions, and until the status of the distributions has changed to Deployed.
  - To request permission, select the link. https://aws.amazon.com/cloudfront/custom-ssl-domains/


Advanced SSL features support
- High security ciphers
  - to improve the security of HTTPS connections.
  - CloudFront edge servers and clients, such as browsers, automatically agree on a cipher as part of the SSL handshake process.
  - The connections can now use ciphers with advanced features, such as Elliptic Curve signatures and key exchanges.
- Perfect forward secrecy
  - uses a unique random session key to provide additional safe guards against the eavesdropping of encrypted data.
  - This feature prevents the decoding of captured data, even if the secret long-term key is compromised.
- OCSP stapling
  - improves the time taken for individual SSL handshakes.
  - It moves the Online Certificate Status Protocol (OSCP) check, which is used to obtain the revocation status of an SSL certificate, from the client to a periodic, secure check by the CloudFront servers.
  - With OCSP stapling, the client no longer needs to handle certificate validation, which improves performance.
- Session tickets
  - speed up the time to restart or resume an SSL session.
  - CloudFront encrypts SSL session information and stores it in a ticket.
  - The client can use this ticket to resume a secure connection instead of repeating the SSL handshake process



---

## Restricting content with signed URLs and signed cookies


---


## Restricting Access to S3 Content by signed URLs


make content private
- restrict access to objects in your S3 bucket.
- require that users use signed URLs.
  - create CloudFront key pairs for trusted signers
  - write the code that generates signed URLs.
  - write an application that automatically generates signed URLs
  - Or use a web interface to create signed URLs
  - add trusted signers to your distribution.
  - After you add a trusted signer to your distribution, users must use signed URLs to access the corresponding content.
  - A signed URL:
  - includes additional information
  - such as an expiration date and time,
  - gives you more control over access to your content.
  - This additional information appears in a policy statement that is based on canned / custom policy:
  - canned policy: restrict access to a single object. https://docs.aws.amazon.com/AmazonCloudFront/latest/DeveloperGuide/private-content-creating-signed-url-canned-policy.html
  - custom policy: restrict access to one or more objects by using pattern matching. https://docs.aws.amazon.com/AmazonCloudFront/latest/DeveloperGuide/private-content-creating-signed-url-custom-policy.html



---

## Restricting Access to S3 Content by OAI Origin Access Identity

![Screen Shot 2021-01-22 at 12.30.06](https://i.imgur.com/oGIQIXW.png)

> When your Amazon Cloudfront CDN distributions are using AWS S3 as an origin, the distributions content should be kept private and delivered only via Cloudfront network, using an origin access identity to regulate access.



In general, using an S3 bucket as the origin for a CloudFront distribution
- either <font color=red> allow everyone </font> to have access to the files there,
- or restrict the access.
  - restrict access by <font color=red> CloudFront signed URLs or signed cookies </font>
  - to restrict access to view files by the direct S3 URL for the file.
  - let them only access the files by the CloudFront URL, so the protections work.


<font color=red> Origin access identity </font>

- to restrict access
  - restrict access to S3 content by creating an OAI, a special CloudFront user.
  - CloudFront OAI gets objects from S3 on behalf of the users.
  - Direct access to the objects through S3 URLs will be denied.
- Cloudfront distributions can be much more cost effective
  - the price for CloudFront data transfer is lower than the price for S3 data transfer.
- downloads are faster
  - only the CloudFront service is used to deliver the application objects instead of S3
  - because the objects are copied to all edge locations within the distribution in order to be stored closer to your users.



> Important
> If you use an S3 bucket configured as a website endpoint
> - must set it up with CloudFront as a custom origin.
> - can’t use the origin access identity feature
> - However, you can restrict access to content on a custom origin by setting up custom headers and configuring your origin to require them.
> - For more information, see Restricting access to files on custom origins.


- create an origin access identity by CloudFront console or the CloudFront API.
  - CloudFront console: http://docs.aws.amazon.com/AmazonCloudFront/latest/DeveloperGuide/private-content-restricting-access-to-s3.html#private-content-creating-oai-console
  - CloudFrontAPI: http://docs.aws.amazon.com/AmazonCloudFront/latest/DeveloperGuide/private-content-restricting-access-to-s3.html#private-content-creating-oai-api


- To use this feature to restrict access to content from S3 buckets
  - Create a special CloudFront user called an origin access identity (OAI) and associate it with your distribution.
  - Configure your S3 bucket permissions
    - change the permissions either on your S3 bucket or the objects in bucket
      - so only the origin access identity has read permission.
    - so CloudFront can use the OAI to access the files in your bucket and serve them to your users.
    - Make sure that users can’t use a direct URL to the S3 bucket to access a file there.
  - After these steps, users can only access the files through CloudFront, not directly from the S3 bucket.

---

### Audit
To determine if origin access identity is enabled for your Cloudfront distributions configured with S3 as origin, perform the following:


AWS CLI:

```bash
aws cloudfront list-distributions \
	--output table \
	--query 'DistributionList.Items[*].Id'
# ---------------------
# | ListDistributions |
# +-------------------+
# |   E7GGTQ8UCFC4G   |
# |   G31A16G5KZMUX   |
# |   D8E6G5KZMPDT0   |
# +-------------------+


# return multiple values using query argument
aws cloudfront list-distributions \
  --output text
  --query "DistributionList.Items[*].Origins.Items[*].{id:Id,name:DomainName}"

aws cloudfront list-distributions \
  --output text
  --query "DistributionList.Items[*].{id:Id, origin:Origins.Items[0].Id}[?origin=='S3-BUCKET_NAME'].id"


# expose the name of the origin access identity set for each S3 origin entry associated with the selected AWS Cloudfront distribution:
aws cloudfront get-distribution-config \
	--id DistributionList.Id \

aws cloudfront get-distribution-config \
	--id E7GGTQ8UCFC4G \
	--query 'DistributionConfig.Origins.Items[*].S3OriginConfig.OriginAccessIdentity'

```



---

### OAI Setup

When set up an S3 bucket as the origin for a CloudFront distribution
- grant everyone permission to read the files in your bucket.
  - allows anyone to access your files either <font color=blue> through CloudFront or using the S3 URL </font>
  - CloudFront doesn't expose S3 URLs,
  - but the users might have those URLs if your application serves any files directly from S3
  - or if anyone gives out direct links to specific files in S3.


- <font color=red> use CloudFront signed URLs or signed cookies </font> to restrict access
  - but it not prevent users from <font color=blue> accessing S3 files by using S3 URLs </font>
  - If users access your files directly in S3,
    - they <font color=blue> bypass the controls provided by CloudFront signed URLs or signed cookies </font>
      - This includes
      - control over the date and time that a user can no longer access your content,
      - and control over which IP addresses can be used to access content.
    - <font color=blue> CloudFront access logs are less useful </font>
      - because they're incomplete, users access files both through CloudFront and directly by using S3 URLs,

To ensure access by only CloudFront URLs, regardless of whether the URLs are signed, do the following:

1. Create an origin access identity
   - a special CloudFront user
   - associate the origin access identity with the distribution.
   - associate the origin access identity with origins, so that you can secure all or just some of your S3 content.
   - can also create an origin access identity and add it to your distribution when you create the distribution.

2. Change the permissions either on your S3 bucket or on the files in bucket
   - only the origin access identity has read permission (or read and download permission).
   - When your users access your S3 files through CloudFront,
     - the CloudFront origin access identity gets the files on behalf of your users.
   - If your users request files directly by using S3 URLs, they're denied access.
   - The origin access identity has permission to access files in your S3 bucket, but users don't.


An AWS account can have up to 100 CloudFront origin access identities (OAIs)
- can add an OAI to as many distributions as you want, so one OAI is usually sufficient.


1. create an OAI and add it to the distribution when created the distribution
2. create and add one now by either the CloudFront console or the CloudFront API:
   - CloudFront console
     - can create an OAI and add it to the distribution at the same time.
   - CloudFront API version 2009-09-09 or later.
     - create an OAI, and then you add it to your distribution.


---


### create OAI and update CloudFront distribution


---


#### Creating an OAI by CloudFront console
1. AWS Management Console > CloudFront console > Choose the ID of a distribution that has an S3 origin.
2. Choose the **Origins and Origin Groups** tab.
3. Choose the check box next to an origin, and then choose **Edit**.

4. For **Restrict Bucket Access**, choose **Yes**.
   - If you already have an OAI that you want to use,
     - choose **Use an Existing Identity**.
     - Then choose the OAI in the **Your Identities** list.
   - to create an OAI, choose **Create a New Identity**.
     - You can replace the bucket name in the **Comment** field with a custom description.

5. If you want CloudFront to automatically give the OAI permission to read the files in the S3 bucket specified in **Origin Domain Name**, choose **Yes, Update Bucket Policy**.
   - CloudFront updates bucket permissions to grant the specified OAI permission to read files in your bucket.
   - However, CloudFront does not remove existing permissions.
   - If users currently have permission to access the files in your bucket using S3 URLs, they will still have that permission after CloudFront updates your bucket permissions.
   - To view or remove existing bucket permissions, use a method provided by S3.
   - If you want to manually update permissions on your S3 bucket, choose **No, I Will Update Permissions**.
6. Choose **Yes, Edit**.
7.  If you have more than one origin, repeat the steps to add an OAI for each one.

---


#### Creating an OAI by CloudFront API

1. create a CloudFront OAI.
   - `POST Origin Access Identity` API action.

    ```bash
    # creates a CloudFront origin access identity (OAI) by providing the OAI configuration as a command line argument:
    aws cloudfront create-cloud-front-origin-access-identity \
        --cloud-front-origin-access-identity-config \
        CallerReference="cli-example",Comment="Example OAI"

    # providing the OAI configuration in a JSON file:
    aws cloudfront create-cloud-front-origin-access-identity \
        --cloud-front-origin-access-identity-config file://OAI-config.json

    # The file OAI-config.json:
    {
        "CallerReference": "cli-example",
        "Comment": "Example OAI"
    }

    # the output is the same:
    {
        "Location": "https://cloudfront.amazonaws.com/2019-03-26/origin-access-identity/cloudfront/E74FTE3AEXAMPLE",
        "ETag": "E2QWRUHEXAMPLE",
        "CloudFrontOriginAccessIdentity": {
            "Id": "E74FTE3AEXAMPLE",
            # the value used to associate the OAI with your distribution.
            "S3CanonicalUserId": "cd13868f797c227fbea2830611a26fe0a21ba1b826ab4bed9b7771c9aEXAMPLE",
            # the value used when you use S3 object ACLs to give the OAI access to your S3 objects.
            "CloudFrontOriginAccessIdentityConfig": {
                "CallerReference": "cli-example",
                "Comment": "Example OAI"
            }
        }
    }
    ```


2. Adding OAI to Distribution by CloudFront API
   - use the CloudFront API to add a CloudFront OAI to an existing distribution or to create a new distribution that includes an OAI.
   - In either case, include an `OriginAccessIdentity` element.
     - This element contains the value of the `Id` element
   - You can add the `OriginAccessIdentity` element to one or more origins.
   - Create a new web distribution
   - Update an existing web distribution


```bash
# Run get-distribution-config command to extract the configuration metadata from the Cloudfront distribution that you want to reconfigure
# returns the configuration details of an AWS Cloudfront CDN distribution identified by the ID E7GGTQ8UCFC4G:
aws cloudfront get-distribution-config \
	--id E7GGTQ8UCFC4G
  --profile yourrole

# output
{
    "ETag": "E1VEIGDP0YISPR",
    "DistributionConfig": {
        "Comment": "",
        "CacheBehaviors": { "Quantity": 0 },
        "IsIPV6Enabled": true,
        "Origins": {
            "Items": [
                {
                    # modify this
                    # "S3OriginConfig": { "OriginAccessIdentity": "" },
                    "S3OriginConfig": {
                        "OriginAccessIdentity": "access-identity-cloudconformity-web-assets.s3.amazonaws.com"
                    },
                    "OriginPath": "/static",
                    "CustomHeaders": { "Quantity": 0 },
                    "Id": "S3-cloudconformity-web-assets",
                    "DomainName": "cloudconformity-web-assets..."
                }
            ],
            "Quantity": 1
        },

       ...

        "CallerReference": "1495036941163",
        "ViewerCertificate": {
            "CloudFrontDefaultCertificate": true,
            "MinimumProtocolVersion": "SSLv3",
            "CertificateSource": "cloudfront"
        },
        "CustomErrorResponses": { "Quantity": 0 },
        "HttpVersion": "http2",
        "Restrictions": {
            "GeoRestriction": {
                "RestrictionType": "none",
                "Quantity": 0
            }
        },
        "Aliases": { "Quantity": 0 }
    }
}

# to enable origin access identity for other Cloudfront CDN distributions
# Run update-distribution to update your AWS Cloudfront distribution in order to enable origin access identity and restrict user access to the S3 bucket used as distribution origin.
# updates an AWS CloudFront CDN web distribution with the ID E7GGTQ8UCFC4G and the ETag E1VEIGDP0YISPR, using the JSON configuration document named cloudfront-distconfig-enable-oai.json, created at the previous step:
aws cloudfront update-distribution \
	--id E7GGTQ8UCFC4G \
	--distribution-config file://cloudfront-distconfig-enable-oai.json \
	--if-match E1VEIGDP0YISPR

```

---



### Granting the OAI Permission to Read Files in S3 Bucket

When create or update a distribution
- add an origin access identity (OAI) and <font color=red> automatically update the S3 bucket policy </font> to give the OAI permission to access your bucket.
- or <font color=red> manually create or update the bucket policy </font>
- or use object ACLs that control access to individual files in the bucket.

Whichever method, review the permissions to make sure that:
* <font color=red> CloudFront OAI can access files </font> in the bucket on behalf of viewers who are requesting them through CloudFront.
* <font color=red> Viewers can’t use S3 URLs </font> to access your files outside of CloudFront.


Important
- If you configure CloudFront to accept and forward all of the HTTP methods that CloudFront supports, make sure you give your CloudFront OAI the desired permissions.
- For example,
- configure CloudFront to accept and forward requests that use the `DELETE` method,
- configure your bucket policy or object ACLs to handle `DELETE` requests appropriately so viewers can delete only files that you want them to.

Note the following:
- it is easier to use S3 bucket policies than object ACLs
  - S3 bucket policies
    - can add files to the bucket without updating permissions.
  - object ACLs
    - give more fine-grained control
    - it granting permissions on each individual file.

- By default, S3 bucket and all files in it are private.
  - Only the AWS account that created the bucket has permission to read or write the files in it.

- If another AWS account uploads files to your bucket, that account is the owner of those files.
  - Bucket policies only apply to files that the bucket owner owns.
  - if another account uploads files to your bucket,
    - the bucket policy that you created for your OAI is not evaluated for those files.
    - In that case, <font color=red> use object ACLs to give permissions to your OAI </font>

- If adding an OAI to an existing distribution
  - modify the bucket policy or any object ACLs as appropriate
  - to ensure that the files are not publicly available outside of CloudFront.

- Grant additional permissions to one or more secure administrator accounts
  - so you can continue to update the contents of the S3 bucket.

- There might be a brief delay between save changes to S3 permissions and when the changes take effect.
  - Until the changes take effect, might get “permission denied” errors when try to access files in your bucket.


---


### Using S3 Bucket Policies

give CloudFront OAI access to files in S3 bucket by <font color=red> creating or updating the bucket policy </font>
1. Using the S3 bucket’s **Permissions** tab in the [S3 console](https://console.aws.amazon.com/s3/home).
2. Using [PutBucketPolicy](https://docs.aws.amazon.com/AmazonS3/latest/API/API_PutBucketPolicy.html) in the S3 API.
3. Using the [CloudFront console](https://console.aws.amazon.com/cloudfront/home).
   - When you add an OAI to your origin settings in the CloudFront console
   - choose **Yes, Update Bucket Policy** to tell CloudFront to update the bucket policy on your behalf.
   - If you update the bucket policy manually:
     - Specify the correct OAI as the `Principal` in the policy.
     - Give the OAI the permissions it needs to access objects on behalf of viewers.


#### Specify an OAI as the `Principal`

To specify an OAI as the `Principal` in an S3 bucket policy



- use the OAI’s Amazon Resource Name (ARN), which includes the OAI’s ID.
  - To find the OAI’s ID
    - use [Origin Access Identity page](https://console.aws.amazon.com/cloudfront/home?region=us-east-1#oai:) in the CloudFront console
    - use [ListCloudFrontOriginAccessIdentities](https://docs.aws.amazon.com/cloudfront/latest/APIReference/API_ListCloudFrontOriginAccessIdentities.html) in the CloudFront API.

  ```json
  "Principal": {
      "AWS": "arn:aws:iam::cloudfront:user/EH1HDMB1FH2TC"
      // CloudFront Origin Access Identity
  }
  ```

- or specify an OAI as the `Principal` by using its S3 canonical ID
  - find the OAI’s canonical ID in the same ways that you find its ID.

  ```json
  "Principal": {
      "CanonicalUser": "79a59df900b949e55d96a1e698fbacedfd6e09d98eacf8f8d5218e7cd47ef2be"
  }
  ````


OAI’s ARN:
- make it easier to understand the bucket policy.
- easier to understand who the bucket policy is granting access to.

S3 canonical IDs:
- can refer to different kinds of AWS identities, not just CloudFront OAIs
- so difficult to determine which identity a canonical ID refers to.
- Also, when use the OAI’s canonical ID in a bucket policy, AWS replaces the canonical ID with the OAI’s ARN.
  - When you write a policy that specifies an OAI’s canonical ID and then later view the same policy, the canonical ID has been replaced by the corresponding ARN.

---


#### Give Permissions to an OAI

To give the OAI the permissions to access objects in the S3 bucket, use keywords in the policy that relate to specific S3 API operations.
- For example, the `s3:GetObject` permission allows the OAI to read objects in the bucket.
- [Specifying Permissions in a Policy](https://docs.aws.amazon.com/AmazonS3/latest/dev/using-with-s3-actions.html) in the _Amazon Simple Storage Service Developer Guide_.


---

#### S3 Bucket Policy Examples

S3 bucket policies that grant access to a CloudFront OAI.

* OAI’s ID: `EH1HDMB1FH2TC`
* S3 bucket: `awsexamplebucket`

**Example S3 bucket policy that gives the OAI read access**
- allows the OAI to read objects in the specified bucket (`s3:GetObject`).

```json
{
    "Version": "2012-10-17",
    "Id": "PolicyForCloudFrontPrivateContent",
    "Statement": [
        {
            "Effect": "Allow",
            "Principal": {"AWS": "arn:aws:iam::cloudfront:user/EH1HDMB1FH2TC"},
            "Action": "s3:GetObject",
            "Resource": "arn:aws:s3:::awsexamplebucket/*"
        }
    ]
}
```

**Example S3 bucket policy that gives the OAI read and write access**
- allows the OAI to read and write objects in the specified bucket (`s3:GetObject` and `s3:PutObject`).
- This allows viewers to upload files to your S3 bucket through CloudFront.

```json
{
    "Version": "2012-10-17",
    "Id": "PolicyForCloudFrontPrivateContent",
    "Statement": [
        {
            "Effect": "Allow",
            "Principal": {"AWS": "arn:aws:iam::cloudfront:user/EH1HDMB1FH2TC"},
            "Action": [ "s3:GetObject", "s3:PutObject" ],
            "Resource": "arn:aws:s3:::aws-example-bucket/*"
        }
    ]
}
```

---


### Updating S3 Object ACLs

give a CloudFront OAI access to files in an S3 bucket by creating or updating the file’s ACL in the following ways:
* Using the S3 object’s **Permissions** tab in the S3 console.
* Using [PutObjectAcl](https://docs.aws.amazon.com/AmazonS3/latest/API/API_PutObjectAcl.html) in the S3 API.


to grant access to an OAI using an ACL
- must specify the OAI using its S3 canonical user ID
- in CloudFront API, use the value of the `S3CanonicalUserId` element that was returned when you created the OAI, or call [ListCloudFrontOriginAccessIdentities](https://docs.aws.amazon.com/cloudfront/latest/APIReference/API_ListCloudFrontOriginAccessIdentities.html) in the CloudFront API.


---


### Using an OAI in S3 Regions that Support Only Signature Version 4 Authentication

Newer S3 Regions require that you use <font color=red> Signature Version 4 </font> for authenticated requests.
- when create an origin access identity and add it to a CloudFront distribution,
- CloudFront typically uses Signature Version 4 for authentication when it requests files in your S3 bucket.

* `DELETE`, `GET`, `HEAD`, `OPTIONS`, and `PATCH` requests are supported without qualifications.

* to submit `PUT` requests to CloudFront to upload files to your S3 bucket
  * must add an `x-amz-content-sha256` header to the request.
  * The header value must contain an SHA256 hash of the body of the request.

* `POST` requests are not supported.








.
