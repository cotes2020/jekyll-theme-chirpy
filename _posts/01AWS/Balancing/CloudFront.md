

---



## Use IAM to Control API access
￼
- an IAM policy that enforces SSL.
- It grants application permission to access all of CloudFront.
- it also requires secure transport: must use SSL or TLS.
- In an IAM policy, you can specify any and all API actions that CloudFront offers. The action name must be prefixed with the lowercase string cloudfront:.
  - An example action name: cloudfront:GetDistributionConfig
- To learn more about AWS Authentication and Access Control, select the link. https://docs.aws.amazon.com/AmazonCloudFront/latest/DeveloperGuide/UsingWithIAM.html


encrypting data
￼
basic review of encryption.
- get your key, either hardware or software.
- This example uses a symmetric data key to encrypt plaintext data.
- This process results in encrypted data that can be stored in a database or in a S3 bucket.
- When it’s time to decrypt the data and read it, you take the combination of the symmetric data key and the master key, which creates an encrypted data key.
- unlocks the encrypted document, read the document as plaintext data.




---


# Request and Response Behavior

Request and Response Behavior for S3 Origins

Topics
- How CloudFront Processes HTTP and HTTPS Requests
- How CloudFront Processes and Forwards Requests to Your S3 Origin Server
  - Caching Duration and Minimum TTL
  - Client IP Addresses
  - Conditional GETs
  - Cookies
  - Cross-Origin Resource Sharing (CORS)
  - GET Requests That Include a Body
  - HTTP Methods
  - HTTP Request Headers That CloudFront Removes or Updates
  - Maximum Length of a Request and Maximum Length of a URL
  - OCSP Stapling
  - Protocols
  - Query Strings
  - Origin Connection Timeout and Attempts
  - Origin Response Timeout
  - Simultaneous Requests for the Same Object (Traffic Spikes)
- How CloudFront Processes Responses from Your S3 Origin Server

How CloudFront Processes HTTP and HTTPS Requests
For S3 origins, CloudFront accepts requests in both HTTP and HTTPS protocols for objects in a CloudFront distribution by default.
- CloudFront then forwards the requests to your S3 bucket using the same protocol in which the requests were made.
For custom origins, when create your distribution, can specify how CloudFront accesses origin: HTTP only, or matching the protocol that is used by the viewer. For more information about how CloudFront handles HTTP and HTTPS requests for custom origins, see Protocols.
For information about how to restrict your web distribution so that end users can only access objects using HTTPS, see Using HTTPS with CloudFront. (This option doesn't apply to RTMP distributions, which use the RTMP protocol.)
Note: The charge for HTTPS requests is higher than the charge for HTTP requests.
For more information about billing rates, go to the CloudFront pricing plan.


How CloudFront Processes and Forwards Requests to Your S3 Origin Server

Caching Duration and Minimum TTL
For web distributions, to control how long your objects stay in a CloudFront cache before CloudFront forwards another request to your origin, you can:
- Configure your origin to add a Cache-Control or an Expires header field to each object.
- Specify a value for Minimum TTL in CloudFront cache behaviors.
- Use the default value of 24 hours.
- For more information, see Managing How Long Content Stays in an Edge Cache (Expiration).

Client IP Addresses
- If a viewer sends a request to CloudFront and does not include an X-Forwarded-For request header,
  - CloudFront gets the IP address of the viewer from the TCP connection,
  - adds an X-Forwarded-For header that includes the IP address,
  - and forwards the request to the origin.
  - For example, if CloudFront gets the IP address 192.0.2.2 from the TCP connection, it forwards the following header to the origin:
  - X-Forwarded-For: 192.0.2.2
- If a viewer sends a request to CloudFront and includes an X-Forwarded-For request header,
  - CloudFront gets the IP address of the viewer from the TCP connection,
  - appends it to the end of the X-Forwarded-For header,
  - and forwards the request to the origin.
  - For example, if the viewer request includes X-Forwarded-For: 192.0.2.4, 192.0.2.3 and CloudFront gets the IP address 192.0.2.2 from the TCP connection, it forwards the following header to the origin:
  - X-Forwarded-For: 192.0.2.4,192.0.2.3,192.0.2.2
Note  The X-Forwarded-For header contains IPv4 addresses (such as 192.0.2.44) and IPv6 addresses (such as 2001:0db8:85a3:0000:0000:8a2e:0370:7334).

Conditional GETs
- When CloudFront receives a request for an object that has expired from an edge cache,
  - CloudFront forwards the request to the S3 origin
  - either to get the latest version of the object
  - or to get confirmation from S3 that the CloudFront edge cache already has the latest version.
  - When S3 originally sent the object to CloudFront, it included an ETag value and a LastModified value in the response.
  - In the new request that CloudFront forwards to S3, CloudFront adds one or both of the following:
  - An If-Match or If-None-Match header that contains the ETag value for the expired version of the object.
  - An If-Modified-Since header that contains the LastModified value for the expired version of the object.
- S3 uses this information to determine
  - whether the object has been updated
  - whether to return the entire object to CloudFront
  - or to return only an HTTP 304 status code (not modified).

Cookies
S3 doesn't process cookies. If you configure a cache behavior to forward cookies to an S3 origin, CloudFront forwards the cookies, but S3 ignores them.
All future requests for the same object, regardless if you vary the cookie, are served from the existing object in the cache.


Cross-Origin Resource Sharing (CORS)
- If you want CloudFront to respect S3 cross-origin resource sharing settings, configure CloudFront to forward selected headers to S3.
- For more information, see Caching Content Based on Request Headers.


GET Requests That Include a Body
If a viewer GET request includes a body, CloudFront returns an HTTP status code 403 (Forbidden) to the viewer.
HTTP Methods
If you configure CloudFront to process all of the HTTP methods that it supports, CloudFront accepts the following requests from viewers and forwards them to your S3 origin:
- DELETE
- GET
- HEAD
- OPTIONS
- PATCH
- POST
- PUT
- CloudFront always caches responses to GET and HEAD requests.
- can also configure CloudFront to cache responses to OPTIONS requests.
- does not cache responses to requests that use the other methods.
If you use an S3 bucket as the origin for your distribution and if you use CloudFront origin access identities, POST requests aren't supported in some S3 Regions and PUT requests in those Regions require an additional header. For more information, see Using an OAI in S3 Regions that Support Only Signature Version 4 Authentication.
If you want to use multi-part uploads to add objects to an S3 bucket, you must add a CloudFront origin access identity to your distribution and grant the origin access identity the needed permissions. For more information, see Restricting Access to S3 Content by Using an Origin Access Identity.
Important
If you configure CloudFront to accept and forward to S3 all of the HTTP methods that CloudFront supports, you must create a CloudFront origin access identity to restrict access to your S3 content and grant the origin access identity the required permissions. For example, if you configure CloudFront to accept and forward these methods because you want to use PUT, you must configure S3 bucket policies or ACLs to handle DELETE requests appropriately so viewers can't delete resources that you don't want them to. For more information, see Restricting Access to S3 Content by Using an Origin Access Identity.
For information about the operations supported by S3, see the S3 documentation.

HTTP Request Headers That CloudFront Removes or Updates
CloudFront removes or updates some headers before forwarding requests to your S3 origin.
For most headers this behavior is the same as for custom origins. For a full list of HTTP request headers and how CloudFront processes them, see HTTP Request Headers and CloudFront Behavior (Custom and S3 Origins).

Maximum Length of a Request and Maximum Length of a URL
The maximum length of a request, including the path, the query string (if any), and headers, is 20,480 bytes.
CloudFront constructs a URL from the request. The maximum length of this URL is 8192 bytes.
If a request or a URL exceeds these maximums, CloudFront returns HTTP status code 413, Request Entity Too Large, to the viewer, and then terminates the TCP connection to the viewer.

OCSP Stapling
When a viewer submits an HTTPS request for an object, either CloudFront or the viewer must confirm with the certificate authority (CA) that the SSL certificate for the domain has not been revoked. OCSP stapling speeds up certificate validation by allowing CloudFront to validate the certificate and to cache the response from the CA, so the client doesn't need to validate the certificate directly with the CA.
The performance improvement of OCSP stapling is more pronounced when CloudFront receives a lot of HTTPS requests for objects in the same domain. Each server in a CloudFront edge location must submit a separate validation request. When CloudFront receives a lot of HTTPS requests for the same domain, every server in the edge location soon has a response from the CA that it can "staple" to a packet in the SSL handshake; when the viewer is satisfied that the certificate is valid, CloudFront can serve the requested object. If your distribution doesn't get much traffic in a CloudFront edge location, new requests are more likely to be directed to a server that hasn't validated the certificate with the CA yet. In that case, the viewer separately performs the validation step and the CloudFront server serves the object. That CloudFront server also submits a validation request to the CA, so the next time it receives a request that includes the same domain name, it has a validation response from the CA.

Protocols
CloudFront forwards HTTP or HTTPS requests to the origin server based on the protocol of the viewer request, either HTTP or HTTPS.
Important
If your S3 bucket is configured as a website endpoint, you cannot configure CloudFront to use HTTPS to communicate with your origin because S3 doesn't support HTTPS connections in that configuration.

Query Strings
For web distributions, you can configure whether CloudFront forwards query string parameters to your S3 origin. For RTMP distributions, CloudFront does not forward query string parameters. For more information, see Caching Content Based on Query String Parameters.

Origin Connection Timeout and Attempts
Origin connection timeout is the number of seconds that CloudFront waits when trying to establish a connection to the origin.
Origin connection attempts is the number of times that CloudFront attempts to connect to the origin.
Together, these settings determine how long CloudFront tries to connect to the origin before failing over to the secondary origin (in the case of an origin group) or returning an error response to the viewer. By default, CloudFront waits as long as 30 seconds (3 attempts of 10 seconds each) before attempting to connect to the secondary origin or returning an error response. You can reduce this time by specifying a shorter connection timeout, fewer attempts, or both.
For more information, see Controlling Origin Timeouts and Attempts.

Origin Response Timeout
The origin response timeout, also known as the origin read timeout or origin request timeout, applies to both of the following:
- The amount of time, in seconds, that CloudFront waits for a response after forwarding a request to the origin.
- The amount of time, in seconds, that CloudFront waits after receiving a packet of a response from the origin and before receiving the next packet.
CloudFront behavior depends on the HTTP method of the viewer request:
- GET and HEAD requests – If the origin doesn’t respond within 30 seconds or stops responding for 30 seconds, CloudFront drops the connection. If the specified number of origin connection attempts is more than 1, CloudFront tries again to get a complete response. CloudFront tries up to 3 times, as determined by the value of the origin connection attempts setting. If the origin doesn’t respond during the final attempt, CloudFront doesn’t try again until it receives another request for content on the same origin.
- DELETE, OPTIONS, PATCH, PUT, and POST requests – If the origin doesn’t respond within 30 seconds, CloudFront drops the connection and doesn’t try again to contact the origin. The client can resubmit the request if necessary.
You can’t change the response timeout for an S3 origin (an S3 bucket that is not configured with static website hosting).

Simultaneous Requests for the Same Object (Traffic Spikes)
When a CloudFront edge location receives a request for an object and either the object isn't currently in the cache or the object has expired, CloudFront immediately sends the request to your S3 origin. If there's a traffic spike—if additional requests for the same object arrive at the edge location before S3 responds to the first request—CloudFront pauses briefly before forwarding additional requests for the object to your origin. Typically, the response to the first request will arrive at the CloudFront edge location before the response to subsequent requests. This brief pause helps to reduce unnecessary load on S3. If additional requests are not identical because, for example, you configured CloudFront to cache based on request headers or query strings, CloudFront forwards all of the unique requests to your origin.
When the response from the origin includes a Cache-Control: no-cache header, CloudFront typically forwards the next request for the same object to the origin to determine whether the object has been updated. However, when there's a traffic spike and CloudFront pauses after forwarding the first request to your origin, multiple viewer requests might arrive before CloudFront receives a response from the origin. When CloudFront receives a response that contains a Cache-Control: no-cache header, it sends the object in the response to the viewer that made the original request and to all of the viewers that requested the object during the pause. After the response arrives from the origin, CloudFront forwards the next viewer request for the same object to the origin. In CloudFront access logs, the first request is identified as a Miss in the x-edge-result-type column, and all subsequent requests that CloudFront received during the pause are identified as a Hit. For more information about access log file format, see Web Distribution Log File Format.
How CloudFront Processes Responses from Your S3 Origin Server
This topic contains information about how CloudFront processes responses from your S3 origin.
Topics
- Canceled Requests
- HTTP Response Headers That CloudFront Removes or Updates
- Maximum File Size
- Redirects
Canceled Requests
If an object is not in the edge cache, and if a viewer terminates a session (for example, closes a browser) after CloudFront gets the object from your origin but before it can deliver the requested object, CloudFront does not cache the object in the edge location.
HTTP Response Headers That CloudFront Removes or Updates
CloudFront removes or updates the following header fields before forwarding the response from your S3 origin to the viewer:
- Set-Cookie – If you configure CloudFront to forward cookies, it will forward the Set-Cookie header field to clients. For more information, see Caching Content Based on Cookies.
- Trailer
- Transfer-Encoding – If your S3 origin returns this header field, CloudFront sets the value to chunked before returning the response to the viewer.
- Upgrade
- Via – CloudFront sets the value to the following in the response to the viewer:Via: http-version alphanumeric-string.cloudfront.net (CloudFront)For example, if the client makes a request over HTTP/1.1, the value is something like the following:Via: 1.1 1026589cc7887e7a0dc7827b4example.cloudfront.net (CloudFront)
Maximum File Size
The maximum size of a response body that CloudFront will return to the viewer is 20 GB. This includes chunked transfer responses that don't specify the Content-Length header value.

Redirects
You can configure an S3 bucket to redirect all requests to another host name; this can be another S3 bucket or an HTTP server. If you configure a bucket to redirect all requests and if the bucket is the origin for a CloudFront distribution, we recommend that you configure the bucket to redirect all requests to a CloudFront distribution using either the domain name for the distribution (for example, d111111abcdef8.cloudfront.net) or an alternate domain name (a CNAME) that is associated with a distribution (for example, example.com). Otherwise, viewer requests bypass CloudFront, and the objects are served directly from the new origin.
Note
If you redirect requests to an alternate domain name, you must also update the DNS service for your domain by adding a CNAME record. For more information, see Using Custom URLs for Files by Adding Alternate Domain Names (CNAMEs).
Here's what happens when you configure a bucket to redirect all requests:
1. A viewer (for example, a browser) requests an object from CloudFront.
1. CloudFront forwards the request to the S3 bucket that is the origin for your distribution.
1. S3 returns an HTTP status code 301 (Moved Permanently) as well as the new location.
1. CloudFront caches the redirect status code and the new location, and returns the values to the viewer. CloudFront does not follow the redirect to get the object from the new location.
1. The viewer sends another request for the object, but this time the viewer specifies the new location that it got from CloudFront:
- If the S3 bucket is redirecting all requests to a CloudFront distribution, using either the domain name for the distribution or an alternate domain name, CloudFront requests the object from the S3 bucket or the HTTP server in the new location. When the new location returns the object, CloudFront returns it to the viewer and caches it in an edge location.
- If the S3 bucket is redirecting requests to another location, the second request bypasses CloudFront. The S3 bucket or the HTTP server in the new location returns the object directly to the viewer, so the object is never cached in a CloudFront edge cache.
1.
