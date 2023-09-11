









## Top 5 Vulnerabilities to Test for in AWS

1. Testing S3 bucket configuration and permissions flaws
2. Targeting and compromising AWS IAM keys
3. Cloudfront/WAF Misconfiguration Bypasses
4. Establishing private-cloud access through Lambda backdoor functions
5. Cover tracks by obfuscating Cloudtrail logs



The following basic tools can also help identify basic flaws:
1. AWS Inspector (designed for the security of apps deployed on AWS)
1. Nmap (network discovery and service enumeration)
1. Rhino Security Lab’s BucketHead (Identifying misconfigured AWS S3 Buckets)


https://geekflare.com/aws-vulnerability-scanner/

https://github.com/toniblyx/my-arsenal-of-aws-security-tools





1. Check the Service Level Agreement (SLA)
   - to ensure the appropriate Pen Test policy has been identified, and R&R clearly defined. In many cases, elements of Pen Testing are spread across multiple players such as the CSP and the client, so it is necessary to clearly document who does what, and when it is to be done.
2. Governance & Compliance requirements need to be understood.
   - Factors need to include which party will be responsible to define, configure and validate security settings required to meet applicable regulatory controls for your business. This includes providing appropriate evidence for audits and inspections.
3. Security and Vulnerability Patching and general maintenance responsibilities and timeframes need to be documented.
   - You as the client may have responsibility for maintaining your virtual images and resources, but the CSP will likely be accountable for the underlying physical hardware systems. Both need to be actively managed, along with all network and SAN equipment.
4. Computer access and Internet usage policies need to be clearly defined and properly implemented
   - to ensure appropriate traffic is permitted while inappropriate traffic is denied at the perimeter.
5. Ensure all unused ports are disabled and unused protocols are either not installed or disabled and locked down to prevent unauthorized activation.
6. Data encryption
   - both in transit and at rest
   - Ensure that encryption is either set as the default or that appropriate steps are implemented to ensure it is activated.
7. requirements for Two Factor Authentication and One Time Passwords are implemented and actively securing network access.
   - Check if your CSP permits any bypass scenarios.
8. SSL is only as good as the Certificate Authority (CA) that issued the certificates.
   - Ensure SSL is active, and that a reputable CA stands behind the certificates.
9.  Hold your CSP accountable and validate they are using appropriate security controls for physical and logical access to the data center and the infrastructure hardware with which they provide your services.
10. Know your CSP’s policy and procedures relative to data disclosure to third parties, both for unauthorized access and providing data when requested or subpoenaed by law enforcement.
.

who is in charge of security
logs
Giving away too many privileges
Having powerful users and broad roles
Relying heavily on passwords
Exposed secrets and keys
Not taking root seriously
Putting everything in one VPC or account
Leaving wide open connections
encryption
Mistakes, not vulnerabilities

.
