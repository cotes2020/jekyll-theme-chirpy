-----------

### Setting up Cloud Goat

------------

- Create an account at [console](https://aws.amazon.com/console/) and add a valid debit card.

<img width="835" height="423" alt="image" src="https://github.com/user-attachments/assets/e75b4565-2272-4cf5-8582-07c0c1047c59" />

- Create a user-:

<img width="1630" height="286" alt="image" src="https://github.com/user-attachments/assets/db4ddc6c-ad58-4424-923b-a96db74afc6a" />

- Create an `administrator policy`-:

<img width="1453" height="604" alt="image" src="https://github.com/user-attachments/assets/17a5f73b-674b-4a8f-a780-0aca75fd246a" />

- Generate  an access key next. go to IAM->Access Management-> IAM Users, click on the user, and security credentials  and create Access Key-:

<img width="1326" height="760" alt="image" src="https://github.com/user-attachments/assets/ed4f5263-4dd6-4126-8ec7-b28a9bd684a7" />

- Save somewhere-:

<img width="1513" height="404" alt="image" src="https://github.com/user-attachments/assets/0c31ac8b-7b35-4217-a415-021f2ee83212" />


------------

### Dependencies

-------------

-  `awscli` -:

```bash
sudo apt install awscli
aws configure
```

- Prompt-:
```
Access Key Id-> Key you generated above
Secret Access Key-> None
Default region-> use whatever region your account is in e.g us-east-2, check the top right console of your AWS console in what region you are in
default output format-> json
```

<img width="850" height="257" alt="image" src="https://github.com/user-attachments/assets/46ae3dc7-b98a-47b1-8cbb-3291ff9da2ce" />

- Setting up terraform-:

```
wget https://releases.hashicorp.com/terraform/0.12.29/terraform_0.12.29_linux_amd64.zip
unzip terraform_0.12.29_linux_amd64.zip
sudo cp terraform /usr/local/bin/
rm get-pip.py & rm terraform & rm terraform_0.12.29_linux_amd64.zip
```

- Setting up the goat itself-:

```
pipx install cloudgoat
```
- Configuring Cloudgoat-:

```sh
#default as anything asked
cloudgoat config aws
cloudgoat config whitelist --auto
```

<img width="1778" height="167" alt="image" src="https://github.com/user-attachments/assets/6089ba78-7a2e-418e-a5d5-6635d13d4b6e" />

- Final verification

```bash
# aws creds
aws sts get-caller-identity --profile default
# terraform available
terraform version
# cloudgoat config whitelist
cloudgoat config whitelist
```

- Note-: Cloudgoat will not use your `default` AWS profile automatically.This is to prevent you from accidentally deploying vulnerable infrastructure to prevent accidentally vulnerable infrastructure to a production account. You must explicitly pass `--profile default`.
- Most scenarios use free-tier or near free resources but some spin up EC2 instances or other billable services.
- Always run

```sh
cloudgoat destroy <scenario-name> --profile default
```

- Verify with

```
aws iam list-users --profile default | grep cgid
aws iam list-policies --scope local --profile default | grep cgid
```
