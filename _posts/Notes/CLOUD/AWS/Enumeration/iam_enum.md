----------

### Enumeratimg IAM
### Vulnerable Lab( Cloudgoat)

----------

- Command

```bash
cloudgoat create iam_enum_basics
```
<img width="905" height="201" alt="image" src="https://github.com/user-attachments/assets/a0e9e55d-764c-4581-b84c-bfaac80aec1e" />

- Configure bob creds as provided-:

<img width="880" height="137" alt="image" src="https://github.com/user-attachments/assets/d2108fbe-a5fa-4aca-b002-3f0877856277" />

- Enumerate attached managed policies and read their metadata-:

- User-:

```
aws iam list-users --profile bob | jq
```
<img width="858" height="337" alt="image" src="https://github.com/user-attachments/assets/c25c5a85-0e8e-435d-883e-bc8828c69606" />

- Group-:

```bash
aws iam list-groups --profile bob | jq
```

<img width="1246" height="264" alt="image" src="https://github.com/user-attachments/assets/8bdef635-dec9-42ff-b68b-1fe7eae5e187" />

- Roles-:

```bash
aws iam list-roles --profile bob | jq
```

<img width="914" height="401" alt="image" src="https://github.com/user-attachments/assets/cce2f78e-135d-408c-b535-bddf48b0eb2b" />

- Enumerating groups that a specific user belongs to-:

```bash
aws iam list-groups-for-user --user-name <u> --profile <p>
```

<img width="1203" height="257" alt="image" src="https://github.com/user-attachments/assets/a26bfe3f-1525-42a3-92c3-e731e63efeef" />

- Using query -:

```bash
aws iam list-roles --profile <u> --query "Roles[*].[RoleName,Path]" --output table
```
<img width="1132" height="179" alt="image" src="https://github.com/user-attachments/assets/1622afa8-7379-475e-9122-d2987026cdae" />

> Roles explanation-:
> Resource Explorer (Resource discovery and indexing)
> AWS SUPPORT: (Enabling Support Related Diagnostics and Indexing)
> Trusted Advisor (Allowing automated health and best practice checks)

-----------

### Enumerating Policies

-----------

- Inline user policies-:

```sh
aws iam list-user-policies --user-name <u> --profile <p>
```
<img width="944" height="160" alt="image" src="https://github.com/user-attachments/assets/90c192ce-dfdf-4f97-be4c-0429144e93cc" />

-  Managed user policiees-:

```
aws iam list-attached-user-policies --user-name <u> --profile <p>
```

<img width="1124" height="276" alt="image" src="https://github.com/user-attachments/assets/c4ec7555-e94a-43a0-bfa3-94246810bb69" />

- Group/Role policies-:

```bash
#Inline group policies
aws iam list-group-policies --group-name <u> --profile <p>
#attached group policies
aws iam list-attached-group-policies --group-name <u> --profile <p>

#Inline role policies
aws iam list-role-policies --role-name <u> --profile <p>
#attached role policies
aws iam list-attached-role-policies --role-name <u> --profile <p>

```

<img width="1183" height="400" alt="image" src="https://github.com/user-attachments/assets/53da164b-949b-4645-a487-be264ec3866d" />

<img width="1063" height="309" alt="image" src="https://github.com/user-attachments/assets/79a4e258-336c-467e-90b4-8df2a828f111" />


-----------

### Examining User Policy Rules

------------

- We exposed that there is one inline and two managed policies for our user `cg-bob-*` as well as one one  role policy for role `cg-flag4-role-*`
- Let's look at the metadata for the policy name and the ARN (Amazon Resource Name) but have not looked at the metadata or policy document.
- Inspecitng a custom managed policy-:

```bash
#Metadata about a managed Iam policy
aws iam get-policy --policy-arn <arn> --profile <p>

# Get Json policy document
aws iam get-policy-version --policy-arn <arn> --version-id <v> --profile <p>
```

<img width="1701" height="597" alt="image" src="https://github.com/user-attachments/assets/d992cdaa-b356-4cd6-bd35-b44d89126dcd" />

<img width="1643" height="420" alt="image" src="https://github.com/user-attachments/assets/b6687cbd-2130-441f-b942-e2485fd408d7" />

------------

### Investigating user, role, group inline policy

-------------

- Inspecting an inline role policy

```bash
aws iam get-user-policy --user-name <u> --policy-name <p> --profile <p>
aws iam get-group-policy --group-name <u> --policy-name <p> --profile <p>
aws iam get-role-policy --role-name <u> --policy-name <p> --profile <p>
```

<img width="1723" height="347" alt="image" src="https://github.com/user-attachments/assets/dc77e5ee-e1fc-4883-8dd6-8e6d97bbcf17" />

<img width="1505" height="332" alt="image" src="https://github.com/user-attachments/assets/b0cd4c51-bd93-4359-9705-01f3398ab86a" />


- Investigating the metadata-:

```bash
aws iam get-role --role-name cg-flag4-role-c --profile bob | jq
aws iam get-user --user-name cg-bob-c --profile bob | jq
aws iam get-group --group-name cg-bob- --profile bob | jq  
```
<img width="1079" height="753" alt="image" src="https://github.com/user-attachments/assets/a4ff1cf6-3b44-4428-b501-6cff06be6c90" />


-------------
