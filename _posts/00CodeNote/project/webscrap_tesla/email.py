# import os

# import requests


# def send_email(subject, body, recipient):
#     SENDGRID_API_KEY = os.environ.get('SENDGRID_API_KEY')
#     SENDER_EMAIL = os.environ.get('SENDER_EMAIL')

#     if not SENDGRID_API_KEY:
#         raise ValueError("SendGrid API key is not provided.")
#     if not SENDER_EMAIL:
#         raise ValueError("Sender email is not provided.")

#     url = "https://api.sendgrid.com/v3/mail/send"
#     headers = {
#         "Authorization": f"Bearer {SENDGRID_API_KEY}",
#         "Content-Type": "application/json"
#     }
#     payload = {
#         "personalizations": [{"to": [{"email": recipient}]}],
#         "from": {"email": SENDER_EMAIL},
#         "subject": subject,
#         "content": [{"type": "text/plain", "value": body}]
#     }

#     response = requests.post(url, headers=headers, json=payload)

#     if response.status_code == 202:
#         print("Email sent successfully.")
#     else:
#         print("Failed to send email. Status code:", response.status_code)
#         print("Response:", response.text)

# # Example usage
# subject = "Test Email from GitHub Action"
# body = "This is a test email sent from a GitHub Action workflow."
# recipient = "recipient@example.com"

# send_email(subject, body, recipient)
