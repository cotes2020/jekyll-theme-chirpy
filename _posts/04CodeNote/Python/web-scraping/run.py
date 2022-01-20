import requests

res = requests.get('https://codedamn.com')

print(res.text)
print(res.status_code)