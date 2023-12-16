




uvicorn main:app --reload --port 3000



```py
from fastapi import FastAPI, Request, Form
from typing import Optional
from starlette.responses import RedirectResponse

from onelogin.saml2.auth import OneLogin_Saml2_Auth
from onelogin.saml2.settings import OneLogin_Saml2_Settings
from onelogin.saml2.utils import OneLogin_Saml2_Utils

app = FastAPI()

saml_settings = {
  "strict": False, # can set to True to see problems such as Time skew/drift
  "debug": True,
  "idp": {
    "entityId": "test-saml-client",
    "singleSignOnService": {
      "url": "http://127.0.0.1:8081/auth/realms/test/protocol/saml",
      "binding": "urn:oasis:names:tc:SAML:2.0:bindings:HTTP-Redirect"
    },
    "x509cert": "MIIClzCCAX8CBgF6A0sAhDANBgkqhkiG9w0BAQsFADAPMQ0wCwYDVQQDDAR0ZXN0MB4XDTIxMDYxMzAyNTMwNFoXDTMxMDYxMzAyNTQ0NFowDzENMAsGA1UEAwwEdGVzdDCCASIwDQYJKoZIhvcNAQEBBQADggEPADCCAQoCggEBAK97NlCcNOhtH0a0wz5boYKb7TaxogxnlyysOWUre1uI8SC6EBV3G5DHMdg4aWXwuXwy61+JJu70xNzJj155MJ+atGS7eLrxxGl0DNoLu/LU7Vhht+j09MZt5J60DnS76H3pkvzAtRfd1P/d5JEFzWYkI4drBJccYX/nrrx2KZBkXOjwjVcEhsyK5ykA0LX+M+yFDy2w8qEWhxHuSL6enzw8IZ7qdtsF8SHqw7cdCgCJU6+0dxaRAAqmzMkO7BDEkwCJl0M8VaOPGo/SnZIAMYHLIUg1x0h/ecST4NPdqAwgDGtWAcD+Gp7Lr7xfBbKKqnLfg2PJdjs7Z0+NFOeVTvcCAwEAATANBgkqhkiG9w0BAQsFAAOCAQEAeJ2r2yoaQAo6v8MC6iAobOeJoBoezQg/OSQqeA9lygMWmGHpDIjSV7m3PCXwf5H9/NpHgBLt8y5PcjEs99uPfPeUBV/qitTFMuznMyr35e60iaHSdhZVjyCmrKgnIuGa07lng2wFabtpijqzbQJ99kYsWxbBDgbdVnt3jxohG1KKaXkGMyy7suwPgwrbwXfDrpyyj33NT/Dk/2W4Fjrjg8rIkuQypwB0SQLG1cZL9Z2AgW39JeVnP/sOH1gNpCCQwbpgE9hEed80jsYWlvucnFm2aHBtGV+/7/7N3qRRpIvzrNVJoznqDDWU41RxS0NphAwX2ZqprWvN+SCEEhPGGQ=="
  },
  "sp": {
    "entityId": "test-saml-client",
    "assertionConsumerService": {
      "url": "http://127.0.0.1:3000/api/saml/callback",
      "binding": "urn:oasis:names:tc:SAML:2.0:bindings:HTTP-POST"
    },
    "x509cert": "MIIClzCCAX8CBgF6A0sAhDANBgkqhkiG9w0BAQsFADAPMQ0wCwYDVQQDDAR0ZXN0MB4XDTIxMDYxMzAyNTMwNFoXDTMxMDYxMzAyNTQ0NFowDzENMAsGA1UEAwwEdGVzdDCCASIwDQYJKoZIhvcNAQEBBQADggEPADCCAQoCggEBAK97NlCcNOhtH0a0wz5boYKb7TaxogxnlyysOWUre1uI8SC6EBV3G5DHMdg4aWXwuXwy61+JJu70xNzJj155MJ+atGS7eLrxxGl0DNoLu/LU7Vhht+j09MZt5J60DnS76H3pkvzAtRfd1P/d5JEFzWYkI4drBJccYX/nrrx2KZBkXOjwjVcEhsyK5ykA0LX+M+yFDy2w8qEWhxHuSL6enzw8IZ7qdtsF8SHqw7cdCgCJU6+0dxaRAAqmzMkO7BDEkwCJl0M8VaOPGo/SnZIAMYHLIUg1x0h/ecST4NPdqAwgDGtWAcD+Gp7Lr7xfBbKKqnLfg2PJdjs7Z0+NFOeVTvcCAwEAATANBgkqhkiG9w0BAQsFAAOCAQEAeJ2r2yoaQAo6v8MC6iAobOeJoBoezQg/OSQqeA9lygMWmGHpDIjSV7m3PCXwf5H9/NpHgBLt8y5PcjEs99uPfPeUBV/qitTFMuznMyr35e60iaHSdhZVjyCmrKgnIuGa07lng2wFabtpijqzbQJ99kYsWxbBDgbdVnt3jxohG1KKaXkGMyy7suwPgwrbwXfDrpyyj33NT/Dk/2W4Fjrjg8rIkuQypwB0SQLG1cZL9Z2AgW39JeVnP/sOH1gNpCCQwbpgE9hEed80jsYWlvucnFm2aHBtGV+/7/7N3qRRpIvzrNVJoznqDDWU41RxS0NphAwX2ZqprWvN+SCEEhPGGQ==",
  }
}

async def prepare_from_fastapi_request(request, debug=False):
  form_data = await request.form()
  rv = {
    "http_host": request.client.host,
    "server_port": request.url.port,
    "script_name": request.url.path,
    "post_data": { },
    "get_data": { }
    # Advanced request options
    # "https": "",
    # "request_uri": "",
    # "query_string": "",
    # "validate_signature_from_qs": False,
    # "lowercase_urlencoding": False
  }
  if (request.query_params):
    rv["get_data"] = request.query_params,
  if "SAMLResponse" in form_data:
    SAMLResponse = form_data["SAMLResponse"]
    rv["post_data"]["SAMLResponse"] = SAMLResponse
  if "RelayState" in form_data:
    RelayState = form_data["RelayState"]
    rv["post_data"]["RelayState"] = RelayState
  return rv

@app.get("/")
async def root():
  return { "message": "Hello World" }

@app.post("/test")
async def test(request: Request, p1: Optional[str] = Form(None), p2: Optional[str] = Form(None)):
  req = await prepare_from_fastapi_request(request)
  return req

@app.get('/api/saml/login')
async def saml_login(request: Request):
  req = await prepare_from_fastapi_request(request)
  auth = OneLogin_Saml2_Auth(req, saml_settings)
  # saml_settings = auth.get_settings()
  # metadata = saml_settings.get_sp_metadata()
  # errors = saml_settings.validate_metadata(metadata)
  # if len(errors) == 0:
  #   print(metadata)
  # else:
  #   print("Error found on Metadata: %s" % (', '.join(errors)))
  callback_url = auth.login()
  response = RedirectResponse(url=callback_url)
  return response

@app.post('/api/saml/callback')
async def saml_login_callback(request: Request):
  req = await prepare_from_fastapi_request(request, True)
  auth = OneLogin_Saml2_Auth(req, saml_settings)
  auth.process_response() # Process IdP response
  errors = auth.get_errors() # This method receives an array with the errors
  if len(errors) == 0:
    if not auth.is_authenticated(): # This check if the response was ok and the user data retrieved or not (user authenticated)
      return "user Not authenticated"
    else:
      return "User authenticated"
  else:
    print("Error when processing SAML Response: %s %s" % (', '.join(errors), auth.get_last_error_reason()))
    return "Error in callback"
```
