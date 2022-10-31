

- [AWS vs GCP](#aws-vs-gcp)
  - [Env](#env)
    - [Install](#install)
  - [s3](#s3)
    - [Making a "client"](#making-a-client)
    - [Checking access to it](#checking-access-to-it)
    - [Checking object](#checking-object)
    - [Uploading a file with a special Content-Encoding](#uploading-a-file-with-a-special-content-encoding)
    - [Downloading and uncompress](#downloading-and-uncompress)


---

# AWS vs GCP

This blog post is a rough attempt to log various activities in both Python libraries:

---


## Env

### Install

**boto3**

```bash
$ pip install boto3
$ emacs ~/.aws/credentials
```

**google-cloud-storage**

```bash
$ pip install google-cloud-storage
$ cat ./google_service_account.json
```

---


## s3

### Making a "client"

**boto3**

**Note**,there are easier shortcuts for this but with this pattern you can have full control over things like like `read_timeout`,`connect_timeout`,etc. with that `confi_params` keyword.

```py
import boto3
from botocore.config import Config
def get_s3_client ( region_name = None,**config_params ):
    options = { "config" :
    Config ( **config_params )}
    if region_name :

        options [ "region_name" ] = region_name
    session = boto3.session.Session ()
    return session.client ( "s3",**options )
``````

**google-cloud-storage**

```py
from google.cloud import storage
def get_gcs_client ():
    return storage.Client.from_service_account_json (
        settings.GOOGLE_APPLICATION_CREDENTIALS_PATH
)
```

### Checking access to it

**boto3**

```py
from botocore.exceptions import ClientError,EndpointConnectionError
try :
    s3_client.head_bucket ( Bucket = bucket_name )
except ClientError as exception :

    if exception.response [ "Error" ][ "Code" ] in ( "403","404" ):
        raise BucketHardError ( f "Unable to connect to bucket= { bucket_name !r}  " f "ClientError ( { exception.response !r} )" )
    else :
        raise
except EndpointConnectionError :

    raise BucketSoftError ( f "Unable to connect to bucket= { bucket.name !r}  " f "EndpointConnectionError" )
else :

    print ( "It exists and we have access to it." )
```

**google-cloud-storage**

```py
from google.api_core.exceptions import BadRequest
try :

    gcs_client.get_bucket ( bucket_name )
except BadRequest as exception :

    raise BucketHardError ( f "Unable to connect to bucket= { bucket_name !r}," f "because bucket not found due to  { exception } " )
else :

    print ( "It exists and we have access to it." )
```

### Checking object

**boto3**

```py
from botocore.exceptions import ClientError
def key_existing ( client,bucket_name,key ):
    """
    return a tuple of (
        key's size if it exists or 0,
        S3 key metadata
    )
    If the object doesn't exist, return None for the metadata.
    """
    try :
        response = client.head_object ( Bucket = bucket_name,Key = key )
        return response [ "ContentLength" ], response.get ( "Metadata" )
    except ClientError as exception :
        if exception.response [ "Error" ][ "Code" ] == "404" :

            return 0, None
        raise
```


**google-cloud-storage**

```py
def key_existing ( client,bucket_name,key ):
    """
    return a tuple of (
        key's size if it exists or 0,
        S3 key metadata
    )
    If the object doesn't exist, return None for the metadata.
    """
    bucket = client.get_bucket ( bucket_name )
    blob = bucket.get_blob ( key )
    if blob :
        return blob.size,blob.metadata
    return 0,None
```



### Uploading a file with a special Content-Encoding

**boto3**

```py
def upload ( file_path,bucket_name,key_name,metadata = None,compressed = False ):
    content_type = get_key_content_type ( key_name )
    metadata = metadata or {}
    extras = {}
    if content_type : extras [ "ContentType" ] = content_type
    if compressed : extras [ "ContentEncoding" ] = "gzip"
    if metadata : extras [ "Metadata" ] = metadata

    with open ( file_path,"rb" ) as f :
        s3_client.put_object ( Bucket = bucket_name,Key = key_name,Body = f,**extras )
```

**google-cloud-storage**
```py
def upload ( file_path,bucket_name,key_name,metadata = None,compressed = False ):
    content_type = get_key_content_type ( key_name )
    metadata = metadata or {}
    bucket = gcs_client.get_bucket ( bucket_name )
    blob = bucket.blob ( key_name )
    if content_type : blob.content_type = content_type
    if compressed : blob.content_encoding = "gzip"

    blob.metadata = metadata
    blob.upload_from_file ( f )
```




### Downloading and uncompress

**boto3**

```py
from io import BytesIO
from gzip import GzipFile
from botocore.exceptions import ClientError
from .utils import iter_lines

def get_stream ( bucket_name,key_name ):
    try :
        response = source.s3_client.get_object ( Bucket = bucket_name,Key = key )
        except ClientError as exception :
            if exception.response [ "Error" ][ "Code" ] == "NoSuchKey" :
    raise KeyHardError ( "key not in bucket" )
            raise
    stream = response [ "Body" ]
    if response.get ( "ContentEncoding" ) == "gzip" :
        body = response [ "Body" ].read ()
        bytestream = BytesIO ( body )
        stream = GzipFile ( None,"rb",fileobj = bytestream )
    for line in iter_lines ( stream ):
        yield line.decode ( "utf-8" )
```

**google-cloud-storage**

```py
from io import BytesIO
from gzip import GzipFile
from botocore.exceptions import ClientError

from .utils import iter_lines

def get_stream ( bucket_name,key_name ):
    bucket = gcs_client.get_bucket ( bucket_name )
    blob = bucket.get_blob ( key )
    if blob is None :
        raise KeyHardError ( "key not in bucket" )
    bytestream = BytesIO ()
    blob.download_to_file ( bytestream )
    bytestream.seek ( 0 )
    for line in iter_lines ( bytestream ):
        yield line.decode ( "utf-8" )
```

**Note!** That here `blob.download_to_file` works a bit like `requests.get()` in that it automatically notices the `Content-Encoding` metadata and does the gunzip on the fly.
