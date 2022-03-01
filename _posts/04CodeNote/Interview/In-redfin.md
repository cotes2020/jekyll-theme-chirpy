


Instructions

Implement a RESTful API that supports 4 operations: 
- adding a user, 
- authenticating, 
- retrieving user details, 
- and logging out. 
- The user details should be stored in a persistent back-end data store such as Sqlite or even a text file. 
- Password and session generation mechanisms should follow current secure development best practices.

While many mechanisms exist for `secure storage and generation`, this exercise is to determine how you would implement these solutions for yourself. 
While you may write in any language you want, an example of libraries and frameworks to use in a Python project would be using `hashlib`, but not Flask’s HTTPBasicAuth - we would like to see how you implement those mechanisms. 
Do not `JWT tokens` unless you’re implementing the algorithms, management, and token creation yourself.

Finally, include a `README` file that describes your project, the overall flow for a user, why you made any specific architectural decisions, and what you would change given appropriate frameworks, libraries, etc.

Don’t spend more than 3 hours on this - the code is not meant to be production-ready, it is meant to show your understanding of best practices and how these interesting challenges work in your code.
No deployment or infrastructure is necessary, code running on localhost is sufficient.
   

Testing
Test data will be username, Firstname Lastname, password, and Mother’s Favorite Search Engine. 

Some possible values to test with might be:
gh0st,William L. Simon,,Searx
jet-setter,Frank Abignale,r0u7!nG,Bing
kvothe,Patrick Rothfuss,3##Heel7sa*9-zRwT,Duck Duck Go 
tpratchett,Terry Pratchett,Thats Sir Terry to you!,Google 
lmb,Lois McMaster Bujold,null,Yandex


Final Project
The final project deliverable should include 
- `the source code` of your application, 
- a `README` file targeting other engineers, 
- and a plain `text dump of your data store with test data loaded`, 
- and at least 2 users logged in. 
- This can be in a tar file, zip file, or as separate attachments.
  



```py
def lambda_handler(event, context):
    # test one regions example
    regions = ['us-east-1']

    # set the region
    region =

    # set the client
    client = boto3.client('ec2', region)

    response = client.enable_ebs_encryption_by_default()


    # result =

    print("Default EBS Encryption setup for region", region,": ", response['EbsEncryptionByDefault'])

    return {
        'statusCode': 200,
        'body': json.dumps(event)
    }
```

```py
# If I was to give you 2 ordered sets of any data type you like (array, set, list, etc),
# such as 2,4,6,8 and 3,5,7,9 can you combine them into 1 ordered set so the final set
# is ordered 2,3,4,5,6,7,8,9.

# Extended challenge: can you write the logic to integrate the two sets of ordered numbers
# manually, so that while they are being merged they are in order (eg whever you are in
# merging the sets, the final set remains in incremental order)

def sortThem():
  #  example:
  a = [2,4,6,8]
  b = [3,5,7,9]
  # a = a+b
  # a.sort()
  # print(a)

  c = []

  i = 0
  j = 0

  while i < len(a):
    if (a[i] < b[j]):
      # print("a is smaller")
      # c[i]=a[i] #c.add(a[i])
      c.append(a[i])
      i+=1
    else:
      c.append(b[j])
      j+=1
    print(i,j)

  c = c + a[i:] + b[j:]

  print(c)

sortThem()

```


