---
title: AWS - boto3 - boto3.resource('dynamodb') - DynamoDB
date: 2020-07-18 11:11:11 -0400
categories: [01AWS, boto3]
tags: [AWS]
toc: true
image:
---

- [Amazon DynamoDB `boto3.resource('dynamodb')`](#amazon-dynamodb-boto3resourcedynamodb)

---

# Amazon DynamoDB `boto3.resource('dynamodb')`

By following this guide, you will learn how to use the `DynamoDB.ServiceResource` and `DynamoDB.Table` resources in order to create tables, write items to tables, modify existing items, retrieve items, and query/filter the items in the table.

```py
dynamodb = boto3.resource('dynamodb')
table = dynamodb.create_table(
    TableName='users',
    KeySchema=[
        {
            'AttributeName': 'username',
            'KeyType': 'HASH'
        },
        {
            'AttributeName': 'last_name',
            'KeyType': 'RANGE'
        }
    ],
    AttributeDefinitions=[
        {
            'AttributeName': 'username',
            'AttributeType': 'S'
        },
        {
            'AttributeName': 'last_name',
            'AttributeType': 'S'
        },
    ],
    ProvisionedThroughput={
        'ReadCapacityUnits': 5,
        'WriteCapacityUnits': 5
    }
)

table = dynamodb.Table('users')
table.item_count
table.creation_date_time

table.put_item(Item={'a':'x', 'b':'y', 'c':'z'})

table.update_item(
    Key={
        'username': 'janedoe',
        'last_name': 'Doe'
    },
    UpdateExpression='SET age = :val1',
    ExpressionAttributeValues={
        ':val1': 26
    }
)

response = table.get_item(Key={'a':'x', 'b':'y'})
item = response['Item']


with table.batch_writer() as batch:
    batch.put_item(Item={'a':'x', 'b':'y', 'c':'z'})
    batch.delete_item(Key={'partition_key': 'a','sort_key': 'b'})

with table.batch_writer(overwrite_by_pkeys=['partition_key', 'sort_key']) as batch:
    batch.put_item(Item={'a':'x', 'b':'y', 'c':'z'})
    batch.put_item(Item={'a':'x', 'b':'y', 'c':'z'})

from boto3.dynamodb.conditions import Key, Attr
response = table.query(KeyConditionExpression = Key('username').eq('johndoe'))
response = table.scan(FilterExpression = Attr('a').eq('x'))
response = table.scan(FilterExpression = Attr('a').eq('x') & Attr('a').begins_with('x'))
response = table.scan(FilterExpression = Attr('a.aa').eq('x'))

table.delete()
```


```py
# Get the service resource.
dynamodb = boto3.resource('dynamodb')

# -------------------------- Creating a new table --------------------------
# -------------------------- DynamoDB.ServiceResource.create_table()
# Create the DynamoDB table.
table = dynamodb.create_table(
    TableName='users',
    KeySchema=[
        {
            'AttributeName': 'username',
            'KeyType': 'HASH'
        },
        {
            'AttributeName': 'last_name',
            'KeyType': 'RANGE'
        }
    ],
    AttributeDefinitions=[
        {
            'AttributeName': 'username',
            'AttributeType': 'S'
        },
        {
            'AttributeName': 'last_name',
            'AttributeType': 'S'
        },
    ],
    ProvisionedThroughput={
        'ReadCapacityUnits': 5,
        'WriteCapacityUnits': 5
    }
)

# Wait until the table exists.
table.meta.client.get_waiter('table_exists').wait(TableName='users')

# Print out some data about the table.
print(table.item_count)

# Expected output:
# 0

# This creates a table named users that respectively has the hash and range primary keys username and last_name.
# This method will return a DynamoDB.Table resource to call additional methods on the created table.




# -------------------------- Using an existing table --------------------------
# create a DynamoDB.Table resource from an existing table:
# Instantiate a table resource object without actually creating a DynamoDB table.
# Note that the attributes of this table
# are lazy-loaded: a request is not made nor are the attribute
# values populated until the attributes on the table resource are accessed or its load() method is called.
table = dynamodb.Table('users')

# Print out some data about the table.
# This will cause a request to be made to DynamoDB and its attribute
# values will be set based on the response.
print(table.creation_date_time)

# Expected output
# 2015-06-26 12:42:45.149000-07:00






# -------------------------- Creating a new item
# -------------------------- DynamoDB.Table.put_item()
# Valid DynamoDB types: all of the valid types that can be used for an item
table.put_item(
   Item={
        'username': 'janedoe',
        'first_name': 'Jane',
        'last_name': 'Doe',
        'age': 25,
        'account_type': 'standard_user',
    }
)




# -------------------------- Getting an item
# -------------------------- DynamoDB.Table.get_item()
response = table.get_item(
    Key = {
        'username': 'janedoe',
        'last_name': 'Doe'
    }
)
item = response['Item']
print(item)
# Expected output:
# {u'username': u'janedoe',
#  u'first_name': u'Jane',
#  u'last_name': u'Doe',
#  u'account_type': u'standard_user',
#  u'age': Decimal('25')}




# -------------------------- Updating an item
# -------------------------- DynamoDB.Table.update_item()
table.update_item(
    Key={
        'username': 'janedoe',
        'last_name': 'Doe'
    },
    UpdateExpression='SET age = :val1',
    ExpressionAttributeValues={
        ':val1': 26
    }
)




# -------------------------- Deleting an item
# -------------------------- DynamoDB.Table.delete_item()
table.delete_item(
    Key={
        'username': 'janedoe',
        'last_name': 'Doe'
    }
)




# -------------------------- Batch writing
# loading a lot of data at a time, to both speed up the process and reduce the number of write requests made to the service.
# -------------------------- DynamoDB.Table.batch_writer()
# This method returns a handle to a batch_writer object that will automatically handle buffering and sending items in batches.
# In addition, the batch_writer will also automatically handle any unprocessed items and resend them as needed.
# All you need to do is call put_item for any items you want to add, and delete_item for any items you want to delete:
with table.batch_writer() as batch:
    batch.put_item(
        Item={
            'account_type': 'standard_user',
            'username': 'johndoe',
            'first_name': 'John',
            'last_name': 'Doe',
            'age': 25,
            'address': {
                'road': '1 Jefferson Street',
                'city': 'Los Angeles',
                'state': 'CA',
                'zipcode': 90001
            }
        }
    )
    batch.put_item(
        Item={
            'account_type': 'super_user',
            'username': 'janedoering',
            'first_name': 'Jane',
            'last_name': 'Doering',
            'age': 40,
            'address': {
                'road': '2 Washington Avenue',
                'city': 'Seattle',
                'state': 'WA',
                'zipcode': 98109
            }
        }
    )
# The batch writer is even able to handle a very large amount of writes to the table.
with table.batch_writer() as batch:
    for i in range(50):
        batch.put_item(
            Item={
                'account_type': 'anonymous',
                'username': 'user' + str(i),
                'first_name': 'unknown',
                'last_name': 'unknown'
            }
        )

# The batch writer can help to de-duplicate request by specifying overwrite_by_pkeys=['partition_key', 'sort_key']
# if you want to bypass no duplication limitation of single batch write request as botocore.exceptions.ClientError: An error occurred (ValidationException) when calling the BatchWriteItem operation: Provided list of item keys contains duplicates.
# It will drop request items in the buffer if their primary keys(composite) values are the same as newly added one, as eventually consistent with streams of individual put/delete operations on the same item.

with table.batch_writer(overwrite_by_pkeys=['partition_key', 'sort_key']) as batch:
    batch.put_item(
        Item={
            'partition_key': 'p1',
            'sort_key': 's2',
            'other': '111',
        }
    )
    batch.delete_item(
        Key={
            'partition_key': 'p1',
            'sort_key': 's2'
        }
    )
    batch.put_item(
        Item={
            'partition_key': 'p1',
            'sort_key': 's2',
            'other': '444',
        }
    )

# after de-duplicate:
batch.put_item(
    Item={
        'partition_key': 'p1',
        'sort_key': 's1',
        'other': '222',
    }
)
batch.put_item(
    Item={
        'partition_key': 'p1',
        'sort_key': 's1',
        'other': '444',
    }
)






# -------------------------- Querying and scanning the items in the table using
# -------------------------- DynamoDB.Table.query() or DynamoDB.Table.scan()
# To add conditions to scanning and querying the table
# import the boto3.dynamodb.conditions.Key and boto3.dynamodb.conditions.Attr] classes.
# boto3.dynamodb.conditions.Key:  used when the condition is related to the key of the item.
# boto3.dynamodb.conditions.Attr:  used when the condition is related to an attribute of the item:
from boto3.dynamodb.conditions import Key, Attr

# queries for all of the users whose username key equals johndoe:
response = table.query(KeyConditionExpression = Key('username').eq('johndoe'))
items = response['Items']
print(items)

# Expected output:
# [{u'username': u'johndoe',
#   u'first_name': u'John',
#   u'last_name': u'Doe',
#   u'account_type': u'standard_user',
#   u'age': Decimal('25'),
#   u'address': {u'city': u'Los Angeles',
#                u'state': u'CA',
#                u'zipcode': Decimal('90001'),
#                u'road': u'1 Jefferson Street'}}]


# scan the table based on attributes of the items.
# scans for all the users whose age is less than 27:
response = table.scan(FilterExpression = Attr('age').lt(27))
items = response['Items']
print(items)


# chain conditions together using the logical operators: & (and), | (or), and ~ (not).
# scans for all users whose first_name starts with J and account_type is super_user:
response = table.scan(FilterExpression = Attr('first_name').begins_with('J') & Attr('account_type').eq('super_user'))
items = response['Items']
print(items)



# scan based on conditions of a nested attribute.
# For example this scans for all users whose state in their address is CA:
response = table.scan(FilterExpression=Attr('address.state').eq('CA'))
items = response['Items']
print(items)



# -------------------------- Deleting a table
# -------------------------- DynamoDB.Table.delete()
table.delete()
```



---
