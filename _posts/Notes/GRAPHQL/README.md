-----------------

### GRAPHQL tricks

-----------------

### Using Curl

------------------

- Syntax-:

```bash
curl https://localhost:5000/api/v1/graphql -H "Content-Type: application/json" -d @introspection.json | jq ".data" | tee schema.json
```

------------------

### Graphql Dorks

-----------------

- FOFA-:

```query
body="GraphQL Server" && title="Graphql"
```

--------------------

### Common list so far

-------------------

- Urls-:

```text
/query
/graphql
```

-----------------

### Creating Fragments

-------------------

- Fragment-:

```grapqhl
fragment UserFields on User {
  id
  username
  email
}
```

- Calling it-:

```graphql
query GetUsers {
  activeUsers {
    ...UserFields
    status
  }
  adminUsers {
    ...UserFields
    permissions
  }
```
