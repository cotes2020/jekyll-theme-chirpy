---
title: AI - JSON Schema
date: 2021-08-11 11:11:11 -0400
description:
categories: [51AI]
# img: /assets/img/sample/rabbit.png
tags: [AI, ML]
---

- [JSON Schema](#json-schema)
  - [overall](#overall)
  - [validate JSON data using Python](#validate-json-data-using-python)
  - [JSON Schema](#json-schema-1)
  - [Why developers use JSON Schema](#why-developers-use-json-schema)
  - [Create schema](#create-schema)
    - [Creating a schema definition](#creating-a-schema-definition)
    - [Defining properties](#defining-properties)
    - [Nesting data structures](#nesting-data-structures)
    - [Adding outside references](#adding-outside-references)
    - [Validating JSON data against the schema](#validating-json-data-against-the-schema)


---

# JSON Schema

---

## overall

JSON Schema is a declarative language that you can use to annotate and validate the structure, constraints, and data types of the JSON documents. It provides a way to standardize and define expectations for the JSON data.


![Screenshot 2024-05-13 at 15.07.47](/assets/img/Screenshot%202024-05-13%20at%2015.07.47.png)


Using JSON Schema, you can define rules and constraints that JSON data should adhere to.

- When the JSON documents adhere to these constraints, it becomes easier to exchange structured data between applications because the data follows a consistent pattern.


## validate JSON data using Python

```py
# Without JSON Schema
data = {
"product": {
    "name": "Widget",
    "price": 10.99,
    "quantity": 5
    }
}
# Performing basic validation
if "product" in data and isinstance(data["product"], dict) and "name" in data["product"] and "price" in data["product"]:
    print("Valid JSON object.")
else:
    print("Invalid JSON object.")
```

The basic validation logic checks whether the required fields exist in the JSON object.
- However, as the structure becomes more complex, the validation code becomes more complicated and prone to errors.
- Moreover, this approach lacks support for checking data types, handling nested structures, and enforcing specific constraints.

---

## JSON Schema


JSON document:

- represents a piece of data that follows the syntax and structure defined by the JSON format. It is a collection of key-value pairs, arrays, and nested objects.

- JSON documents are used to store and transfer data between systems and applications.

- a specification language for JSON that allows you to describe the structure, content, and semantics of a JSON instance.
- With JSON Schema, you can define metadata about an object's properties, specify whether fields are optional or required, and define expected data formats.
- By using JSON Schema, people can better understand the structure and constraints of the JSON data they are using. It enables applications to validate data, ensuring it meets the defined criteria.
- With JSON Schema, you can make the JSON more readable, enforce data validation, and improve interoperability across different programming languages.

An example of a JSON document representing a customer order:

```json
{
  "order_id": "123456",
  "customer_name": "John Doe",
  "items": [
    {
      "product_id": "P001",
      "name": "T-shirt",
      "quantity": 2,
      "price": 19.99
    },
    {
      "product_id": "P002",
      "name": "Jeans",
      "quantity": 1,
      "price": 49.99
    }
  ],
  "total_amount": 89.97,
  "status": "pending"
}
```


Using the same example, we can validate the data by making use of the jsonschema Python library:


```py
# Without JSON Schema
data = {
"product": {
    "name": "Widget",
    "price": 10.99,
    "quantity": 5
    }
}

schema = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "Product",
    "type": "object",
    "properties": {
        "product": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string"
                },
                "price": {
                    "type": "number",
                    "minimum": 0
                },
                "quantity": {
                    "type": "integer",
                    "minimum": 1
                }
            },
            "required": ["name", "price", "quantity"]
        }
    },
    "required": ["product"]
}
try:
    validate(data, schema)
    print("Valid JSON object.")
except Exception as e:
    print("Invalid JSON object:", e)
```

By using JSON Schema, we can easily define and enforce constraints, making the validation process more robust and manageable. It improves the readability of the code and reduces the chances of data-related issues.

---

## Why developers use JSON Schema

With JSON Schema:
- **Describe existing data formats**: JSON Schema allows you to `describe the structure, constraints, and data types` of the existing JSON data formats.
- **Define rules and constraints**: When the JSON documents adhere to these constraints, it becomes easier to exchange structured data between applications because the data follows a consistent pattern.
- **Clear and readable documentation**: JSON Schema supports the creation of documentation that is easily understandable by both humans and machines.
- **Highly extensible** and can be tailored to fit the needs.
  - create custom keywords, formats, and validation rules to suit the own requirements.
- **Validate the data**, which helps:
  - **Automate testing**: JSON Schema validation enables automated testing, ensuring that data consistently adheres to the specified rules and constraints.
  - **Enhance data quality**: By enforcing validation rules, JSON Schema helps ensure the quality of client-submitted data, preventing inconsistencies, errors, and malicious inputs.
- **Wide range of tools availability**: The JSON Schema community has a wealth of tools and resources available across many programming languages to help you create, validate, and integrate the schemas.

---

## Create schema

---

### Creating a schema definition

```json
{
  "productId": 1,
  "productName": "A green door",
  "price": 12.50,
  "tags": [ "home", "green" ]
}

// schema definition
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "https://example.com/product.schema.json",
  "title": "Product",
  "description": "A product in the catalog",
  "type": "object"
}

```






---

### Defining properties

```json
// schema definition
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "https://example.com/product.schema.json",
  "title": "Product",
  "description": "A product from Acme's catalog",
  "type": "object",
  "properties": {
    "productId": {
      "description": "The unique identifier for a product",
      "type": "integer"
    },
    "productName": {
      "description": "Name of the product",
      "type": "string"
    },
    "price": {
      "description": "The price of the product",
      "type": "number",
      "exclusiveMinimum": 0
    },
    "tags": {
      "description": "Tags for the product",
      "type": "array",
      "items": {
        "type": "string"
      },
      "minItems": 1,
      "uniqueItems": true
    }
  },
  "required": [ "productId", "productName", "price" ]
}
```

---

### Nesting data structures

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "https://example.com/product.schema.json",
  "title": "Product",
  "description": "A product from Acme's catalog",
  "type": "object",
  "properties": {
    "productId": {
      "description": "The unique identifier for a product",
      "type": "integer"
    },
    "productName": {
      "description": "Name of the product",
      "type": "string"
    },
    "price": {
      "description": "The price of the product",
      "type": "number",
      "exclusiveMinimum": 0
    },
    "tags": {
      "description": "Tags for the product",
      "type": "array",
      "items": {
        "type": "string"
      },
      "minItems": 1,
      "uniqueItems": true
    },
    "dimensions": {
      "type": "object",
      "properties": {
        "length": {
          "type": "number"
        },
        "width": {
          "type": "number"
        },
        "height": {
          "type": "number"
        }
      },
      "required": [ "length", "width", "height" ]
    }
  },
  "required": [ "productId", "productName", "price" ]
}
```


---

### Adding outside references

```json
{
  "$id": "https://example.com/geographical-location.schema.json",
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "title": "Longitude and Latitude",
  "description": "A geographical coordinate on a planet (most commonly Earth).",
  "required": [ "latitude", "longitude" ],
  "type": "object",
  "properties": {
    "latitude": {
      "type": "number",
      "minimum": -90,
      "maximum": 90
    },
    "longitude": {
      "type": "number",
      "minimum": -180,
      "maximum": 180
    }
  }
}
```


```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "https://example.com/product.schema.json",
  "title": "Product",
  "description": "A product from Acme's catalog",
  "type": "object",
  "properties": {
    "productId": {},
    "productName": {},
    "price": {},
    "tags": {},
    "dimensions": {},
    },
    "warehouseLocation": {
      "description": "Coordinates of the warehouse where the product is located.",
      "$ref": "https://example.com/geographical-location.schema.json"
    }
  },
  "required": [ "productId", "productName", "price" ]
}
```

---

### Validating JSON data against the schema

```json
{
  "productId": 1,
  "productName": "An ice sculpture",
  "price": 12.50,
  "tags": [ "cold", "ice" ],
  "dimensions": {
    "length": 7.0,
    "width": 12.0,
    "height": 9.5
  },
  "warehouseLocation": {
    "latitude": -78.75,
    "longitude": 20.4
  }
}
```

- To validate this JSON data against the product catalog JSON Schema, you can use any validator. In addition to command-line and browser tools, validation tools are available in a wide range of languages, including Java, Python, .NET, and many others. [find a validator that’s right for your project](https://json-schema.org/implementations)
- Use the example JSON data as the input data and the product catalog JSON Schema as the schema. Your validation tool compares the data against the schema, and if the data meets all the requirements defined in the schema, validation is successful.

---

























.
