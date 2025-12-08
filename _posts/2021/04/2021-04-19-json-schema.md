---
title: "JSON Schema应用开发"
date: 2021-04-19
permalink: /2021-04-19-json-schema/
categories: ["实战分享"]
---

## json schema 作用


用一种通用规范，描述数据结构。


比如对于数据：


```json
{
    "name": "shijisun",
    "age": 24,
    "gender": "male"
}
```


那么按照 json schema 的规范，它的结构是：


```json
{
    "type": "object",
    "properties": {
        "name": {
            "type": "string",
            "minLength": 4
        },
        "age": {
            "type": "integer",
            "minimum": 0,
            "maximum": 130
        },
        "gender": {
            "type": "string",
            "enum": ["male", "female"]
        }
    }
}
```


type 字段代表类型，由于 type 是 object，所以有 properties 字段。properties 字段代表描述对象字段的数据类型。


对于 type = string 的字段，可以定义最大/最小长度（maxLength/minLength）；也可以设置枚举值（enum）。


除此之外，json schema 还有更多细节，可以看**参考链接**的第 1 和第 2 部分。


## 生成 json schema


首先明确的是，直接手写 json schema 的成本非常高，而且也不便于维护。


在 ts 中，typescript-json-schema.js 库提供了将 ts 的类型声明，转换为 json schema 的功能。同时，通过**类型注释**，来支持更多的 json schema 规范。


例如，声明一个 ts 文件：


```typescript
export enum PersonSex {
    male = 0,
    female = 1,
}

export interface Person {
    name: string;
    sex: PersonSex;

    /**
     * @items {"type":"string", "format":"email"}
     */
    emails?: string[];
}

export interface Location {
    /**
     * @default "china"
     */
    country: string;
    /**
     *
     */
    city: string;
}

export interface Student extends Person {
    school: string;
    location: Location;
}
```


下面是解析 ts 类型声明，并且将其转化为 json schema 输出的代码：


```typescript
/*
 * @Author: dongyuanxin
 * @Date: 2021-03-09 10:51:31
 * @Github: https://github.com/dongyuanxin/blog
 * @Blog: https://0x98k.com/
 * @Description: json schema
 */
const { resolve } = require("path");
const TJS = require("typescript-json-schema");

const settings = {
    required: true,
};

const compilerOptions = {
    strictNullChecks: true,
};

const program = TJS.getProgramFromFiles(
    [resolve("./3.ts")],
    compilerOptions,
    "./my-dir"
);

const generator = TJS.buildGenerator(program, settings);

const symbols = generator.getUserSymbols();
console.log(JSON.stringify(generator.getSchemaForSymbol("Student"), null, 4));

```


输出的json schema是：


```json
{
    "type": "object",
    "properties": {
        "school": {
            "type": "string"
        },
        "location": {
            "$ref": "#/definitions/Location"
        },
        "name": {
            "type": "string"
        },
        "sex": {
            "$ref": "#/definitions/PersonSex"
        },
        "emails": {
            "items": {
                "type": "string"
            },
            "type": "array"
        }
    },
    "required": [
        "location",
        "name",
        "school",
        "sex"
    ],
    "definitions": {
        "Location": {
            "type": "object",
            "properties": {
                "country": {
                    "default": "china",
                    "type": "string"
                },
                "city": {
                    "type": "string"
                }
            },
            "required": [
                "city",
                "country"
            ]
        },
        "PersonSex": {
            "enum": [
                0,
                1
            ],
            "type": "number"
        }
    },
    "$schema": "http://json-schema.org/draft-07/schema#"
}
```


## 应用场景


在很多数据驱动的平台中，经常使用 json schema 来定义数据模型。例如国内的云开发低码（腾讯云 WeDa）、API 编排系统等等。


通常，前端提供可视化拖拽界面，用户通过拖拽或者表单填写，生成一份平台可以理解的 json配置。后台接收到配置，然后用默认的json-schma验证成功后，再落库存储。


注：验证可以使用 [jsonschema.js](https://www.npmjs.com/package/jsonschema) 提供的 API。


## 参考链接

- [json schema docs](https://json-schema.org/learn/getting-started-step-by-step#intro): json schema 官方文档
- [json schema 参考书](https://imweb.io/topic/57b5f69373ac222929653f23): json schema 规范参考书
- [jsonschema.js](https://www.npmjs.com/package/jsonschema): 用于 json schema 格式验证
- [typescript-json-schema](https://github.com/YousefED/typescript-json-schema): 将 typescript 转换为 json schema
- [typescript json schema demos](https://github.com/YousefED/typescript-json-schema/blob/master/api.md): typescript 转为 json schema 的写法规范

