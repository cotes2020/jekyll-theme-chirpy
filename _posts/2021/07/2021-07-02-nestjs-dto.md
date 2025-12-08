---
title: "NestJS源码: DTO实现"
date: 2021-07-02
permalink: /2021-07-02-nestjs-dto/
categories: ["B源码精读", NestJS]
---
## 概述


介绍 class-validator 和 class-transformer 实现 DTO，以及一些复杂功能的实现，部分实现参考了 NestJS 的管道源码。


项目参考：[github.com/dongyuanxin/cloudpress](https://github.com/dongyuanxin/cloudpress)


## NestJS 中的 DTO


NestJS 文档中提到了 2 种定义 DTO 的方法：

- 方法 1: 通过 joi 库创建 schema
- 方法 2: 通过 class-validator 和 class-transformer

**方法 1 就是按照 joi 库提供的规范，编写数据声明**，例如规定对象的 username 字段，长度范围是`[3, 30]`，且 username 字段不能为空，写法如下：


```typescript
const schema = Joi.object({
    username: Joi.string()
        .min(3)
        .max(30)
        .required()
)}
```


这种方法简单易懂，功能支持多（比如校验邮箱、网址），能在 js 中快速使用和验证，下载量最大。**不足之处，在 nestjs 中，编写 schema 的同时，还得编写 typescript 的类型声明。**例如一个接口，要求 body 中的 username 符合上述要求。那么除了编写 schema，为了更好的配合 ts 的使用，这里还得声明此接口的需要的 body 的类型，如下所示：


```typescript
export interface DemoApiBodyDto {
    public readonly username: string;
}
```


**方法 2 使用了 class-validator 和 class-transformer 这两个库**，通过 ES6 语法、元编程和装饰器，实现了只需定义 schema，无须特别声明 typescript。例如：


```typescript
import { MaxLength, MinLength, IsString } from "class-validator";

export class DemoApiBodyDto {
    @MaxLength(10)
    @MinLength(1)
    @IsString()
    public readonly username: string;
}
```


在 nestjs controller 中使用的时候：


```typescript
import { Controller, Body, Post, ValidationPipe } from "@nestjs/common";

@Controller("demo")
export class DemoController {
    constructor(private readonly demoService: DemoService) {}

    @Post()
    async demoApi(
        @Body(new ValidationPipe({ whitelist: true, transform: true }))
        body: DemoApiBodyDto
    ) {
        // 这里访问body，ts类型声明会生效，body上只有username字段
        // ...
    }
}
```


这种方法写法上更简单，通过装饰器来声明字段的属性，不需要编写冗长的 schema 以及配套的 ts 类型文件。缺点就是某些场景下使用成本高，文档和生态没有 joi 完善。


## 如何验证嵌套对象？


例如接口要求的入参数据格式是：


```typescript
{
    address_detail: {
        address_info: {
            user_name: "这个字段不能为空";
        }
    }
}
```


那么说明入参中，address_detail 是非空对象，address_info 是非空对象，并且 user_name 需要为字符串。在 class-validator 中，写法如下：


```typescript
import { IsString, ValidateNested, IsNotEmptyObject } from "class-validator";

class AddressInfoDto {
    @IsString()
    public readonly user_name: string;
}

class AddressDetailDto {
    @IsNotEmptyObject()
    @ValidateNested()
    @Type(() => AddressInfoDto)
    public readonly address_info: AddressInfoDto;
}

export class AddProductAddressDto {
    @IsNotEmptyObject() // 此字段不是空对象
    @ValidateNested() // 此字段需要递归验证嵌套结构
    @Type(() => AddressDetailDto) // 此字段对应的嵌套结构的DTO
    public readonly address_detail: AddressDetailDto;
}
```


## 如何在 Nestjs Provider 方法中使用 class-validator ？


Nestjs 提供的是在 Controller 上使用 DTO，底层实现是借助 `reflect-metadata` 实现的。


但是，在对应的 Provider 上想使用 DTO，或者更通俗的说，在普通方法上，使 class-validator 定义的 DTO 生效，应该怎么做？


通过翻看 nestjs 的 `ValidationPipe` 管道的源码发现，它的实现逻辑是：

1. 将数据转换为 DTO 实例对象
2. 验证转化后的对象是否符合 DTO 要求
3. 有错误，抛出 Nestjs Standard Http Exception；无错误，返回转化后的对象

这里将其抽成一个单元函数，可以被任何函数调用：


```typescript
import { plainToClass } from "class-transformer";
import { validate } from "class-validator";

export async function validateDto(Clazz, data) {
    if (!Clazz) {
        return data;
    }
    // excludeExtraneousValues = true 时，无关属性会被过滤。这里用默认值false
    const obj = plainToClass(Clazz as any, data, {
        excludeExtraneousValues: false,
    });
    const errors = await validate(obj);
    if (errors.length > 0) {
        // inspectLimit 是基于 util.inspect 二次封装的打印函数，请忽视
        const errMsg = inspectLimit(
            errors.map((err) => {
                return (
                    err.constraints || {
                        [err.property]: `Please check ${err.property} format`,
                    }
                );
            })
        );
        throw new Error(errMsg);
    }
    return obj;
}

```


在 Provider 的函数中，使用此函数，就能使得 class-validato 定义的 DTO 生效：


```typescript
@Injectable()
export class DemoService {
    constructor(
        private readonly env: EnvService,
        private readonly httpAgent: HttpAgentService,
        private readonly logger: LoggerService,
        private readonly requestCtx: RequestCtxService,
    ) { }

    /**
     * 转发流量
     */
    public async getDemoInfo(params?: GetDemoInfoDto) {
        params = await validateDto(GetDemoInfoDto, params)
        // ...
    }
```


## 如何使用 class-transformer 进行字段转化，并填入默认值？


假设定义了一个分页的 DTO：

- page 和 page_size 都是 number 类型，并且不能为空
- 经过 `@IsOptional()` 声明的参数，前端不传不会报错；否则，不论是否有默认值，还是 `@Transform` 转换，都会报错
- 如果前端没传入，会读取默认值，不会触发 `@Transform`
- 如果前端有传入，跳过默认值，会触发绑定的  `@Transform`

这些可以借助 `@Transform` 来实现。代码实现如下：


```typescript
export class DtoTransformBuilder {
    public static defaultString(defaultValue: string) {
        return ({ value }) =>
            typeof value === "string" ? value : defaultValue;
    }

    public static defaultInt(defaultValue: number) {
        return ({ value }) => parseInt(value, 10) || defaultValue;
    }
}

export class PaginationDto {
		@IsOptional()
    @Transform(DtoTransformBuilder.defaultInt(1))
    @IsInt()
    public readonly page: number = 1;
		
		@IsOptional()
    @Transform(DtoTransformBuilder.defaultInt(10))
    @IsInt()
    public readonly page_size: number = 10;

		@IsOptional()
	  @IsIn(['v1', 'v3'])
	  public readonly signVersion: 'v1' | 'v3' = 'v3';
}

export class SearchListDto extends PaginationDto {
    @IsString()
    public readonly kewword: string;
}

```


## 如何复用 DTO 定义？


借助 ES6 语法，可以通过“继承”实现复用：


```typescript
export class SearchListDto extends PaginationDto {
    @IsString()
    public readonly kewword: string;
}
```


或者通过`@Type`，以“组合”的方式实现复用：


```typescript
class AddressInfoDto {
    @IsString()
    public readonly user_name: string;
}

class AddressDetailDto {
    @IsNotEmptyObject()
    @ValidateNested()
    @Type(() => AddressInfoDto)
    public readonly address_info: AddressInfoDto;

    @IsNotEmptyObject()
    @ValidateNested()
    @Type(() => PaginationDto)
    public readonly pagination: PaginationDto;
}
```


## 参考链接

1. NestJS 源码: [https://github.com/nestjs/nest/blob/master/packages/common/pipes/validation.pipe.ts](https://github.com/nestjs/nest/blob/master/packages/common/pipes/validation.pipe.ts)
2. class-validator 替代 Joi: [https://cnodejs.org/topic/5c2a0ad376c4964062a1f60f](https://cnodejs.org/topic/5c2a0ad376c4964062a1f60f)
3. 验证嵌套对象：[https://dev.to/avantar/validating-nested-objects-with-class-validator-in-nestjs-1gn8](https://dev.to/avantar/validating-nested-objects-with-class-validator-in-nestjs-1gn8)

