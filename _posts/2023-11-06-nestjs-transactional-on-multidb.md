---
title: NestJS 특수한 상황에서 “@Transactional()” 직접 만들어먹기
author: chungjung # \_data/authors.yml 에 있는 author id (여러명인경우 authors: [id1, id2, ...])
date: 2023-11-06 10:33:00 +0900 # +0900은 한국의 타임존  (날짜가 미래인경우 빌드시 스킵함.)
categories: [Backend, nestjs] # 카테고리는 메인, 서브 2개만 설정 가능 (띄어쓰기 가능)
tags: [nestjs, backend, transaction] # 태그는 개수제한 X (띄어쓰기 가능)
---

Spring은 국내에서 매우 대중적인 프레임워크인 만큼, 백엔드를 잠시나마 배웠던 사람이라면 Spring에서 쓰이는 `@Transactional()` 어노테이션에 대해서 꼭 한번쯤은 보게 되지만 NestJS에서는 공식적으로 사용할 수는 없었습니다.

최근에 NestJS에서 `@Transactional()`을 사용할 수 있게 하는 관련된 자료와 라이브러리가 많이 나왔지만, 제 프로젝트 환경에서 사용하기엔 환경이 맞지 않았습니다. 그래서 이번 글에서는 제가 직접 Spring의 `@Transactional()` 을 만들어야 했던 이유와 구현 과정, 최종적으로 구현했던 결과물에 대해서 보여드리려고 합니다.

⚠️ **글 내에 첨부된 코드는 완전하지 않습니다. 필요한 부분만 불러서 가져왔습니다.**

⚠️ **사이드 프로젝트에서 발췌한 코드들이 대부분이라, 추후 기회가 된다면 Gtihub으로 따로 분리해서 올리겠습니다.**

## 그래서 직접 Transactional 데코레이터를 만들 필요가 있을까?

실제로 TypeORM환경에서 nestjs에서 `@Transactional()`을 사용할 수 있게 해주는 라이브러리가 존재 합니다. 그리고 조금만 찾아보면 이를 직접 구현하는 방법들도 나와 있습니다.

그러나 **제 프로젝트는 Neo4J라는 그래프 데이터베이스와, MongoDB라는 NoSQL 기반 데이터베이스를 사용합니다.** 우선적으로 Neo4j는 현재 기준 TypeORM에서 지원하지 않으며, MongoDB는 TypeORM에서 지원하지만, neo4j로 인해서 TypeORM을 고집할 이유가 없는 프로젝트에서 사용할 이유가 없다고 생각하여 공식문서에서 사용하는 @nestjs/mongoose를 사용하기로 정했습니다.

따라서 이러한 상황을 해결 할 수 있는 라이브러리가 없었기 때문에 직접 만들어야 할 필요성이 있었습니다.

## @Transactoinal() 사용했을 때의 이점

우선 `@Transactional()` 데코레이터를 만들기 전에 앞서서, 이 데코레이터를 사용하기 전 후를 코드로 보여드리겠습니다.

### @Transactoinal() 미사용

```tsx
 async create (createDto: CreateDto): Promise<void> {
    const neo4jTransaction = this.neo4jService
      .getWriteSession()
      .beginTransaction();
    const mongoSession = await this.mongoConnection.startSession();
    const mongoTransaction = mongoSession.startTransaction();

    try {

	  //neo4j database repository
      await this.neo4jRepository.createBarNode(neo4jModel, neo4jTransaction);

	  //mongoDB database repository
      await this.mongoRepository.create(mongoModel);

      await neo4jTransaction.commit();
      await mongoSession.commitTransaction();
    } catch (e) {
      await neo4jTransaction.rollback();
      await mongoSession.abortTransaction();
      throw e;
    } finally {
			await mongoSession.endSession();
		}
  }

```

### @Transactoinal() 사용

```tsx
@Transactional()
async create (createDto: CreateDto): Promise<void> {

    await this.neo4jRepository.createNode(neo4jModel);
    await this.mongoRepository.create(mongoModel);
}

```

위의 차이가 바로 `@Transactional()` 데코레이터를 사용하는 이유입니다. 이 데코레이터를 사용하게 되면, 데코레이터가 적용된 하위의 함수를 감싸서 함수 내부의 전체 로직이 같은 트랜잭션 내에서 동작하게 만들어줍니다.

즉 거추장스럽게 비즈니스 로직 내부에 트랜잭션을 시작하고, 종료하는 코드를 넣을 필요가 없습니다. 또한 Neo4J같은 경우 `this.neo4jRepository.create(neo4jModel,neo4jTransaction);` 같이 repository의 메서드에 transaction을 관리하는 객체를 인자로 넘겨야 하는데, 데코레이터 적용된 버전을 보면 그도 필요없는 것을 확인할 수 있습니다.

## @Transactoinal()의 기본 원리

위에서 `@Transactional()` 을 사용함으로써 매우 극적인 변화가 코드에 일어난 것을 확인했습니다. 그렇다면 과연 `@Transactional()` 은 어떻게 만드는 것일까요?

![Transaction basic principle.png](/assets/img/2023-11-06-nestjs-transactional-on-multidb/Transaction%20basic%20principle.png)

자세하게 들어가면 많은 복잡한 개념들이 있지만, 기본적인 개념자체는 엄청 간단합니다. Java Spring에서 각 Http Request요청 자체는 기본적으로 하나의 스레드에서 처리하도록 되어 있는데, 이에 따라서 각 스레드에는 스레드 고유의 정보를 저장하기 위한 Local Storage같은 공간인 ThreadLocal이 존재합니다. 따라서 이 공간에다가 Transaction을 관리해주는 Transaction Manager같은 역할을 하는 객체를 넣어두고 `@Transactional()` 이 달린 함수를 호출할 때마다 Thread Local에서 Transaction Manager를 주입해주는 방식으로 동작합니다.

그러나 이 개념을 NestJS에 그대로 도입하면 심각한 문제가 발생합니다. 바로 NestJS가 사용하는 Node 기반 런타임은 싱글스레드이기 때문입니다. 그렇기 때문에 Thread에 종속된 local variable이 존재한다면, 결과적으로 NestJS의 모든 요청이 어디서든 똑같은 local variable을 사용하게 됩니다.

이게 문제가 되는 이유는 Transaction은 요청별로 다른 상태값을 가져야 하지만, 모든 요청이 동일한 Transaction을 관리하는 객체를 주입받게 되면, 그 순간 온갖 request가 혼재되어 서비스가 동작할 수 없는 상태가 될 수 있습니다.

## Cls-hooked

이 문제를 해결하고자 나온 라이브러리가 바로 cls-hooked입니다. cls-hooked는 callback-chain단위로 격리된 저장 공간을 만들어줍니다. 즉 이 라이브러리를 이용한다면 적어도 한 Request당 같은 callback-chain상에서 동작하도록 되어있기 때문에 (message queue 같은 특이한 방법을 쓴다고 가정하지 않은 경우) request단위로 격리된 Transaction manager를 생성하고, 이를 같은 callback-chain내에 있는 함수 어디서든지 호출해서 사용할 수 있습니다.

![Transaction basic principle by using cls.png](/assets/img/2023-11-06-nestjs-transactional-on-multidb/Transaction%20basic%20principle%20by%20using%20cls.png)

## 추가적인 문제, Neo4J의 Transaction Manager은 수동적이다

자 위의 Cls-hooked를 이용하면 callback-chain단위의 스토리지를 만들 수 있다는 사실을 알았습니다. 여기서 바로 문제가 해결이 되었으면 좋겠지만, 한 가지 심각한 문제가 더 남아있습니다. Mongo DB의 경우 비교적 대중적인 툴이라 `mongoSession.startTransaction()` 을 이용하면 transactoin이 read인지 write인지 지동으로 판단해서 적용이 되지만 Neo4J는 그렇지 않습니다.

현재 사용중인 버전 기준으로 Neo4J에서 Transaction을 시작하는 방법은 두 가지 입니다.

```json
{
  "neo4j-driver": "^5.10.0",
  "nest-neo4j": "^0.3.1"
}
```

```tsx
const neo4jTransaction = this.neo4jService.getWriteSession().beginTransaction();

const neo4jTransaction = this.neo4jService.getReadSession().beginTransaction();
```

코드를 확인하면 아시겠지만, **Neo4J의 경우 트랜잭션이 Read인지 Write인지 수동으로 명시를 하고 transaction을 시작**해야 합니다.

## 한가지 더, 어떤 DB를 사용하는지 알 수 있는 방법이 없다.

거기에 한 가지 문제가 더 있습니다. 요청이 들어오는 컨트롤러에 할당된 비즈니스 로직에 MongoDB를 사용하고 있는지, Neo4J를 사용하고 있는지를 명시적으로 알 수 있는 방법이 없습니다. 만약 이런상황이 지속된다면 우선 **transaction manager를 mongoDB, neo4j에 대한 것 두 개를 다 만들어줘야** 합니다.

## Inject Metadata can solve this problem

위에서 설명한 두 가지 문제 모두 결국 transaction을 만드는 시점에서 어떤 유형의 transactin을 시작해야 하는지 모르기 때문에 발생한 문제입니다. **그렇다면 미리 transactoin이 만들어지기 전에, Request가 맨 처음 시작 될때, 이 정보를 메타데이터로 주입**해놓으면 어떨까요?

위의 다이어그램이 방금의 아이디어를 고려해서 설계한 @Transactional()입니다.

![Transactional using cls.png](/assets/img/2023-11-06-nestjs-transactional-on-multidb/Transactional%20using%20cls.png)

우선 코드에서 SetMetaData를 통해서 모든 컨트롤러의 메서드에 MongoDB와 Neo4J의 트랜잭션 정보를 기록합니다. 이렇게 하면 다음과 같은 순서를 따라서 Transaction이 진행되게 됩니다.

1. Interceptor에서 metadata정보를 읽어들여서 정보에 맞는 트랜잭션 매니저를 생성하고 cls-namespace에 저장합니다.

   예를 들어서 `{ mongo : true , neo4j : READ}` 라는 정보가 메타데이터로 저장되어 있으면, mongo의 session을 생성하고, neo4j read transaction을 만들어서 mongo session과 neo4j read transaction을 시작하고 namespace에 저장합니다.

2. `@Transactional()` 데코레이터가 걸린 메서드를 호출하게 되면, 데코레이터에서 namespace에 저장되어 있는 트랜잭션 매니저를 불러옵니다.
3. 트랜잭션을 시작하고 원래 로직을 시작합니다.
4. repository에서 namseapce에 있는 트랜잭션 메니저를 가져옵니다.
5. 트랜잭션 매니저를 이용해서 DB 입출력에 관한 작업을 수행합니다.
6. 작업이 성공적으로 완료되면 commit을 실패하면 rollback을 합니다.

## Now implement @transactional()

이제 코드로 구현해보겠습니다.

### Set Metadata && Create Transaction manager

```tsx
export function InjectMongoAndWriteNeo4j(): MethodDecorator {
  return (target, key, descriptor) => {
    const mongoMeataData: MongoMeataData = {
      useMongo: true
    };

    const neo4jMetaData = {
      useNeo4j: true,
      txType: "WRITE"
    };

    SetMetadata(MONGO_METADATA, mongoMeataData)(target, key, descriptor);
    SetMetadata(NEO4J_METADATA, neo4jMetaData)(target, key, descriptor);
    UseInterceptors(TransactionalInterceptor)(target, key, descriptor);

    return descriptor;
  };
}
```

```tsx
export const TRANSACTION_NAMESPACE = "TRANSACTION_NAMESPACE";
export const TRANSACTION_MONGO = "TRANSACTION_MONGO";
export const TRANSACTION_NEO4J = "TRANSACTION_NEO4J";

export const MONGO_METADATA = "mongoMetaData";
export const NEO4J_METADATA = "neo4jMetaData";

export type MongoMeataData = {
  useMongo: boolean;
};

export type Neo4jMetaData = {
  useNeo4j: boolean;
  txType?: "READ" | "WRITE";
};

@Injectable()
export class TransactionalInterceptor implements NestInterceptor {
  constructor(
    private readonly neo4jService: Neo4jService,
    @InjectConnection() private readonly mongoConnection: mongoose.Connection
  ) {}

  intercept(context: ExecutionContext, next: CallHandler): Observable<any> {
    const nameSpace =
      getNamespace(TRANSACTION_NAMESPACE) ??
      createNamespace(TRANSACTION_NAMESPACE);

    const mongoMetaData: MongoMeataData = Reflect.getMetadata(
      "mongoMetaData",
      context.getHandler()
    );
    const neo4jMetaData: Neo4jMetaData = Reflect.getMetadata(
      "neo4jMetaData",
      context.getHandler()
    );

    return from(
      nameSpace.runAndReturn(async () => {
        await this.setMongoTransaction(mongoMetaData);
        await this.setNeo4jTransaction(neo4jMetaData);
        return await lastValueFrom(next.handle());
      })
    );
  }

  private async setMongoTransaction(
    mongoMeataData: MongoMeataData | undefined
  ) {
    const namespace = getNamespace(TRANSACTION_NAMESPACE);

    if (mongoMeataData === undefined || mongoMeataData.useMongo === false) {
      namespace.set("TRANSACTION_MONGO", null);
      return;
    }

    const mongoSession = await this.mongoConnection.startSession();
    namespace.set("TRANSACTION_MONGO", mongoSession);
  }

  private async setNeo4jTransaction(neo4jMetaData: Neo4jMetaData | undefined) {
    const namespace = getNamespace(TRANSACTION_NAMESPACE);

    if (neo4jMetaData === undefined || neo4jMetaData.useNeo4j === false) {
      namespace.set("TRANSACTION_NEO4J", null);
      return;
    }

    if (neo4jMetaData.txType === "READ") {
      const neo4jTransaction = this.neo4jService
        .getReadSession()
        .beginTransaction();
      namespace.set("TRANSACTION_NEO4J", neo4jTransaction);
      return;
    }

    const neo4jTransaction = this.neo4jService
      .getWriteSession()
      .beginTransaction();
    namespace.set("TRANSACTION_NEO4J", neo4jTransaction);
  }
}
```

위의 두 코드 블럭이 1번을 담당하는 코드 블럭입니다.

### 첫 번째 코드 블록

메타데이터를 적용하고, 그 메타데이터가 적용된 상태로 TransactionalInterceptor를 호출합니다.

### 두 번째 코드 블록

호출 된 TransactionalInterceptor인데 이 인터셉터는 namespace를 생성 혹은 가져와서 `nameSpace.runAndReturn` 을 통해서 격리를 시작합니다. 그리고 메타데이터를 읽어들여서 그에 따른 트랜잭션 매니저를 생성하고 cls-hooked의 namespace에 저장합니다. 예를 들어서 `mongoMetaData` 가 ture이면 mongosession을 만들어서 cls-hooked의 `TRANSACTION_MONGO`라는 키로 저장합니다.

참고로 `nameSpace.runAndReturn` 이 적용된 순간부터 callback-chain단위로 격리되어 적용되게 됩니다.

아래 다이어그램을 보시면 조금 더 이해하기 쉽습니다.

![Inject metadata workflow.png](/assets/img/2023-11-06-nestjs-transactional-on-multidb/inject%20metadata%20workflow.png)

실제 컨트롤러에는 다음과 같이 적용됩니다.

```tsx
@Post()
@InjectMongoAndWriteNeo4j()
async create(@Body() createDto: CreateDto) {
  await this.service.create(createDto);
  return { message: 'created successfully' };
}
```

### @Transactional()

실제로 트랜잭션을 제어하는 코드입니다. 다이어그램을 보여주며 설계했던 부분에서 2,3,6 부분을 담당합니다.

```tsx
export function Transactional() {
  return function (
    _target: any,
    _propertyKey: string | symbol,
    descriptor: TypedPropertyDescriptor<any>
  ) {
    const originalMethod = descriptor.value;

    async function transactionWrapped(...args: unknown[]) {
      const nameSpace = getNamespace(TRANSACTION_NAMESPACE);

      const mongoSession: mongoose.ClientSession =
        nameSpace.get(TRANSACTION_MONGO);

      // Get Neo4j Transaction
      const neo4jTransaction: TransactionPromise =
        nameSpace.get(TRANSACTION_NEO4J);

      try {
        if (mongoSession) {
          mongoSession.startTransaction();
        }

        const result = await originalMethod.apply(this, args);

        if (neo4jTransaction) {
          neo4jTransaction.commit();
        }

        if (mongoSession) {
          mongoSession.commitTransaction();
        }

        return result;
      } catch (error) {
        if (neo4jTransaction) {
          neo4jTransaction.rollback();
        }

        if (mongoSession) {
          mongoSession.abortTransaction();
        }

        throw error;
      } finally {
        if (mongoSession) {
          mongoSession.endSession();
        }
      }
    }

    descriptor.value = transactionWrapped;
  };
}
```

![Transactional decorator workflow.png](/assets/img/2023-11-06-nestjs-transactional-on-multidb/Transactional%20decorator%20workflow.png)

예외 처리 관련된 코드를 제거하고 핵심 기능만 넣었습니다. 이해가 안된다면 다이어그램을 대신 봐도 됩니다.

1.  namespace에서 mongo session과 neo4j transaction을 불러와서 각각 트랜잭션 매니저가 있는지 확인합니다.
2.  트랜잭션 메니저가 있는 경우에만 해당 데이터베이스에 대한 트랜잭션을 시작합니다
3.  원래 코드를 실행시킵니다
4.  트랜잭션이 성공하면 commit, 실패하면 rollback합니다.

실제 코드는 다음과 같이 사용합니다.

```tsx
@Transactional()
async create (createDto: CreateDto): Promise<void> {

    await this.neo4jRepository.createNode(neo4jModel);
    await this.mongoRepository.create(mongoModel);
}
```

### Repository에서 transaction 정보 불러오기

mongoDB는 트랜잭션 정보를 불러오지 않아도 내부적으로 mongo session으로 트랜잭션을 시작하면 자동으로 트랜잭션 정보를 불러오는 것 같습니다. 하지만 neo4j는 그렇지 못하고, 트랜잭션 객체를 직접 주입해줘야만 합니다.

다만 함수의 파라미터로 transaction 정보를 넘겨줄수는 없으니, namespace에 접근해서 transaction manger를 가져와야 합니다.

```tsx
export type TransactionManagerList = {
  neo4jTransaction: TransactionPromise;
};

@Injectable()
export class TransactionalHelper {
  getTransaction(): TransactionManagerList {
    const nameSpace = getNamespace(TRANSACTION_NAMESPACE);
    return { neo4jTransaction: nameSpace.get(TRANSACTION_NEO4J) };
  }
}
```

이렇게 TransactionalHelper클래스를 만들어서 namespace에 접근해서 transaction manager를 가져옵니다. 이 provider를 repository에 명시적으로 주입받아서 사용해도 되고, neo4j관련된 Repository에 abstract calss를 만들어서 자동으로 TransactionalHelper클래스를 주입받도록 설정 할 수도 있습니다.

저는 후자를 선택했습니다.

## Why metadata is in controller?

이렇게 만든 프로젝트에는 한 가지 치명적인 단점이 존재했으니, 바로 컨트롤러에서 메타데이터를 선언했다는 점 입니다.

Transaction이 시작되기 전에 어떤 트랜잭션의 유형이 필요한지 알아야 했기에 발생한 일이였지만, Controller의 역할에 맞지 않는 코드가 존재하는 것 자체가 코드 컨벤션적 관점에서 보기에 크게 좋은 일은 아닙니다. 이를 해결하기 위해서 저는 다음과 같은 생각을 하게 됩니다.

“**Repository에 각 메서드마다 필요한 트랜잭션 타입의 메타데이터를 설정하고 그 메타데이터를 동적으로 transaction manager를 만들기 전에 전부 읽어들여서 확인하고 그 정보를 바탕으로 transaction manager를 cls-hooked가 생성한 namspace에 주입해주 주면 되지 않을까?**”

현재는 다음과 같이 @Transactional() 을 사용하고 있습니다.

```tsx
//controller

@Post()
@InjectMongoAndWriteNeo4j()
async create(@Body() createDto: CreateDto) {
  await this.service.create(createDto);
  return { message: 'created successfully' };
}
```

```tsx
//service

@Transactional()
async create (createDto: CreateDto): Promise<void> {

    await this.neo4jRepository.createNode(neo4jModel);
    await this.mongoRepository.create(mongoModel);
}
```

```tsx
//repository

async createNode(neo4jModel: Neo4jModel): Promise<Neo4jModel> {
  //create logic
}
```

그러나 위에서 말한 생각이 적용된다면 아래와 같이 controller에 데코레이터가 사라지고 repository에 데코레이터가 생기게 됩니다.

```tsx
//controller

@Post()
async create(@Body() createDto: CreateDto) {
  await this.service.create(createDto);
  return { message: 'created successfully' };
}
```

```tsx
//service

@Transactional()
async create (createDto: CreateDto): Promise<void> {

    await this.neo4jRepository.createNode(neo4jModel);
    await this.mongoRepository.create(mongoModel);
}
```

```tsx
//repository

@Neo4jTransaction(Neo4jTransactionTypeEnum.WRITE)
async createNode(neo4jModel: Neo4jModel): Promise<Neo4jModel> {
  //create logic
}
```

이렇게 된다면 적어도 controller, service, repository가 각각의 역할에 맞는 코드가 될 수 있습니다.

## JavaScript Proxy

위와 같이 구현하기위해서 중요하게 알아야 하는 개념인 JS Proxy개념에 대해서 먼저 다루겠습니다.

![JS fundamental operation.png](/assets/img/2023-11-06-nestjs-transactional-on-multidb/JS%20fundamental%20operation.png)

JS 에서는 항상 다른 Object나 특정 property에 접근하기 위해서는 fundamental operation이 실행 됩니다. 예를 들어서 Class B에서 `this.a.method1()` 를 호출할떄는 A의 method1에 대해서 get operation이 실행됩니다.

![JS proxy.png](/assets/img/2023-11-06-nestjs-transactional-on-multidb/JS%20proxy.png)

JS에서 Proxy는 이러한 기본적인 fundamental operation을 가로채는데 사용됩니다. get동작을 가로채는 코드를 살펴보겠습니다.

```tsx
const target = {};
const proxy = new Proxy(target, {
  get(target, prop) {
    return `${prop} 의 값은 ${target[prop]}`;
  }
});

proxy.test = 5;
console.log(proxy.test); // test 의 값은 5
```

이와같이 proxy는 이러한 기초적인 정보를 변경할 수 있습니다.

이제 새롭게 디자인된 Transactional 데코레이터에 대해서 다시 한번 살펴보겠습니다.

## New Transactional decorator

Proxy가 Transactional의 구현에서 어떻게 사용되었는지 알기 전에 앞서서 먼저 어떻게 새로운 Transactaionl 데코레이터가 구현되는지 알아야 합니다.

![New transactional workflow.png](/assets/img/2023-11-06-nestjs-transactional-on-multidb/New%20transactional%20workflow.png)

우선 Metadata Cacher 인스턴스가 생겼습니다. 이 Metadata Cacher 은 Tx에 관한 두개의 데이터를 저장하고 있습니다.

### TxAbleMethodMetaData

TxAbleMethodMetaData는 TxAbleMethod의 Tx 메타데이터를 저장합니다. **여기서 TxAbleMethod란 각 Repository에 있는 모든 메서드들, 즉 직접 DB에서 데이터를 주고 받는 계층에 있는 모든 메서드를 뜻합니다.** 이곳에 있는 메타데이터들은 전부 nestJs가 bootstrap되면서 저장됩니다. 즉 본격적인 request가 시작되기 전에 이곳의 데이터들은 모두 수집이 완료된 상태입니다.

이에 대해서는 이후에 더 자세히 다루겠습니다.

### TxMetaData

TxMetaData같은 경우 `@Transactional()` 로 감싸진 모든 함수에 대한 메타데이터 입니다. 이 정보는 맨 처음에는 아무것도 저장되어 있지 않으며, 함수가 한번 실행된 이후에 그 함수에 대한 메타데이터를 저장하는 공간입니다.

그러면 이제 한번 어떻게 이 새로운 Transactonal이 동작하는지 살펴보겠습니다.

우선 이 새로운 트랜잭션에는 두 가지 케이스에 따라 분기되어 동작합니다.

### Original method’s Tx meta data is exist

첫 번째는 `@Transactional()` 이 감싸고 있는 메서드가 **본인이 TxManager를 생성하기 위한 메타데이터를 Metadata Cacher 인스턴스의 TxMetadata**에서 발견했다고 가정합니다. (TxMetadata가 어떻게 메타데이터를 보유하고 있는지는 나중에 설명하겠습니다.)

1.  Metadata Cacher 인스턴스의 TxMetadata에서 TxManager를 생성하기 위한 메타데이터를 가져옵 니다.
2.  메타데이터에 따라서 TxManager를 만들고, cls-hook namespace에 주입합니다.
3.  주입 후 다시 namespace에 TxManager를 가져와서 트랜잭션을 시작합니다.
4.  원본 메서드를 호출합니다.
5.  Tx를 종료하고 Commit, Rollback을 수행합니다.

### Original method’s Tx meta data is not exist

두 번째는 `@Transactional()` 이 감싸고 있는 메서드가 **본인이 TxManager를 생성하기 위한 메타데이터를 Metadata Cacher 인스턴스의 TxMetadata에 존재하지 않을 경우 입니다.** 이 경우 다음과 같은 행동을 진행합니다.

1. Metadata Cacher 인스턴스의 TxMetadata에서 TxManager를 생성하기 위한 메타데이터를 가져오지 못했습니다.
2. 일단 mongoDB와 neo4j write tx에 대한 TxManager를 생성해서 비효율적이더라도 현재 요청이 정상적으로 작동되게 합니다.(락 수준을 가장 높힌 상태로 Tx를 겁니다.)
3. 이와 동시에 original method가 실행되면서 original method가 호출했던 모든 method들을 기록합니다.
4. 성공적으로 오류 없이 메서드의 작업이 완료 되었다면, original method가 호출하며 기록된 모든 함수들에 대한 Tx 메타데이터를 **Metadata Cacher의 TxAbleMethodMetaData에서 조회**합니다.
5. 유효한 Tx 메타데이터들을 모아서 , 최종적으로 현재 original method가 가져야 할 Tx Metadata를 결정합니다.
6. 결정한 Tx 메타데이터를 [original method 있는 class name]-[original method name]을 key값을 고 해서 **Metadata Cacher의 TxMetadata에 저장합니다.**
7. 다음번에 같은 함수가 호출되면 **Metadata Cacher의 TxMetadata가 존재하므로 존재 할 때의 동작을 따릅니다.**

## JS Proxy and Transactional()

그렇다면 도대체 JS의 Proxy는 어디에 사용될까요? **Original method’s Tx meta data is not exist의 3번에서 사용됩니다**. 코드로 설명하기 전에 먼저 아래 다이어그램과 설명을 봅시다.

![Proxy in transactional.png](/assets/img/2023-11-06-nestjs-transactional-on-multidb/Proxy%20in%20transactional.png)

간단하게 설명하면 만약에 original method의 메타데이터가 Metadata cacher의 TxMetaData에 존재하지 않는다면 `@Transactional()` 은 특이한 동작을 수행하기 시작합니다.

original method가 속한 클래스내부에 선언된 객체를 전부 가져와서 객체에 대한 proxy를 생성합니다. 이후 생성한 proxy를 전부 원래 선언되었던 객체에 덮어씌웁니다. 이러면 만약에 original method에서 코드상에서 `b.method()` 를 호출하더라도, 사실은 `bProxy.method()`를 호출한 것이 됩니다.

그렇다면 도대체 객체의 proxy는 어떤 fundamental operation을 가로채고 있을까요?

![Proxy in transactional 2.png](/assets/img/2023-11-06-nestjs-transactional-on-multidb/Proxy%20in%20transactional%202.png)

위에서 생생한 객체의 proxy는 객체 내부에 선언된 함수, 즉 method를 호출할 때 메서드의 이름을 기록해서 임시로 `@Transactional()` 내부에 성성된 리스트인 **txAbleMethodMetaDataKeyList**에 저장하는 역할을 합니다. fundamental operation의 get method를 가로채서 기록하는 것이죠.

그렇다면 코드로 어떻게 구현했는지 살펴보겠습니다.

아래 코드는 `@Transactional()` 에서 original method에서 사용하는 TxAbleMethod 정보를 동적으로 얻어오는데 핵심이 되는 부분입니다.

```tsx
const txAbleMethodMetaDataKeyList: Array<string> = [];

for (const key in this) {
  if (this[key] && typeof this[key] === "object") {
    const proxy = new Proxy(this[key], {
      get: (target, property, receiver) => {
        const originalMethod = target[property];
        if (typeof originalMethod === "function") {
          const txAbleMethodMetadataKey = `${
            target.constructor.name
          }-${property.toString()}`;

          txAbleMethodMetaDataKeyList.push(txAbleMethodMetadataKey);

          return originalMethod;
        }
        return Reflect.get(target, property, receiver);
      }
    });

    this[key] = proxy;
  }
}
```

매우 중요한 부분이기 떄문에 코드를 한줄 한줄 설명해보겠습니다.

```tsx
for (const key in this) {
  if (this[key] && typeof this[key] === 'object')
```

맨 처음에 있는 부분의 `this` 는 `@transactional()` 데코레이터가 감싸고 있는 원본 메서드가 속한 객체를 가르킵니다. 즉 this 내부에 나열되어 있는 key값은 class내부에 속한 변수, 메서드, 주입받은 객체가 될 수 있습니다. 그 중에 object 타입만 선택한다는 것은 결국 클래스 내부의 객체들만 선택한다는 말이 됩니다.

```tsx
const proxy = new Proxy(this[key], {
  get: (target, property, receiver) => {
    const originalMethod = target[property];
    if (typeof originalMethod === "function") {
      const txAbleMethodMetadataKey = `${
        target.constructor.name
      }-${property.toString()}`;

      txAbleMethodMetaDataKeyList.push(txAbleMethodMetadataKey);

      return originalMethod;
    }
    return Reflect.get(target, property, receiver);
  }
});
```

그리고 아래와 같이 주입받은 각 객체에 대해서 Proxy를 생성하는데, 여기서 자세히 봐야할 부분은 get을 trap하고 있다는 점과 이떄 각 객체에 속한 메서드에 get을 통해서 접근하게 된다면 “클래스이름-메서드이름” 을 txAbleMethodMetaDataKeyList 배열에 저장하는 것을 확인할 수 있습니다.

그 외의 경우에는 get으로 접근해도 그냥 원본 get과 같게 동작합니다.

```tsx
this[key] = proxy;
```

마지막으로 생성한 proxy객체를 원본 객체에 덮어씌우는 것을 볼 수 있습니다.

결과적으로 만약 함수에 메타데이터가 없을경우 `@Transactional()` 에 감싸진 함수는 아래 같이 원본 객체를 호출하는 것이 아니라 프록시를 호출하는 것과 같은 동작을 하는 함수와 똑같게 변합니다.

```tsx
//service

@Transactional()
async create (createDto: CreateDto): Promise<void> {

    await this.neo4jRepositoryProxy.createNode(neo4jModel);
    await this.mongoRepositoryProxy.create(mongoModel);
}
```

이렇게 되면 예를 들어서 `this.neo4jRepsitoryProxy,createNode(neo4JModel)` 메서드가 호출 된다고 하면 get trap이 발동해서 `txAbleMethodMetaDataKeyList` 에 함수명이 저장되게 됩니다.

이 상태로 오류 없어 성공적으로 요청이 마무리 되게 되면, 결국 `txAbleMethodMetaDataKeyList` 내부에 호출된 모든 메서드에 대한 리스트가 저장되게 됩니다.

이후에는 위에 적었던 flow와 같게 `txAbleMethodMetaDataKeyList` 에 저장된 메서드 들 중 Metadata Cacher의 TxAbleMethodMetaData 스토리지에 저장된 메타데이터가 있는지를 확인합니다. 그리고 있다면 각 메서드에 대한 메타데이터를 모두 불러와서 계산해서 `@Transactional()` 이 감싼 메서드가 가져야 할 Tx에 대한 메타데이터를 결정하고, TxMetadata에 저장합니다.

## Metadata Cacher TxMetaData Storage

자 그러면 이제 거의 다 되었습니다. 이제 그러면 Metadata Cacher에서 TxMetaData가 어떻게 채워지는지 알아 보겠습니다.

![Metadata cacher.png](/assets/img/2023-11-06-nestjs-transactional-on-multidb/Metadata%20cacher.png)

먼저 위에서 말했듯이 Proxy를 통해서 Original method에서 호출된 method의 리스트를 가져옵니다. 그 이후에 어떤 일이 일어나는지 확인할 필요가 있습니다.

1. Proxy의 get trap에 의해 호출된 함수는, `@Transactional()` 의 내부에 선언된 **txAbleMethodMetaDataKeyList에 [class name]-[method name]으로** 저장됩니다. 위의 예시에서는 a.method1()이 a-method1으로 저장되는 것을 볼수 있습니다.
2. 어떤 경로로 Metadata Cacher의 **TxAbleMethodMetaData가 차 있는지는 모르지만** 일단 있다고 가정해 봅니다. 이때, 아까 위에서 저장했던 txAbleMethodMetaDataKeyList 배열에 있는 함수 중 Metadata Cacher에 정보가 있는 것도 있고 없는 것도 있습니다.

   만약 정보가 있다면 정보를 받아오고, 메타데이터 정보가 없으면 무시합니다. 그러면 최종적으로 위의 예시에서는 **a-method: {neo4j :write, mongo: false}, b-method:{neo4j:undefined, ture}** 라는 정보를 \*\*\*\*`@Transactional()`이 들고 있게 됩니다.

3. 위의 정보를 바탕으로 우리는 original method의 TxMetadata를 결정할 수 있습니다. 위의 예시에서는 **{neo4j :write, mongo: true}**라는 최종 TxMetadata가 결정됩니다.
4. 결정된 메타 데이터를 Metadata Cacher의 TxMetaData에 [original method 있는 class name]-[original method name]을 키 값으로 저장합니다.

## Metadata Cacher **TxAbleMethodMetaData** Storage

그러면 이제 도대체 TxAbleMethodMetaData는 어떻게 채워지는 것일까요?

우선 두 가지의 개념을 먼저 알아야 합니다.

1. TxAbleMethod
2. NestJs Basic

### TxAbleMethod

여기서 TxAbleMethod는 제가 만든 개념입니다. 새로운 `@Transactional()` 을 소개할 떄 잠시 본 코드를 가져와 보겠습니다.

```tsx
//controller

@Post()
async create(@Body() createDto: CreateDto) {
  await this.service.create(createDto);
  return { message: 'created successfully' };
}
```

```tsx
//service

@Transactional()
async create (createDto: CreateDto): Promise<void> {

    await this.neo4jRepository.createNode(neo4jModel);
    await this.mongoRepository.create(mongoModel);
}
```

```tsx
//repository

@Neo4jTransaction(Neo4jTransactionTypeEnum.WRITE)
async createNode(neo4jModel: Neo4jModel): Promise<Neo4jModel> {
  //create logic
}
```

여기서 마지막 repository의 `@Neo4JTransaction()` 이나 비슷한 역할을 하는 `@MongoTransaction()` 같은 데코레이터들이 바로 함수에 TxAble속성을 부여 해 주는 역할을 합니다. 즉 이러한 데코레이터는 코드를 작성하는 사람이 직접 Datasource에 access하는 함수 위에 붙혀서, 이 비즈니스 로직은 어떤 트랜잭션 유형을 가지고 있다를 Metadata를 통해서 명시해주어야 합니다.

그리고 이런 데코레이터들이 붙어 있는 메서드를 TxAbleMethod라고 하고 이 메서드를 인식해서 Metadata Cacher 는 TxAbleMethodMetaData에 관련한 메타데이터를 적제합니다.

## NestJS Basic

### Nestjs Lifecycle event

그렇다면 이제 진짜로 TxAbleMethodMetaData에 Metadata Cacher는 어떻게 데이터를 적제하는지 알아보겠습니다.

![NestJS lifecycle event.png](/assets/img/2023-11-06-nestjs-transactional-on-multidb/NestJS%20lifecycle%20event.png)

모든 프레임워크에는 각 프레임워크 내부의 인스턴스가 어떻게 관리되는지에 따라 각각 고유한 lifecycle이 있고, 그 lifecycle에 따라서 특정 시점에 걸리는 이벤트들이 존재합니다. 위에는 파란색 박스는 nestjs서버가 시작될 때 호출되는 이벤트 들입니다.

여기서 주목해야 할 점은 **OnApplicationBootstrap Event**입니다. 이 어플리케이션은 nestjs 내부에서 모든 인스턴스가 전부 초기화 된 뒤에야 이벤트가 발생합니다. 즉 이 시점에서 이벤트가 발생하면, 내부적으로 객체는 전부 생성되었으나, 아직 nestjs서버의 listener는 시작되지 않아, 내부적으로는 완료됬지만 외부에서 요청은 받지 못하는 상태입니다.

---

### Nestjs Module & Provider & controller

이 부분에 대한 설명은 다른 외부 글에도 많기 떄문에 간략하게 설명하겠습니다.

![NestJS basic structure.png](/assets/img/2023-11-06-nestjs-transactional-on-multidb/NestJS%20basic%20structure.png)

근본적으로 nestjs의 핵심적인 개념중에 Module, Provider, Controller가 있습니다. 여기서 우리가 중요하게 봐야 할 것은 nestjs는 모듈단위로 동작한다는 것입니다. 모듈안에 실제로 비즈니스 로직이나, 라우터기능을 하는 것을 전부 넣고, 모듈은 다른 모듈과의 의존성을 관리하고, 주입을 하는데 초점을 맞추고 있습니다.

**@Module()** 데코레이터가 붙어서 클래스가 모듈임을 표시하고 그 내부에 실제 기능을 하는 객체들을 적재합니다. 그리고 모듈에 적재될 수 있는 객체는 **@Controller()** 혹은 **@Injectable()**이 달려있는 객체들인데, 전자는 routing을 잡는 역할을 주로 수행하고 후자는 주로 대부분의 비즈니스 로직을 담당하며 nestjs에서 공식적으로 provider라고 불립니다.

```tsx
//module
@Module({
  imports: [Neo4jConfigModule],
  providers: [BRepository, ARepository, CRepository, TransactionalHelper],
  exports: [BRepository, ARepository, CRepository]
})
export class RepositoryModule {}
```

```tsx
//provider
@Injectable()
export class BRepository extends Repository {}
```

```tsx
//controller
@Controller({ path: "b", version: "1" })
export class BController {
  constructor(private readonly bService: BService) {}

  @Post()
  async create(@Body() createDto: CreateDto) {
    await this.bService.create(createDto);
    return { message: "created successfully" };
  }
}
```

위와 같이 순서대로 module, provider, controller입니다. 여기서 한 가지 더 생각해 봐야 할 점은 nestjs는 IOC를 지원하는 프레임워크 입니다. 즉 저 위에 데코레이터가 붙은 클래스들은 모두 nestjs의 DI와 IOC의 영향을 받습니다. 그렇다면, **OnApplicationBootstrap** 이벤트가 호출 됬을 쯤이면 어딘가에 DI와 IOC로 관리하기 위해서 nestjs 프레임워크 차원에서 모든 provider, controller, module의 데이터를 저장해 놓은 공간이 있지 않을까요?

### ModulesContainer

NestJS는 NestContainer 내부에 관련된 자료를 전부 저장합니다. 그 중에서도 NestContainer 내부에 ModulesContainer에 Module 단위로 모듈에 주입된 provider등에 대한 데이터들이 저장됩니다. 그렇다면 **OnApplicationBootstrap** 이벤트에 호출됬을 때, 이를 감지해서 그 즉시 ModulesContainer를 전체 팀색한다면 모든 Provider에 대한 정보를 얻을 수 있을 것이고, 각 provider도 전체 탐색한다면 provider 내부에 잇는 메서드의 정보와 method에 대한 메타데이터더도 얻을 수 있을 것입니다.

그렇게 해서 구현된 것이 Metadata Cacher의 TxAbleMethodMetaData 저장소입니다.

```tsx
@Injectable()
export class MetadataCache implements OnApplicationBootstrap {
  constructor(
    private readonly modulesContainer: ModulesContainer,
    private readonly reflector: Reflector
  ) {}

  private readonly txAbleMethodMetaData: Map<string, TransactionMetadata> =
    new Map();

  onApplicationBootstrap() {
    [...this.modulesContainer.values()].forEach((module) => {
      [...module.providers.values()].forEach((provider) =>
        this.scanProviderMethod(provider)
      );
    });
  }

  private scanProviderMethod(wrapper: InstanceWrapper) {
    const { instance } = wrapper;
    if (instance && typeof instance === "object") {
      const prototype = Object.getPrototypeOf(instance);
      Object.getOwnPropertyNames(prototype)
        .filter(
          (methodName) =>
            methodName !== "constructor" &&
            typeof prototype[methodName] === "function"
        )
        .forEach((methodName) => this.scanMethodMetadata(instance, methodName));
    }
  }

  private scanMethodMetadata(instance: any, methodName: string) {
    const neo4jTxMetadata: Neo4jTransactionType = this.reflector.get(
      NEO4J_TRANSACTION_METADATA,
      instance[methodName]
    );

    const mongoTxMetadata = this.reflector.get(
      MONGO_TRANSACTION_METADATA,
      instance[methodName]
    );

    if (neo4jTxMetadata || mongoTxMetadata) {
      const key = `${instance.constructor.name}-${methodName}`;
      const txMetadata: TransactionMetadata = {
        neo4jTransactionMetadata: neo4jTxMetadata,
        mongoTransactionMetadata: mongoTxMetadata
      };

      this.txAbleMethodMetaData.set(key, txMetadata);
    }
  }
}
```

복잡해 보이지만 사실 간단합니다.

1. `onApplicationBootstrap` : 모든 인스턴스가 초기화 되고 실행되고 그 아레에 있는 함수는 전부 modules containers에서 모든 모듈을 불러오고 모듈 내부에서 모든 provider를 하나씩 탐색합니다.
2. `scanProviderMethod` : provider하나에 있는 모든 메서드를 탐색합니다.
3. `scanMethodMetadata` : 메서드에 있는 메타 데이터중 TxAbleMethod에 걸릴 수 있는 메타데이터를 스캔해서 만약에 존해한다면 [class name]-[method name]을 키값으로 해서 메타데이터를 txAbleMethodMetaData에 저장합니다.

이렇게 하면 최종적으로 메타데이터가 txAbleMethodMetaData에 등록되게 됩니다.

드디어 끝났습니다. 이 프로젝트로 드디어 @Transactional() 을 다음처럼 사용할 수 있게 되었습니다.

```tsx
//controller

@Post()
async create(@Body() createDto: CreateDto) {
  await this.service.create(createDto);
  return { message: 'created successfully' };
}
```

```tsx
//service

@Transactional()
async create (createDto: CreateDto): Promise<void> {

    await this.neo4jRepository.createNode(neo4jModel);
    await this.mongoRepository.create(mongoModel);
}
```

```tsx
//repository

@Neo4jTransaction(Neo4jTransactionTypeEnum.WRITE)
async createNode(neo4jModel: Neo4jModel): Promise<Neo4jModel> {
  //create logic
}
```

## Limitation

이렇게 만든 트랜잭션은 한계가 몇 가지 있습니다.

1. namesapce가 요청단위로 분할되지 않습니다. 사실 이 부분은 다시 미들웨어로 namspace로 runAndRetrurn 함수만 옮기면 해결할 수 있습니다. 다만 현재는 @Transactional() 단위로 분리되어 있습니다.
2. 맨 처음 실행될 때 느립니다. bootstrap 과정에서 전체 탐색을 진행하기 때문에 느릴 수 있습니다.
3. 맨 처음 요청을 받을 때 느립니다. 요청이 처음 들어오면 그 요청에 대한 TxMetadata를 생성해야 하므로 첫 요청이 느립니다. 평균적으로 테스트 해본 결과 200~300ms 의 차이가 발생하였습니다.
4. service에서 무조건 metadataCache를 주입받아야 합니다.`@Transactional()` 에서 Metadata Cacher에 접근해야 하기에 어쩔 수 없는 선택이였습니다. 이 부분이 치명적인 단점이라고 생각하는데, 아직까지는 해법을 찾지는 못했습니다. 아마 `@Injectable()`데코레이터를 커스터마이징 하거나 모든 service레이어 계층에 공통적인 abstract class두는 것이 해결 방법일 것 같습니다.

## Outro

아직 정식으로 production 환경에서 쓰기에는 많은 테스트가 필요하다고 생각합니다. 그러나, 본 프로젝트를 진행하면서 결과적으로 프로젝트의 전반적인 코드가 깔끔하게 리펙토링 되었기 때문에, 충분히 시도해 볼만한 가치가 있는 프로젝트였다는 생각이 듭니다.

현재는 프로젝트에 통합되어 있기에, 따로 원본코드를 repository로 빼서 첨부하지 않았지만, 추후 기회가 된다면 새롭게 repository를 만들어서 제공하겠습니다.
