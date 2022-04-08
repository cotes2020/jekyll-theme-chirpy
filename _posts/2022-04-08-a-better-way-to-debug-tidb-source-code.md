---
title: A Better Way to Debug TiDB Source Code
author: Hulua
date: 2022-04-08 20:55:00 +0800
categories: [Database, TiDB]
tags: [TiDB, debug]
---


[TiDB]( https://github.com/pingcap/tidb ) is an interesting SQL database written in Golang. One major difference compared with popular exsiting relational databases is that in TiDB storage is separated as an independant runtime. The database (SQL) layer will call the storage layer to access data.

For database learners, most of us actually will only be interest in the SQL part, i.e, from SQL statment in string form how we can get the query result. Although other parts like server or authtication are also essential parts for a database product, we do not really care about them if we just need to learn database principles.

The official [document](https://pingcap.github.io/tidb-dev-guide/understand-tidb/the-lifecycle-of-a-statement.html) demenstrated the whole lifecyle of an SQL statement in the production environment, and we can follow this [guide](https://pingcap.github.io/tidb-dev-guide/get-started/debug-and-profile.html) to debug TiDB.

However, the above debug process is cubersome. You will launch the database server, use MySQL client to send SQL querys. This is not really nessesary if we just want to track the execution of a single SQL statement.  Meanwhile, when you set a break point and want to observe your expected SQL statement, the debugger will also break if it encounter some other statements triggered by the server itself or some other components, this is annoying. Is there a better way to debug a single SQL statement?

That is the motivation for this post. We will show how we can achieve this using the below code.

```go

/*
A simple Go program aims for better debugging SQL execution process in TiDB.
Thus you can focus on Parsing, OPtimization and Execution without distraction
from the server part and connecton part. In addtion, no MySQL client is required.

Usuage:

1. Go to the tidb source folder, and create a folder like 'sqlexec', 
   and copy the following code into sqlexec/main.go

2. Compile and execute:
go build sqlexec/main.go && ./main

3. Debug:
dlv debug sqlexec/main.go

Enjoy!
*/

package main

import (
	"context"
	"fmt"

	"github.com/pingcap/tidb/parser/terror"
	"github.com/pingcap/tidb/session"
	kvstore "github.com/pingcap/tidb/store"
	"github.com/pingcap/tidb/store/mockstore"
)

func main() {
	//1. Prepare storage, use your existing store path, tables are there
	storePath := "unistore:///tmp/tidb"
	err := kvstore.Register("unistore", mockstore.EmbedUnistoreDriver{})
	terror.MustNil(err)
	storage, err := kvstore.New(storePath)
	terror.MustNil(err)

	//2. Prepare a session and set the working database
	se, err := session.CreateSession4Test(storage)
	terror.MustNil(err)
	se.GetSessionVars().CurrentDB = "test"
	terror.MustNil(err)

	//3. The sql statement you want to execute
	//sql := "insert into t1 values(5,'lucy','newyork','female')"
	sql := "select * from t1 where id <=3"

	//4. Parse and Execute.
	ctx := context.Background()
	stmts, err := se.Parse(ctx, sql)
	terror.MustNil(err)
	rs, err := se.ExecuteStmt(ctx, stmts[0])
	terror.MustNil(err)

	//5. Get result for select statement
	if rs != nil {
		sRows, err := session.ResultSetToStringSlice(ctx, se, rs)
		terror.MustNil(err)
		fmt.Println("Query Result:", sRows)
	} else {
		fmt.Println("Execution Succeed")
	}
}
``` 

```console
go build sqlexec/main.go && ./main
[2022/04/08 08:29:01.448 -04:00] [INFO] [store.go:74] ["new store"] [path=unistore:///tmp/tidb]
[2022/04/08 08:29:01.493 -04:00] [INFO] [db.go:143] ["replay wal"] ["first key"=7580000000000000255f728000000000000001f9ffe9defe5fffff(432369895244562432)]
[2022/04/08 08:29:01.494 -04:00] [INFO] [store.go:80] ["new store with retry success"]
[2022/04/08 08:29:01.494 -04:00] [INFO] [tidb.go:72] ["new domain"] [store=6ebcd41f-95db-42e6-a785-e40ae59d567b] ["ddl lease"=1s] ["stats lease"=3s] ["index usage sync lease"=0s]
[2022/04/08 08:29:01.503 -04:00] [INFO] [domain.go:169] ["full load InfoSchema success"] [currentSchemaVersion=0] [neededSchemaVersion=37] ["start time"=5.741453ms]
[2022/04/08 08:29:01.503 -04:00] [INFO] [domain.go:432] ["full load and reset schema validator"]
[2022/04/08 08:29:01.503 -04:00] [INFO] [ddl.go:366] ["[ddl] start DDL"] [ID=c7216a3b-520a-4471-86fe-080e30c4d09b] [runWorker=true]
[2022/04/08 08:29:01.504 -04:00] [INFO] [ddl.go:355] ["[ddl] start delRangeManager OK"] ["is a emulator"=true]
[2022/04/08 08:29:01.504 -04:00] [WARN] [sysvar_cache.go:54] ["sysvar cache is empty, triggering rebuild"]
[2022/04/08 08:29:01.504 -04:00] [INFO] [delete_range.go:142] ["[ddl] start delRange emulator"]
[2022/04/08 08:29:01.504 -04:00] [INFO] [ddl_worker.go:156] ["[ddl] start DDL worker"] [worker="worker 1, tp general"]
[2022/04/08 08:29:01.504 -04:00] [INFO] [ddl_worker.go:156] ["[ddl] start DDL worker"] [worker="worker 2, tp add index"]
Query Result: [[1 zhangsan chongqing male] [2 lishi beijing female] [3 wang shandong male]]

```

With the above code, you can perfectly dive into the internals of SQL execution process, no need to worry about server start, connect with MySQL client, etc. After get familar with the the three essential steps (Parser, Planner, Executor, this may take some time), we can then go back to learn other components.