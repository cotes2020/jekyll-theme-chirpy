---
title: MySQL Explain JSON
author: Tai Le
date: 2021-12-25
tags: [Back-end Engineering, Database, MySQL]
---

Asides from the tasks that my team is working on in my company, I am also supporting a software architect to find some solutions to optimize the slow SQL queries in several functionalities. And now I will share the experience that I have learned while working on this task.


## 1. Introduction

Slow SQL queries come from multiple sources, perhaps the business is fairly complicated, or the implementation of the programmers make it complicated. Especially in the large applications, where features are developed across multiple team and everything isn't implemented in one consistent way. But our job is not to blame for the things have been done, we need to resolve them for a better and faster application.

In my case, obviously there are many queries that I haven't found any solutions yet, and my approach maybe doesn't seem correct to some of you, but I would like to share it as a growing programmer. I haven't tried partitioning yet because it breaks the data apart without returning point, and taking that action requires a deep understanding of business logic. Therefore, I only focus on indexing, the approach includes two steps:

1. Debug and analyze the slow queries
2. Insert better indexes


## 2. Debug and analyze the slow queries

#### a. Source of the slow queries

As I already told you, there are several factors that cause slow queries, the business logic is complex or the programmers make it complex. On top of that, according to Schwarts (2012): "MySQL uses a cost-based optimizer, which means it tries to predict the cost of various execution plans and choose the least expensive", the cost is evaluated by many metrics:

- The number of pages per table or index
- The cardinality
- The length of the rows and keys
- The key distribution

There are also several cases that high-cost queries produce faster result than low-cost ones (often due to FileSort). In general, the best approach in each situation depends on the business logic and rules that you are working on.


#### b. MySQL Explain JSON

MySQL Explain Statement is a great tool that I use to understand my queries better, I can know the candidate indexes and which one is chosen based on the information that it shows. I will explain some of the concepts below that are necessary to reason a query is slow. We will use [Sakila](https://dev.mysql.com/doc/sakila/en/) database to demonstrate our JSON.

__Describe table:__
```text
mysql> DESCRIBE film;
+----------------------+---------------------------------------------------------------------+------+-----+-------------------+-----------------------------+
| Field                | Type                                                                | Null | Key | Default           | Extra                       |
+----------------------+---------------------------------------------------------------------+------+-----+-------------------+-----------------------------+
| film_id              | smallint(5) unsigned                                                | NO   | PRI | NULL              | auto_increment              |
| title                | varchar(128)                                                        | NO   | MUL | NULL              |                             |
| description          | text                                                                | YES  |     | NULL              |                             |
| release_year         | year(4)                                                             | YES  |     | NULL              |                             |
| language_id          | tinyint(3) unsigned                                                 | NO   | MUL | NULL              |                             |
| original_language_id | tinyint(3) unsigned                                                 | YES  | MUL | NULL              |                             |
| rental_duration      | tinyint(3) unsigned                                                 | NO   |     | 3                 |                             |
| rental_rate          | decimal(4,2)                                                        | NO   |     | 4.99              |                             |
| length               | smallint(5) unsigned                                                | YES  |     | NULL              |                             |
| replacement_cost     | decimal(5,2)                                                        | NO   |     | 19.99             |                             |
| rating               | enum('G','PG','PG-13','R','NC-17')                                  | YES  |     | G                 |                             |
| special_features     | set('Trailers','Commentaries','Deleted Scenes','Behind the Scenes') | YES  |     | NULL              |                             |
| last_update          | timestamp                                                           | NO   |     | CURRENT_TIMESTAMP | on update CURRENT_TIMESTAMP |
+----------------------+---------------------------------------------------------------------+------+-----+-------------------+-----------------------------+
```

__List indexes:__
```
mysql> SHOW INDEX FROM film;
+-------+------------+-----------------------------+--------------+----------------------+-----------+-------------+----------+--------+------+------------+---------+---------------+
| Table | Non_unique | Key_name                    | Seq_in_index | Column_name          | Collation | Cardinality | Sub_part | Packed | Null | Index_type | Comment | Index_comment |
+-------+------------+-----------------------------+--------------+----------------------+-----------+-------------+----------+--------+------+------------+---------+---------------+
| film  |          0 | PRIMARY                     |            1 | film_id              | A         |        1000 |     NULL | NULL   |      | BTREE      |         |               |
| film  |          1 | idx_title                   |            1 | title                | A         |        1000 |     NULL | NULL   |      | BTREE      |         |               |
| film  |          1 | idx_fk_language_id          |            1 | language_id          | A         |           1 |     NULL | NULL   |      | BTREE      |         |               |
| film  |          1 | idx_fk_original_language_id |            1 | original_language_id | A         |           1 |     NULL | NULL   | YES  | BTREE      |         |               |
+-------+------------+-----------------------------+--------------+----------------------+-----------+-------------+----------+--------+------+------------+---------+---------------+
```

__Visualize the first query:__
```sql
EXPLAIN format=json
    SELECT * FROM film
    WHERE rental_duration = 5 AND language_id = 1 AND film_id != 1
    ORDER BY last_update
    LIMIT 10;
```

__Output:__
```json
{
  "query_block": {
    "select_id": 1,
    "cost_info": {
      "query_cost": "203.09"
    },
    "ordering_operation": {
      "using_filesort": true,
      "table": {
        "table_name": "film",
        "access_type": "range",
        "possible_keys": [
          "PRIMARY",
          "idx_fk_language_id"
        ],
        "key": "PRIMARY",
        "used_key_parts": [
          "film_id"
        ],
        "key_length": "2",
        "rows_examined_per_scan": 501,
        "rows_produced_per_join": 50,
        "filtered": "10.00",
        "cost_info": {
          "read_cost": "193.07",
          "eval_cost": "10.02",
          "prefix_cost": "203.09",
          "data_read_per_join": "27K"
        },
        "used_columns": [
          "film_id",
          "title",
          "description",
          "release_year",
          "language_id",
          "original_language_id",
          "rental_duration",
          "rental_rate",
          "length",
          "replacement_cost",
          "rating",
          "special_features",
          "last_update"
        ],
        "attached_condition": "((`sakila`.`film`.`language_id` = 1) and (`sakila`.`film`.`rental_duration` = 5) and (`sakila`.`film`.`film_id` <> 1))"
      }
    }
  }
}
```

You should pay attention to some important keys like:

- `query_cost`: The execution cost of the whole query.

- `using_filesort`: This key appears when you use `ORDER BY`. If the value is `true`, it means there is no index used to sort the rows, MySQL has to read the table rows and sort them. Filesort is extremely slow and should be avoided.

- `access_type`: The document for this key is vary, but you can refer to the table below (from MySQL document).
![Access Type Table](/assets/img/2021-12-25/access-type-table.png)

- `possible_keys`: List of candidate indexes that can be used for the query. If no index found, the key is not available.

- `key`: Contains the index which the optimizer choose. If this key is not available and `possible_keys` has one element, the optimizer uses that element itself.

- `used_key_parts`: Parts of the index that are used. This key is important when you use composite indexes.

- `rows_examined_per_scan`

- `rows_produced_per_join`

- `filtered`

- `cost_info`

- `nested_loop`: MySQL uses nested loops to join tables, knowing this key is also necessary.

There are more items than just a few of these related to `JOIN`, `DISTINCT`, `GROUP BY`, etc. But being able to understand the keys above is probably enough to understand where is the problem.


## 3. Insert better indexes

After we understand our query, we should be able to improve our queries by adding some indexes. In general, low-cost queries are the most efficient queries and should be used. You should:

- Check the part which has high `cost_info` to decide whether an index is needed or not
- Avoid `filesort`, which is critical for large tables
- Optimize to have a low query cost
    - Remove unnecessary columns, joins
    - Change the programming rules to reduce the weight for database.
- ...

In my case, even though the optimizer used a lower-cost and higher `filtered` value index, it has `filesort: true` and the whole query costs ~2 seconds.


## 4. Conclusion

These things are all that I want to share, I will continue updating this article whenever I find something useful in the process. Merry Christmas and Happy New Year, everyone.


## 5. References:

- [High Performance MySQL](https://www.amazon.com/High-Performance-MySQL-Optimization-Replication/dp/1449314287)
- [Visual Explain Plan](https://dev.mysql.com/doc/workbench/en/wb-performance-explain.html)


