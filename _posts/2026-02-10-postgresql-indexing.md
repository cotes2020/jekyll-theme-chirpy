---
title: PostgreSQL 인덱스 최적화 실전 가이드
date: 2026-02-10 10:00:00 +0900
categories: [데이터베이스, PostgreSQL]
tags: [postgresql, database, 인덱스, 성능최적화, sql]
---

## 인덱스가 필요한 이유

테이블이 수백만 건 이상이 되면 `WHERE`, `JOIN`, `ORDER BY` 절의 성능이 급격히 떨어집니다. 인덱스는 이 문제를 해결합니다.

## 실행 계획 확인

```sql
EXPLAIN ANALYZE
SELECT * FROM orders
WHERE user_id = 1234
ORDER BY created_at DESC;
```

`Seq Scan` → 인덱스 없이 전체 탐색 (느림)
`Index Scan` → 인덱스 사용 (빠름)

## B-Tree 인덱스 (기본)

```sql
-- 단일 컬럼 인덱스
CREATE INDEX idx_orders_user_id ON orders(user_id);

-- 복합 인덱스 (순서 중요!)
CREATE INDEX idx_orders_user_created ON orders(user_id, created_at DESC);
```

## 부분 인덱스

조건을 걸어 인덱스 크기를 줄일 수 있습니다.

```sql
-- 활성 상태인 주문만 인덱싱
CREATE INDEX idx_active_orders ON orders(user_id)
WHERE status = 'active';
```

## GIN 인덱스 (JSONB, 배열)

```sql
CREATE INDEX idx_metadata ON products USING GIN(metadata);

-- 검색
SELECT * FROM products WHERE metadata @> '{"category": "전자제품"}';
```

## 인덱스 사용 현황 확인

```sql
SELECT
    schemaname,
    tablename,
    indexname,
    idx_scan,
    idx_tup_read,
    idx_tup_fetch
FROM pg_stat_user_indexes
ORDER BY idx_scan DESC;
```

## 주의사항

- 인덱스가 많을수록 **INSERT/UPDATE** 성능 저하
- 쓰기보다 읽기가 훨씬 많은 컬럼에 생성
- 카디널리티(다양성)가 낮은 컬럼은 인덱스 효과가 적음 (예: boolean 컬럼)
