---
title: 'Một cách đồng bộ dữ liệu từ PostgreSQL sang PostgreSQL: Foreign Data Wrapper'
date: 2023-05-10 22:45:00 +0800
categories: [Tutorial]
tags: [PostgreSQL]
pin: true
---

# Bài toán

Việc gom dữ liệu từ các nguồn về thành một nguồn duy nhất là công việc thường gặp ở vị trí Data Engineer, Data Analyst, Data Scientist. Dữ liệu ở các cơ sở dữ liệu (CSDL) khác nhau sẽ có nhu cầu được tập hợp về một nơi để dễ thực hiện truy vấn. Một cách thường thấy là sử dụng một tool của bên thứ ba để sync data từ CSDL nguồn đến CSDL đích

Bài viết này muốn giới thiệu một kỹ thuật khác. Trong PostgreSQL có một kỹ thuật mà ta có thể từ vị trí của một CSDL nguồn có thể kết nối tới và lấy dữ liệu từ CSDL đích, gọi là **Foreign Data Wrapper (FDW)**

# Các bước thực hiện

## Bước 0: Một số chuẩn bị ban đầu

### Bước 0.1: Chuẩn bị 2 database

Thiết lập CSDL nguồn

```bash
docker run --name database-source -e POSTGRES_PASSWORD=password_source -p 2000:5432 -d postgres
```

Cấu hình CSDL nguồn:

- Host: 127.0.0.1
- Port: 2000
- User: postgres
- Password: password_source
- Database: postgres

Thiết lập CSDL đích

```bash
docker run --name database-target -e POSTGRES_PASSWORD=password_target -p 3000:5432 -d postgres
```

Cấu hình CSDL đích:

- Host: 127.0.0.1
- Port: 3000
- User: postgres
- Password: password_target
- Database: postgres

### Bước 0.2: Chuẩn bị data

Giả sử ta có một bảng transaction có cấu trúc như sau

```sql
CREATE TABLE public."transaction" (
	id uuid NOT NULL DEFAULT gen_random_uuid(),
	transaction_date timestamp NULL,
	item_name varchar NULL,
	item_price numeric NULL
);
```

Sau đó Insert một vài dữ liệu random vào bảng

```sql
INSERT INTO public."transaction" (transaction_date,
                                  item_name,
                                  item_price)
SELECT CURRENT_TIMESTAMP,
       md5(random()::text),
       (random() * (10000000-1000+1) + 1000)::int
FROM generate_series(1, 5);
```

### Bước 0.3: Chuẩn bị các tài khoản postgresql

Để có thể CSDL đích có thể truy cập và lấy dữ liệu, thì ở phía CSDL nguồn cần có một user có quyền read dữ liệu đó. Trong bài viết này sử dụng user **postgres** nhưng trong thực tế nên tạo riêng một user, chỉ cấp quyền đủ dùng

## Bước 1: Cài đặt extension

Truy cập vào CSDL đích và thực hiện câu lệnh:

```sql
CREATE EXTENSION postgres_fdw;
```

## Bước 2: Tạo server

Tại CSDL đích, thực hiện câu lệnh sau để tạo một server

```sql
CREATE server any_server_name 
FOREIGN DATA WRAPPER postgres_fdw 
OPTIONS (host '127.0.0.1', dbname 'postgres', port '2000')
```

Trong đó:

- Các giá trị **host**, **dbname**, **port** là cấu hình của CSDL nguồn
- **any_server_name** là một cái tên bất kỳ

## Bước 3: Tạo user mapping

Để tạo kết nối tới CSDL đích, thực hiện câu lệnh sau đây tại CSDL nguồn

```sql
CREATE USER mapping 
FOR CURRENT_USER
SERVER any_server_name 
OPTIONS (user 'postgres', password 'password_source')
```

Lưu ý: Khi muốn setup cho user nào thấy được dữ liệu thì thay **CURRENT_USER** bằng user đó và thực hiện câu lệnh. Chỉ user nào được setup ở bước này mới được nhìn thấy dữ liệu của CSDL nguồn

## Bước 4: Import foreign data

```sql
IMPORT FOREIGN SCHEMA public -- schema của CSDL nguồn
--LIMIT TO (table_name_1, table_name_2) --có thể import nhiều bảng một lúc
LIMIT TO (transaction)
FROM server any_server_name 
INTO public; -- schema của CSDL đích
```

Hoặc cách khác:

```sql
CREATE FOREIGN TABLE public.transaction (
	id uuid NOT NULL,
	transaction_date timestamp NULL,
	item_name varchar NULL,
	item_price numeric NULL
)
SERVER any_server_name
OPTIONS (schema_name 'public', table_name 'transaction');
```

Bảng public.transaction ở CSDL đích lúc này được gọi là **Foreign Table**

# Tổng kết

Trên đây là hướng dẫn chi tiết từng bước setup một foreign data wrapper. Một số điểm mạnh và yếu của kỹ thuật này như sau

## Điểm mạnh

- Cách setup nhanh chóng
- Không cần dùng đến công cụ của bên thứ ba
- Có thể được sử dụng như là một phương pháp stream data từ CSDL nguồn đến CSDL đích theo thời gian thực
- Không chiếm dung lượng ổ đĩa của CSDL đích

## Điểm yếu

- Trong quá trình làm việc, tác giả nhận thấy điểm yếu lớn nhất của phương pháp này so với việc đồng bộ dữ liệu là tốc độ truy vấn dữ liệu. Khi xuất hiện nhu cầu JOIN các bảng mà trong đó có xuất hiện Foreign Table thì sẽ chậm hơn đối với các Physical Table. Như vậy cần chú ý khi phải thực hiện việc SELECT, JOIN với lượng dữ liệu lớn, câu query phức tạp