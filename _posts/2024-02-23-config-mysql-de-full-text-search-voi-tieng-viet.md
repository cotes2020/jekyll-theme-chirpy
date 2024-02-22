---
published: true
date: 2024-01-05
title: Config MySQL để full-text search với tiếng Việt
---
Thêm các config sau cho file my.cnf trong thư mục cài đặt MySQL:

    ft_min_word_len=1
    ft_stopword_file=""
    innodb_ft_enable_stopword="OFF"
    innodb_ft_min_token_size=1

Config này có 2 chức năng chính:

*   Tắt danh sách stopword đi (danh sách stopword bao gồm các từ mà nếu có trong query string thì sẽ không trả về kết quả. Ví dụ như “a”).
    
*   Giảm token size của full-text search về 1 để database có thể tạo index cho các từ dưới 3 kí tự. Mình vẫn chưa biết liệu việc này có ảnh hưởng gì không nhưng trước mắt vẫn hoạt động tốt.