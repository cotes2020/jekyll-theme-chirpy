---
layout: post
title: "Xiao Steganography"
summary: "Introduction to Xiao Steganography"
author: technical
date: '2020-10-23 09:00:00'
category: Tools
thumbnail: assets/img/thumbnail/xiao-steganography.png
keywords: Hacking, Information Gathering
permalink: /blog/Xiao-Steganography/
usemathjax: true
---

<p>Xiao Steganography là một công cụ <strong><a href="https://vi.wikipedia.org/wiki/K%E1%BB%B9_thu%E1%BA%ADt_gi%E1%BA%A5u_tin">steganography</a></strong> cho phép người dùng ẩn giấu tập tin bí mật trong tập tin hình ảnh (<strong>BMP</strong>) hoặc tập tin âm thanh (<strong>WAV</strong>). Công cụ này cũng cho phép người dùng mã hóa tập tin bí mật bằng nhiều thuật toán mã hóa được hỗ trợ (bao gồm <strong>RC4</strong> và <strong>3DES</strong>) và thuật toán <strong>hash</strong> (bao gồm <strong>SHA</strong> và <strong>MD5</strong>). Người dùng cung cấp tập tin bên ngoài (là lớp vỏ để bao bên ngoài tập tin bí mật), tập tin cần ẩn giấu, lựa chọn thuật toán mã hóa và khóa bí mật. Để giải nén tập tin bí mật, người dùng cần cung cấp khóa bí mật.</p>
<h1 id="download">Download</h1>
<blockquote>
<p><a href="https://download.cnet.com/Xiao-Steganography/3000-2092_4-10541494.html">https://download.cnet.com/Xiao-Steganography/3000-2092_4-10541494.html</a></p>
</blockquote>
<h1 id="thuc-hanh">Thực hành</h1>
<h2 id="buoc-1-cai-dat-xiao-steganography">Bước 1: Cài đặt Xiao Steganography</h2>
<p>Sau khi tải phần mềm xong, chúng ta sẽ mở cài đặt và giao diện cài đặt của công cụ sẽ hiện ra giống với hình bên dưới này:</p>
<p><img class="article-img" src="https://raw.githubusercontent.com/minhgiau998/image/develop/2020/10/23/xiao-steganography-01.png#full" alt="xiao-steganography-01.png"></p>
<p>Nhấn <strong>Next</strong> để tiếp tục cài đặt, chúng ta sẽ chọn đường dẫn nơi cài đặt của công cụ.</p>
<p><img class="article-img" src="https://raw.githubusercontent.com/minhgiau998/image/develop/2020/10/23/xiao-steganography-02.png#full" alt="xiao-steganography-02.png"></p>
<p>Nhấn <strong>Install</strong> để bắt đầu cài đặt công cụ. Sau khi cài đặt xong, chúng ta nhấn <strong>Next</strong> để tiếp tục.</p>
<p><img class="article-img" src="https://raw.githubusercontent.com/minhgiau998/image/develop/2020/10/23/xiao-steganography-03.png#full" alt="xiao-steganography-03.png"></p>
<p>Nhấn <strong>Finish</strong> để kết thúc quá trình cài đặt và công cụ sẽ tự động mở ra.</p>
<p><img class="article-img" src="https://raw.githubusercontent.com/minhgiau998/image/develop/2020/10/23/xiao-steganography-04.png#full" alt="xiao-steganography-04.png"></p>
<h2 id="buoc-2-chen-tap-tin-bi-mat-vao-trong-hinh-anh">Bước 2: Chèn tập tin bí mật vào trong hình ảnh</h2>
<p>Giao diện của Xiao Steganography khá đơn giản và chỉ bao gồm 2 chức năng chính: <strong>Add Files</strong> và <strong>Extract Files</strong>.</p>
<p><img class="article-img" src="https://raw.githubusercontent.com/minhgiau998/image/develop/2020/10/23/xiao-steganography-05.png#full" alt="xiao-steganography-05.png"></p>
<p>Chúng ta chọn phần <strong>Add Files</strong>, cửa sổ sẽ hiện ra thông tin của bước đầu tiên là lựa chọn tập tin là phần bao bên ngoài tập tin cần được ẩn đi. Để chọn tập tin chúng ta chọn <strong>Load Target File</strong>.</p>
<p><img class="article-img" src="https://raw.githubusercontent.com/minhgiau998/image/develop/2020/10/23/xiao-steganography-06.png#full" alt="xiao-steganography-06.png"></p>
<p>Sau khi chọn hình xong, chúng ta nhấn <strong>Next</strong> để tiếp tục sang bước 2. Trong bước này, chúng ta sẽ chọn tập tin bí mật mà chúng ta cần ẩn chúng đi bằng cách chọn <strong>Add File</strong>.</p>
<p><img class="article-img" src="https://raw.githubusercontent.com/minhgiau998/image/develop/2020/10/23/xiao-steganography-07.png#full" alt="xiao-steganography-07.png"></p>
<p>Sau khi thêm tập tin xong, chúng ta chọn <strong>Next</strong> để tiến hành bước cuối cùng là chọn thuật toán mã hóa và băm cho tập tin bí mật. Ngoài ra, chúng ta cũng có thể thiết lập mật khẩu cho tập tin bí mật này. Ở trong trường hợp này, mình sẽ chọn thuật toán mã hoá là <strong>DES</strong> và thuật toán băm là <strong>MD5</strong>. Mình sẽ đặt mật khẩu cho tập tin này là <strong><em>Clu6F14</em></strong>.</p>
<p><img class="article-img" src="https://raw.githubusercontent.com/minhgiau998/image/develop/2020/10/23/xiao-steganography-08.png#full" alt="xiao-steganography-08.png"></p>
<p>Chúng ta chọn <strong>Next</strong> để bắt đầu quá trình ẩn tập tin bí mật trong hình ảnh. Sau khi mã hóa xong, chúng ta sẽ đặt tên tập tin mới mà chúng ta cần lưu lại.</p>
<p><img class="article-img" src="https://raw.githubusercontent.com/minhgiau998/image/develop/2020/10/23/xiao-steganography-09.png#full" alt="xiao-steganography-09.png"></p>
<p>Sau khi lưu tập tin xong, chúng ta có thể xem lại bản chính thức của bức ảnh này bằng cách chọn <strong>See final file</strong>. Chúng ta sẽ nhận thấy rằng sự thay đổi này sẽ không hề ảnh hưởng đến hình thái của bức ảnh gốc.</p>
<p><img class="article-img" src="https://raw.githubusercontent.com/minhgiau998/image/develop/2020/10/23/xiao-steganography-10.png#full" alt="xiao-steganography-10.png"></p>
<h2 id="buoc-3-hien-thi-tap-tin-bi-mat-trong-buc-anh">Bước 3: Hiển thị tập tin bí mật trong bức ảnh</h2>
<p>Giả sử chúng ta nhận được bức ảnh này từ bạn bè và họ có gợi ý rằng nên sử dụng công cụ Xiao Steganography để giải mã tập tin bí mật được giấu trong bức ảnh. Chúng ta mở công cụ và chọn phần <strong>Extract Files</strong>. Tương tự như bước 2 nhưng lần này chúng ta sẽ chọn tập tin cần được giải mã bằng cách chọn <strong>Load Source File</strong>.</p>
<p><img class="article-img" src="https://raw.githubusercontent.com/minhgiau998/image/develop/2020/10/23/xiao-steganography-11.png#full" alt="xiao-steganography-11.png"></p>
<p>Chúng ta chọn <strong>Next</strong> để tiếp tục và chọn tài nguyên mà chúng ta cần giải nén ra ngoài. Cần phải có mật khẩu để có quyền truy cập vào tập tin bí mật. Chọn <strong>Extract File</strong> để bắt đầu giải nén tập tin bí mật.</p>
<p><img class="article-img" src="https://raw.githubusercontent.com/minhgiau998/image/develop/2020/10/23/xiao-steganography-12.png#full" alt="xiao-steganography-12.png"></p>
<p>Chúng ta mở ra file vừa giải nén được thì sẽ thấy thông điệp bí mật được ẩn giấu.</p>
<p><img class="article-img" src="https://raw.githubusercontent.com/minhgiau998/image/develop/2020/10/23/xiao-steganography-13.png#full" alt="xiao-steganography-13.png"></p>
<h1 id="ket-luan">Kết luận</h1>
<p>Xiao Steganography là một công cụ cho phép thực hiện việc ẩn giấu các tập tin bí mật dưới các file hình ảnh. Ngoài ra, nó còn hỗ trợ các thuật toán mã hóa và băm mạnh mẽ kèm theo mật khẩu giúp cho việc ẩn tập tin bí mật trở nên dễ dàng hơn bao giờ hết.</p>
