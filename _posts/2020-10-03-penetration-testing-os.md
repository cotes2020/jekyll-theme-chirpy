---
layout: post
title: "Penetration Testing OS"
summary: "What is Penetration Testing OS ?"
author: technical
date: '2020-10-03 09:00:00'
category: RedTeam
thumbnail: assets\img\thumbnail\penetration-testing-os.png
keywords: Security, Penetration Testing
permalink: /blog/Penetration-Testing-os/
usemathjax: true
---

<blockquote>
<p>Nếu bạn là một hacker, người kiểm tra thâm nhập (penetration tester), thợ săn tiền thưởng (bug bounty hunter) hoặc một nhà nghiên cứu bảo mật thì có lẽ bạn đã nghe nói về 2 bản phân phối của Linux này.</p>
</blockquote>
<p>Mình đã sử dụng cả hai hệ điều hành khá thường xuyên cho công việc nghiên cứu và thử nghiệm của mình. Linux có một bộ sưu tập các bản phân phối không đồng nhất và có sẵn trên thị trường. Nhưng bản phân phối nổi tiếng nhất được hầu hết các nhà nghiên cứu bảo mật và kiểm tra thâm nhập sử dụng là <strong>Kali Linux</strong>. Kali đã trải qua nhiều bản cập nhật trong khi một bản phân phối khác liên quan đến an ninh mạng trong kiểm tra thâm nhập cũng đang được phát triển trên khắp thế giới. Trong bài viết này, mình sẽ so sánh giữa Kali Linux với một bản phân phối đã được mọi người chú ý khá nhiều - <strong>ParrotsOS</strong>. Sự so sánh này sẽ giải thích những ưu và nhược điểm khác nhau của cả hai hệ điều hành.</p>
<blockquote>
<p>Kiểm tra thâm nhập không phải là cách mà chúng ta sử dụng công cụ mà nó là những kỹ năng để áp dụng giải quyết vấn đề.</p>
</blockquote>
<h1 id="kali-linux-la-gi">Kali Linux là gì?</h1>
<p><img class="article-img" src="https://raw.githubusercontent.com/minhgiau998/image/develop/2020/10/03/penetration-testing-os-01.png#full" alt="penetration-testing-os-01.png"></p>
<p>Kali Linux là bản phân phối Linux dựa trên <strong>Debian</strong> nhằm mục đích kiểm tra thâm nhập nâng cao và kiểm tra bảo mật. Kali chứa hơn 600 trăm công cụ được cài đặt sẵn nhằm hướng đến các nhiệm vụ an toàn thông tin khác nhau, chẳng hạn như <strong>Kiểm Tra Thâm Nhập</strong> (Penetration Testing), <strong>Nghiên Cứu Bảo Mật</strong> (Security research), <strong>Pháp Y Máy Tính</strong> (Computer Forensics), <strong>Kiểm Thử Ứng Dụng Web</strong> (Web Application Testing) và <strong>Kỹ Thuật Dịch Ngược</strong> (Reverse Engineering). Kali Linux được phát triển và duy trì bởi đội ngũ <strong>Offensive Security</strong> - một công ty đào tạo hàng đầu về bảo mật thông tin. Nó được phát triển như một bản phân phối thử nghiệm và sẽ được dùng để thay thế cho <strong>BacktrackOS</strong>. Nó được phát hành vào ngày 13 tháng 3 năm 2013 như một bản xây dựng lại hoàn chỉnh từ đầu đến cuối của BackTrack Linux, tuân thủ hoàn toàn các tiêu chuẩn phát triển Debian.</p>
<h1 id="tinh-nang">Tính năng</h1>
<h2 id="hon-600-cong-cu-kiem-tra-tham-nhap">Hơn 600 công cụ kiểm tra thâm nhập</h2>
<p>Kali đi kèm với các công cụ kiểm tra thâm nhập khác nhau từ bản cài đặt. Sau khi xem xét mọi công cụ có trong BackTrack, nó đã loại bỏ một số lượng lớn các công cụ đơn giản là không hoạt động hoặc sao chép các công cụ khác đã cung cấp chức năng giống nhau hoặc tương tự.</p>
<h2 id="ho-tro-da-ngon-ngu">Hỗ trợ đa ngôn ngữ</h2>
<p>Mặc dù các công cụ kiểm tra thâm nhập có xu hướng được viết bằng tiếng Anh. Nhưng để cải thiện việc sử dụng của người dùng tiếng Anh không phải là bản ngữ, Kali đã bao gồm hỗ trợ đa ngôn ngữ thực sự và cho phép nhiều người dùng hoạt động bằng ngôn ngữ mẹ đẻ của họ và xác định vị trí các công cụ họ cần cho công việc.</p>
<h2 id="hoan-toan-co-the-tuy-chinh">Hoàn toàn có thể tùy chỉnh</h2>
<p>Thiết kế ban đầu của Kali Linux không đạt tiêu chuẩn vì giao diện không đẹp. Và để tránh vấn đề đó, Kali đã giúp những người dùng ưa thích sự thay đổi hơn có thể tùy chỉnh Kali Linux theo ý thích của họ dễ dàng nhất có thể và tất cả đều áp dụng xuống kernel của Linux.</p>
<h2 id="ho-tro-thiet-bi-khong-day-tren-pham-vi-rong">Hỗ trợ thiết bị không dây trên phạm vi rộng</h2>
<p>Một điểm hay gắn bó với các bản phân phối Linux đã được hỗ trợ cho các giao diện không dây. Kali Linux hỗ trợ nhiều thiết bị không dây nhất có thể, cho phép nó chạy đúng cách trên nhiều loại phần cứng và làm cho nó tương thích với nhiều USB và các thiết bị không dây khác.</p>
<h2 id="kernel-co-the-tuy-chinh-va-luon-dc-cap-nhat-nhung-ban-va-loi">Kernel có thể tùy chỉnh và luôn được cập nhật những bản vá lỗi</h2>
<p>Là người kiểm tra thâm nhập, chúng ta thường cần thực hiện đánh giá và kiểm tra không dây, vì vậy kernel của chúng ta nên có các bản vá lỗi mới nhất.</p>
<h2 id="va-cuoi-cung-la-mien-phi">Và cuối cùng là MIỄN PHÍ</h2>
<p>Kali Linux, nó được sử dụng miễn phí như BackTrack, chúng ta sẽ không bao giờ cần phải trả tiền cho Kali Linux.</p>
<h1 id="parrot-os-la-gi">Parrot OS là gì ?</h1>
<p><img class="article-img" src="https://raw.githubusercontent.com/minhgiau998/image/develop/2020/10/03/penetration-testing-os-02.png#full" alt="penetration-testing-os-02.png"></p>
<p>Parrot Linux (Parrot Security, parrot OS, Parrot GNU / Linux) là bản phân phối GNU / Linux mã nguồn mở và miễn phí dựa trên phiên bản Debian được thiết kế cho các chuyên gia bảo mật, nhà phát triển và những người có hiểu biết về quyền riêng tư. Khi chúng ta nó nói dựa trên Debian, điều đó có nghĩa là các đoạn code trong Parrot OS sẽ luôn tuân theo các tiêu chuẩn phát triển Debian. Nó bao gồm một kho các công cụ đầy đủ cho các hoạt động bảo mật CNTT và pháp y kỹ thuật số, nhưng nó cũng bao gồm mọi thứ chúng ta cần để phát triển các chương trình của riêng mình hoặc bảo vệ quyền riêng tư của chúng ta khi tham gia vào môi trường mạng. Hệ điều hành này đi kèm với giao diện MATE được cài đặt sẵn và có nhiều phiên bản để phù hợp với nhu cầu của bạn.</p>
<blockquote>
<p>Hệ điều hành Parrot được phát hành lần đầu tiên vào năm 2013 và được phát triển bởi một nhóm các chuyên gia bảo mật, những người đam mê Linux và các nhà phát triển mã nguồn mở. Đội ngũ này do <strong>Lorenzo Faletra</strong> làm trưởng nhóm.</p>
</blockquote>
<h1 id="tinh-nang">Tính năng</h1>
<h2 id="bao-mat">Bảo mật</h2>
<p>Nó luôn được cập nhật, phát hành thường xuyên và được đóng gói đầy đủ! Mọi thứ đều nằm trong tầm kiểm soát hoàn toàn của chúng ta.</p>
<h2 id="mien-phi">Miễn phí</h2>
<p>Nó miễn phí và mã nguồn mở, chúng ta có thể xem mã nguồn và tùy chỉnh nó theo yêu cầu của chúng ta.</p>
<h2 id="nhe">Nhẹ</h2>
<p>Hệ điều hành này đã được chứng minh là cực kỳ nhẹ và chạy rất nhanh, ngay cả trên phần cứng rất cũ hoặc với các tài nguyên rất hạn chế.</p>
<h1 id="kali-linux-vs-parrot-os">Kali Linux vs Parrot OS</h1>
<p>Cả hai hệ điều hành này đều nhằm mục đích giống nhau, tức là kiểm tra khả năng thâm nhập và bảo mật không gian mạng. Hầu hết các yếu tố trong những trường hợp như vậy tập trung vào vấn đề sở thích cá nhân hơn là so sánh khách quan. Bây giờ, trước khi chúng ta bắt đầu so sánh Parrot OS và Kali Linux, hãy để mình liệt kê những điểm tương đồng giữa hai hệ điều hành.</p>
<h2 id="giong-nhau">Giống nhau</h2>
<ul>
<li>Cả hai hệ điều hành đều được tùy chỉnh để phù hợp với mục đích kiểm tra thâm nhập.</li>
<li>Cả hai hệ điều hành đều hỗ trợ kiến ​​trúc 32bit và 64bit.</li>
<li>Cả hai hệ điều hành đều hỗ trợ VPN đám mây.</li>
<li>Dựa trên các tiêu chuẩn phát triển Debian.</li>
<li>Đi kèm với kho công cụ hacking đã được cài đặt sẵn.</li>
<li>Cả hai hệ điều hành đều hỗ trợ cho các thiết bị nhúng và IoT.</li>
</ul>
<h2 id="khac-nhau">Khác nhau</h2>
<h3 id="phan-cung">Phần cứng</h3>
<table>
<thead>
<tr>
<th>Parrot OS</th>
<th>Kali Linux</th>
</tr>
</thead>
<tbody>
<tr>
<td>Không cần tăng tốc đồ họa.</td>
<td>Yêu cầu tăng tốc đồ họa.</td>
</tr>
<tr>
<td>Yêu cầu RAM tối thiểu 320mb.</td>
<td>Yêu cầu RAM tối thiểu 1GB.</td>
</tr>
<tr>
<td>Yêu cầu CPU lõi kép 1GHZ tối thiểu.</td>
<td>Yêu cầu CPU lõi kép 1GHZ tối thiểu.</td>
</tr>
<tr>
<td>Nó cũng có thể khởi động ở chế độ Legacy và UEFI.</td>
<td>Nó cũng có thể khởi động ở chế độ Legacy và UEFI.</td>
</tr>
<tr>
<td>Cần ít nhất 16GB dung lượng đĩa cứng để cài đặt hệ điều hành.</td>
<td>Cần ít nhất 20GB dung lượng đĩa cứng để cài đặt hệ điều hành.</td>
</tr>
</tbody>
</table>
<p>Yêu cầu phần cứng là thứ mà chúng ta bỏ qua hầu hết thời gian, chủ yếu vì chúng ta biết rằng hệ thống của chúng ta mạnh hơn nhiều so với các máy tính yêu cầu phần cứng tối thiểu. Nhưng Parrot cần phần cứng thông số kỹ thuật thấp hơn khi so sánh với Kali, có nghĩa là nó có thể chạy trên máy tính xách tay.</p>
<p>Đây là một trong những lý do tại sao mình thích Parrot OS hơn Kali Linux, nhưng mình thích và sử dụng cả hai và như mình đã nói ở trên đầu bài viết, không quan trọng hệ điều hành hay công cụ nào mà bạn đang sử dụng, tất cả phụ thuộc vào kỹ năng của bạn.</p>
<h3 id="giao-dien">Giao diện</h3>
<p>Khi nói đến giao diện của một hệ điều hành, mình chủ yếu thích hệ điều hành Parrot OS hơn Kali Linux vì:</p>
<p>Giao diện của Parrot OS được xây dựng giống với phong cách của hệ điều hành Ubuntu (Mate). Ở trên cùng, bạn thấy một ngăn chứa Ứng dụng, Địa điểm, Hệ thống giống như Kali. Hệ điều hành Parrot cũng cung cấp một số thông tin thú vị về nhiệt độ CPU cùng với biểu đồ sử dụng. Ngăn dưới cùng chứa trình quản lý menu và trình quản lý máy trạm.</p>
<p><img class="article-img" src="https://raw.githubusercontent.com/minhgiau998/image/develop/2020/10/03/penetration-testing-os-03.png#full" alt="penetration-testing-os-03.png"></p>
<p>Mặt khác, Kali Linux tuân theo giao diện GNOME. Theo ý kiến ​​của mình, nó có chức năng tương tự như hệ điều hành Parrot OS nhưng không mang lại vẻ ngoài gọn gàng, tinh tế.</p>
<p><img class="article-img" src="https://raw.githubusercontent.com/minhgiau998/image/develop/2020/10/03/penetration-testing-os-04.png#full" alt="penetration-testing-os-04.png"></p>
<h3 id="cong-cu-hacking">Công cụ Hacking</h3>
<p>Khi nói đến các công cụ và tính năng chung, Parrot OS thắng Kali Linux. Hệ điều hành Parrot có tất cả các công cụ có sẵn trong Kali Linux và cũng bổ sung các công cụ của riêng nó. Có một số công cụ bạn sẽ tìm thấy trên Parrot OS mà không có trên Kali Linux. Hãy thảo luận về một vài công cụ được liệt kê dưới đây:</p>
<h4 id="anonsurf">AnonSurf</h4>
<blockquote>
<p>Ẩn danh đối với một hacker là bước đầu tiên trước khi hack một hệ thống.</p>
</blockquote>
<p>Chắc hẵn bạn đã từng nghe thấy câu nói trên, mỗi hacker đều thay đổi nhân dạng của mình vào lúc nửa đêm.
Mình không muốn đi quá sâu vào công cụ này nhưng Parrot đã cài đặt và cấu hình sẵn Anonsurf, vì vậy nếu bạn đang làm điều gì đó lén lút và muốn ẩn danh, bạn có thể che giấu bản thân chỉ bằng một cú nhấp chuột.</p>
<h4 id="wifiphisher">Wifiphisher</h4>
<p>Đây là một framework Access Point giả mạo được sử dụng để tiến hành kiểm tra bảo mật của Wi-Fi. Sử dụng Wifiphisher, người kiểm tra thâm nhập có thể dễ dàng trở thành vai trò trung gian trong <strong>man-in-the-middle</strong> để nhắm vào các máy khách không dây bằng cách thực hiện các cuộc tấn công vào Wi-Fi mục tiêu. Wifiphisher có thể được sử dụng để thay đổi trang web giả mạo và dẫn dụ các máy khách được kết nối nhằm lấy thông tin xác thực hoặc lây nhiễm phần mềm độc hại cho máy nạn nhân.</p>
<h3 id="cac-phien-ban">Các phiên bản</h3>
<p>Cả hai hệ điều hành đều có nhiều phiên bản, nhưng Parrot OS có nhiều hơn thế về sự đa dạng.</p>
<p><em>Parrot OS</em></p>
<ul>
<li>Kali Full Edition</li>
<li>Kali Lite Edition</li>
<li>Kali armhf/armel (IoT devices)</li>
<li>Kali Desktop Variation (e17/KDE/Xfce)</li>
</ul>
<p><em>Kali Linux</em></p>
<ul>
<li>Parrot Sec OS Full Edition</li>
<li>Parrot Sec OS Lite Edition</li>
<li>Parrot Sec OS Studio Edition</li>
<li>Parrot Sec OS Air Edition</li>
</ul>
<p>Như chúng ta thấy, Parrot có một số tính năng đa dạng với một bản phát hành tập trung vào kiểm tra thâm nhập không dây (AIR) và một bản được điều chỉnh để tạo nội dung đa phương tiện (studio). Ngoài ra, nó cũng có các bản phát hành có hỗ trợ đám mây và hỗ trợ cho các thiết bị IoT. Kali cung cấp phiên bản đầy đủ và nhẹ cơ bản cùng với các giao diện máy tính để bàn tùy chỉnh (e17 / KDE / Matter / LXDE). Kali cũng có hỗ trợ cho các thiết bị IoT và đám mây.</p>
<h3 id="hieu-nang">Hiệu năng</h3>
<p>Khi chúng ta nói về hiệu suất, Kali có lẽ hơi chậm khi bạn chạy nó trên một hệ thống cấp thấp. Đôi khi đó là một cơn ác mộng khi bạn có một cuộc tấn công mạng đang diễn ra trong nền và bạn đang làm việc khác. Nhưng Parrot nó rất nhẹ và không bị lag nhiều. vì nó cũng chạy trên các hệ thống thông số kỹ thuật thấp.</p>
<h1 id="ket-luan">Kết luận</h1>
<p>Mình hy vọng bạn đã hiểu rõ về hệ điều hành bảo mật Parrot OS và Kali Linux. Chúng ta đã thảo luận khá nhiều về mọi thứ về cả hai hệ điều hành một cách chi tiết. Nhưng lựa chọn hệ điều hành dựa trên sở thích và sự lựa chọn của bạn, nếu bạn có hệ thống thông số kỹ thuật thấp, mình khuyên bạn nên sử dụng hệ điều hành Parrot OS.</p>
