---
layout: post
title: "Penetration Testing"
summary: "What is Penetration Testing ?"
author: technical
date: '2020-10-16 09:00:00'
category: RedTeam
thumbnail: assets/img/thumbnail/penetration-testing.png
keywords: Security, Penetration Testing, Hacking
permalink: /blog/Penetration-Testing/
usemathjax: true
---

<p>Penetration Testing (Kiểm Tra Thâm Nhập) là quy trình xác định các lỗ hổng bảo mật trong ứng dụng bằng cách đánh giá hệ thống hoặc mạng bằng các kỹ thuật khác nhau. Các điểm yếu của một hệ thống được khai thác trong quá trình này đều thông qua một cuộc tấn công mô phỏng (được sự cho phép của các tổ chức).</p>
<p>Mục đích của sự kiểm tra này là để bảo mật dữ liệu quan trọng khỏi những kẻ bên ngoài như hacker để có thể truy cập trái phép vào hệ thống. Khi lỗ hổng đã được xác định, nó sẽ được sử dụng để khai thác hệ thống nhằm truy cập các thông tin nhạy cảm.</p>
<p>Kiểm tra thâm nhập còn được gọi là <strong>pen test</strong> và người kiểm tra thâm nhập còn được gọi là <strong>penetration tester</strong>.</p>
<p>Người kiểm tra thâm nhập có thể tìm ra các lỗ hổng của hệ thống máy tính, ứng dụng web hoặc mạng thông qua kiểm tra thâm nhập.</p>
<p>Kiểm tra thâm nhập còn cho chúng ta biết liệu các biện pháp phòng thủ đang được sử dụng trên hệ thống có đủ mạnh để ngăn chặn bất kỳ hành vi vi phạm bảo mật nào hay không. Các báo cáo kiểm tra thâm nhập cũng đề xuất các biện pháp đối phó có thể được thực hiện để giảm thiểu nguy cơ hệ thống bị tấn công.</p>
<h1 id="nguyen-nhan-cua-cac-lo-hong">Nguyên nhân của các lỗ hổng</h1>
<ul>
<li><strong>Lỗi thiết kế và phát triển</strong>: Có thể có sai sót trong quá trình thiết kế phần cứng và phần mềm. Những lỗi này có thể khiến dữ liệu quan trọng trong doanh nghiệp của bạn có nguy cơ bị lộ.</li>
<li><strong>Cấu hình hệ thống kém</strong>: Đây là một nguyên nhân khác của lỗ hổng bảo mật. Nếu hệ thống được cấu hình kém, thì nó có thể tạo ra các kẽ hở mà qua đó kẻ tấn công có thể xâm nhập vào hệ thống và đánh cắp thông tin.</li>
<li><strong>Con người</strong>: Các yếu tố con người như xử lý tài liệu không đúng cách, để tài liệu không cần giám sát, lỗi mã hóa, mối đe dọa từ nội bộ, chia sẻ mật khẩu qua các trang web lừa đảo, v.v. có thể dẫn đến vi phạm bảo mật.</li>
<li><strong>Khả năng kết nối</strong>: Nếu hệ thống được kết nối với một mạng không an toàn (kết nối mở) thì nó sẽ nằm trong tầm ngắm của các hacker.</li>
<li><strong>Độ phức tạp</strong>: Lỗ hổng bảo mật tăng lên tương ứng với mức độ phức tạp của hệ thống. Hệ thống càng có nhiều tính năng thì càng có nhiều khả năng hệ thống bị tấn công.</li>
<li><strong>Mật khẩu</strong>: Mật khẩu được sử dụng để ngăn chặn truy cập trái phép. Chúng phải đủ mạnh để không ai có thể đoán được mật khẩu của bạn. Mật khẩu không được chia sẻ với bất kỳ ai bằng bất kỳ giá nào và mật khẩu nên được thay đổi định kỳ. Bất chấp những hướng dẫn này, đôi khi mọi người tiết lộ mật khẩu của họ cho người khác, hãy ghi chúng vào đâu đó và giữ những mật khẩu dễ đoán.</li>
<li><strong>Đầu vào của người dùng</strong>: Chắc hẳn bạn đã nghe nói về <strong>SQL injection</strong>, <strong>Buffer Overflow</strong>, v.v. Dữ liệu nhận được dưới dạng điện tử thông qua các phương thức này có thể được sử dụng để tấn công hệ thống.</li>
<li><strong>Quản lý</strong>: Quản lý bảo mật rất khó khăn và tốn kém. Đôi khi các tổ chức thiếu kinh nghiệm trong việc quản lý rủi ro và do đó lỗ hổng bảo mật được tạo ra bên trong hệ thống.</li>
<li><strong>Nhân viên chưa có nhận thức về bảo mật</strong>: Điều này dẫn đến sai sót của con người và các lỗ hổng khác.</li>
<li><strong>Truyền thông</strong>: Các kênh như mạng di động, internet, điện thoại mở ra phạm vi trộm cắp dữ liệu.</li>
</ul>
<h1 id="tai-sao-phai-kiem-tra-tham-nhap">Tại sao phải Kiểm Tra Thâm Nhập?</h1>
<p>Bạn chắc hẳn đã nghe nói về cuộc tấn công ransomware <strong>WannaCry</strong> bắt đầu vào tháng 5 năm 2017. Nó đã khóa hơn 2 vạn máy tính trên khắp thế giới và yêu cầu thanh toán tiền chuộc bằng tiền điện tử <strong>Bitcoin</strong>. Cuộc tấn công này đã ảnh hưởng đến nhiều tổ chức lớn trên toàn cầu.</p>
<p>Với những cuộc tấn công mạng lớn và nguy hiểm đang diễn ra ngày nay, việc kiểm tra thâm nhập thường xuyên là điều cần thiết để bảo vệ hệ thống thông tin chống lại các vi phạm bảo mật.</p>
<p>Dưới đây là những lý do chính mà Kiểm Tra Thâm Nhập là yêu cầu bắt buộc:</p>
<ul>
<li>Dữ liệu tài chính hoặc dữ liệu quan trọng phải được bảo mật trong khi chuyển dữ liệu giữa các hệ thống khác nhau hoặc qua mạng.</li>
<li>Nhiều khách hàng đang yêu cầu thực hiện kiểm tra thâm nhập như một phần của chu kỳ phát hành phần mềm.</li>
<li>Để bảo mật dữ liệu người dùng.</li>
<li>Để tìm các lỗ hổng bảo mật trong một ứng dụng.</li>
<li>Để khám phá các nguy cơ bảo mật trong hệ thống.</li>
<li>Để đánh giá tác động kinh doanh khi bị ảnh hưởng bởi các cuộc tấn công mạng.</li>
<li>Để đáp ứng việc tuân thủ bảo mật thông tin trong tổ chức.</li>
<li>Để thực hiện chiến lược bảo mật hiệu quả trong tổ chức.</li>
</ul>
<p>Việc xác định các vấn đề bảo mật trong mạng nội bộ và máy tính là rất quan trọng đối với bất kỳ tổ chức nào. Sử dụng những thông tin này có thể lập kế hoạch phòng thủ chống lại bất kỳ nỗ lực tấn công nào. Quyền riêng tư của người dùng và bảo mật dữ liệu là mối quan tâm lớn nhất hiện nay.</p>
<h1 id="cai-gi-can-dc-kiem-tra">Cái gì cần được kiểm tra?</h1>
<ul>
<li>Phần mềm (Hệ điều hành, dịch vụ, ứng dụng)</li>
<li>Phần cứng</li>
<li>Mạng</li>
<li>Các tiến trình</li>
<li>Hành vi người dùng cuối</li>
</ul>
<h1 id="cac-loai-kiem-tra-tham-nhap">Các loại Kiểm Tra Thâm Nhập</h1>
<h2 id="social-engineering-test">Social Engineering Test</h2>
<p>Trong phương pháp kiểm tra này, các nỗ lực được thực hiện để khiến một người tiết lộ thông tin nhạy cảm như mật khẩu, dữ liệu quan trọng trong kinh doanh, v.v. Những kiểm tra này chủ yếu được thực hiện thông qua điện thoại hoặc internet và nó nhắm mục tiêu vào một số bộ phận như hỗ trợ, nhân viên và quy trình nhất định.</p>
<p>Lỗi của con người là nguyên nhân chính gây ra lỗ hổng bảo mật. Các tiêu chuẩn và chính sách bảo mật nên được tất cả nhân viên tuân theo để tránh nỗ lực xâm nhập của phương thức kiểm tra này. Ví dụ về các tiêu chuẩn này bao gồm không đề cập đến bất kỳ thông tin nhạy cảm nào trong giao tiếp qua email hoặc điện thoại. Đánh giá bảo mật có thể được tiến hành để xác định và sửa chữa các sai sót trong quy trình làm việc.</p>
<p><img class="article-img" src="https://raw.githubusercontent.com/minhgiau998/image/develop/2020/10/16/penetration-testing-01.png#full" alt="penetration-testing-01.png"></p>
<h2 id="web-application-test">Web Application Test</h2>
<p>Sử dụng các phần mềm Kiểm Tra Thâm Nhập tự động, người ta có thể xác minh xem ứng dụng có bị lộ các lỗ hổng bảo mật hay không. Nó kiểm tra lỗ hổng bảo mật của các ứng dụng web và chương trình phần mềm được đặt trong môi trường của mục tiêu.</p>
<p><img class="article-img" src="https://raw.githubusercontent.com/minhgiau998/image/develop/2020/10/16/penetration-testing-02.png#full" alt="penetration-testing-02.png"></p>
<h2 id="physical-penetration-test">Physical Penetration Test</h2>
<p>Các phương pháp bảo mật vật lý được áp dụng để bảo vệ dữ liệu nhạy cảm. Phương thức này thường được sử dụng trong các cơ sở quân sự và chính phủ. Tất cả các thiết bị mạng vật lý và điểm truy cập đều được kiểm tra khả năng xảy ra bất kỳ vi phạm bảo mật nào. Phương thức kiểm tra này không liên quan nhiều đến phạm vi kiểm tra phần mềm.</p>
<p><img class="article-img" src="https://raw.githubusercontent.com/minhgiau998/image/develop/2020/10/16/penetration-testing-03.png#full" alt="penetration-testing-03.png"></p>
<h2 id="network-services-test">Network Services Test</h2>
<p>Đây là một trong những phương thức Kiểm Tra Thâm Nhập được thực hiện phổ biến nhất, trong đó các lỗ hổng trong mạng được xác định bằng thành phần đang được thực hiện trong các hệ thống trên mạng để kiểm tra xem có loại lỗ hổng nào. Nó có thể được thực hiện tại địa phương hoặc từ xa.</p>
<p><img class="article-img" src="https://raw.githubusercontent.com/minhgiau998/image/develop/2020/10/16/penetration-testing-04.png#full" alt="penetration-testing-04.png"></p>
<h2 id="client-side-test">Client-side Test</h2>
<p>Mục đích của kiểm tra này là tìm kiếm và khai thác các lỗ hổng trong các phần mềm phía máy khách.</p>
<p><img class="article-img" src="https://raw.githubusercontent.com/minhgiau998/image/develop/2020/10/16/penetration-testing-05.png#full" alt="penetration-testing-05.png"></p>
<h2 id="remote-dial-up-war-dial">Remote dial-up war dial</h2>
<p>Các hacker tìm kiếm các modem trong môi trường xung quanh và cố gắng đăng nhập vào các hệ thống được kết nối thông qua các modem này bằng cách đoán mật khẩu hoặc sử dụng kỹ thuật &quot;vét cạn&quot;.</p>
<p><img class="article-img" src="https://raw.githubusercontent.com/minhgiau998/image/develop/2020/10/16/penetration-testing-06.png#full" alt="penetration-testing-06.png"></p>
<h2 id="wireless-security-test">Wireless security test</h2>
<p>Các hacker khám phá các điểm phát sóng hoặc mạng Wi-Fi công cộng và kết nối chúng để thực hiện các hành vi vi phạm bảo mật.</p>
<p><img class="article-img" src="https://raw.githubusercontent.com/minhgiau998/image/develop/2020/10/16/penetration-testing-07.png#full" alt="penetration-testing-07.png"></p>
<p>Ngoài 7 loại kiểm tra mà chúng ta đã nói ở trên, chúng ta có thể phân loại pen test thành 3 phần như sau:</p>
<p><img class="article-img" src="https://raw.githubusercontent.com/minhgiau998/image/develop/2020/10/16/penetration-testing-08.png#full" alt="penetration-testing-08.png"></p>
<p>Chúng ta hãy cùng thảo luận từng phương pháp kiểm tra này:</p>
<ul>
<li><strong>Black Box Penetration Testing</strong>: Người kiểm tra sẽ đánh giá hệ thống, mạng hoặc quy trình của mục tiêu mà không cần biết về các chi tiết bên trong. Họ chỉ có những thông tin cơ bản như URL hoặc tên công ty mà họ sử dụng để thâm nhập vào môi trường mục tiêu. Không có đoạn code nào được kiểm tra với phương thức kiểm tra này.</li>
<li><strong>White Box Penetration Testing</strong>: Người kiểm tra được trang bị đầy đủ thông tin chi tiết về môi trường của mục tiêu như hệ thống, mạng, hệ điều hành, địa chỉ IP, mã nguồn, lược đồ, v.v. Điều đó sẽ giúp cho người kiểm tra dễ dàng kiểm tra code và tìm ra lỗi thiết kế &amp; phát triển phần mềm hoặc phần cứng. Phương pháp kiểm tra này được xem như một sự mô phỏng của cuộc tấn công bảo mật trong nội bộ.</li>
<li><strong>Grey Box Penetration Testing</strong>: Người kiểm tra sẽ bị giới hạn chi tiết về môi trường của mục tiêu. Phương pháp kiểm tra này được xem như một sự mô phỏng của cuộc tấn công bảo mật bên ngoài.</li>
</ul>
<p><img class="article-img" src="https://raw.githubusercontent.com/minhgiau998/image/develop/2020/10/16/penetration-testing-09.png#full" alt="penetration-testing-09.png"></p>
<h1 id="cac-ky-thuat-kiem-tra-tham-nhap">Các kỹ thuật Kiểm Tra Thâm Nhập</h1>
<p>Trong Kiểm Tra Thâm Nhập sẽ có 3 kỹ thuật chính được liệt kê dưới đây:</p>
<ul>
<li>Kiểm Tra Thâm Nhập thủ công</li>
<li>Sử dụng các côn cụ kiểm tra thâm nhập tự động</li>
<li>Kết hợp cả 2 phương pháp trên</li>
</ul>
<p>Kỹ thuật thứ 3 sẽ được sử dụng thường hơn để nhận dạng các loại lỗ hổng khác nhau.</p>
<h1 id="cac-cong-cu-kiem-tra-tham-nhap">Các công cụ Kiểm Tra Thâm Nhập</h1>
<p>Các công cụ tự động có thể được sử dụng để xác định một số lỗ hổng tiêu chuẩn có trong ứng dụng. Các công cụ này sẽ quét code để kiểm tra xem có mã độc nào có thể dẫn đến vi phạm bảo mật tiềm ẩn hay không. Các công cụ này có thể xác minh các lỗ hổng bảo mật có trong hệ thống bằng cách kiểm tra các kỹ thuật mã hóa dữ liệu và tìm ra các giá trị được mã hóa cứng như tên người dùng và mật khẩu.</p>
<h1 id="cac-tieu-chi-lua-chon-cac-cong-cu-kiem-tra-tham-nhap">Các tiêu chí để lựa chọn các công cụ Kiểm Tra Thâm Nhập</h1>
<ul>
<li>Dễ dàng triển khai, cấu hình và sử dụng.</li>
<li>Quét hệ thống của bạn một cách dễ dàng.</li>
<li>Phân loại các lỗ hổng dựa trên mức độ nghiêm trọng cần sửa chữa ngay lập tức.</li>
<li>Có thể tự động xác minh các lỗ hổng.</li>
<li>Xác minh lại các khai thác được tìm thấy trước đó.</li>
<li>Có thể tạo ra các báo cáo và nhật ký lỗ hổng chi tiết.</li>
</ul>
<p>Dưới đây là một số công cụ Kiểm Tra Thâm Nhập được sử dụng phổ biến (bao gồm miễn phí và có tính phí):</p>
<ul>
<li><a href="https://www.netsparker.com/">Netsparker</a></li>
<li><a href="https://www.acunetix.com/">Acunetix</a></li>
<li><a href="https://s4applications.uk/">Core Impact</a></li>
<li><a href="https://www.hackerone.com/">Hackerone</a></li>
<li><a href="https://www.intruder.io/">Intruder</a></li>
<li><a href="https://www.indusface.com/">Indusface</a></li>
<li><a href="https://www.breachlock.com/">BreachLock</a></li>
<li><a href="https://www.metasploit.com/">Metasploit</a></li>
<li><a href="https://www.wireshark.org/">Wireshark</a></li>
<li><a href="http://w3af.org/">w3af</a></li>
<li><a href="https://nmap.org/">Nmap</a></li>
<li><a href="https://www.tenable.com/products/nessus">Nessus</a></li>
<li><a href="https://portswigger.net/burp">Burpsuite</a></li>
<li><a href="https://cain-abel.en.softonic.com/">Cain &amp; Abel</a></li>
<li><a href="https://www.zaproxy.org/">Zed Attack Proxy (ZAP)</a></li>
<li><a href="https://www.openwall.com/john/">John The Ripper</a></li>
<li><a href="https://www.beyondtrust.com/">Retina</a></li>
<li><a href="http://sqlmap.org/">Sqlmap</a></li>
<li><a href="http://www.immunitysec.com/products/canvas/index.html">Canvas</a></li>
<li><a href="https://www.social-engineer.org/framework/se-tools/">Social-Engineer Toolkit (SET)</a></li>
<li><a href="https://beefproject.com/">BeEF</a></li>
<li><a href="http://dradisframework.org/">Dradis</a></li>
</ul>
<p>Các dịch vụ pen test có tính phí:</p>
<ul>
<li><a href="https://securitybox.vn/">SecurityBox</a></li>
<li><a href="https://giaiphapviettel.vn/dich-vu/88/penetration-testing-service--dich-vu-kiem-tra-danh-gia-an-toan-thong-tin-mang.html">Viettel</a></li>
<li><a href="https://cystack.net/vi/services/pentest?utm_medium=ppc&amp;utm_source=gsa&amp;utm_campaign=dvpentest&amp;gclid=CjwKCAjwrKr8BRB_EiwA7eFapqbbFgjZk4CDAvxngy2tZSPFuVYNLsumB8nXqEdx-C-S1SeRGV8sOxoCCdwQAvD_BwE">CyStack</a></li>
<li><a href="https://www.rapid7.com/services/security-consulting/penetration-testing-services/">Rapid7</a></li>
</ul>
<h1 id="kiem-tra-tham-nhap-thu-cong">Kiểm Tra Thâm Nhập thủ công</h1>
<p>Rất khó để có thể tìm thấy tất cả các lỗ hổng bảo mật bằng các công cụ tự động. Có một số lỗ hổng chỉ có thể được xác định bằng cách quét thủ công. Người kiểm tra thâm nhập có thể thực hiện các cuộc tấn công tốt hơn vào ứng dụng dựa trên kỹ năng và kiến ​​thức của họ về hệ thống bị xâm nhập. Các phương pháp như social engineering chỉ có thể được thực hiện bởi con người. Kiểm Tra Thâm Nhập thủ công bao gồm thiết kế, logic nghiệp vụ cũng như xác minh các đoạn code.</p>
<h1 id="qua-trinh-thuc-hien-kiem-tra-tham-nhap-thu-cong">Quá trình thực hiện Kiểm Tra Thâm Nhập thủ công</h1>
<ul>
<li><strong>Data collection</strong>: Nhiều phương pháp khác nhau bao gồm việc tìm kiếm bằng Google được sử dụng để lấy dữ liệu hệ thống của mục tiêu. Chúng ta cũng có thể sử dụng kỹ thuật phân tích code trang web để biết thêm thông tin về hệ thống, phần mềm và các phiên bản plugin. Có rất nhiều công cụ và dịch vụ miễn phí có sẵn trên thị trường có thể cung cấp cho bạn thông tin như cơ sở dữ liệu hoặc tên bảng, phiên bản database, phiên bản phần mềm, phần cứng được sử dụng và các plugin bên thứ ba khác nhau được sử dụng trong hệ thống của mục tiêu.</li>
<li><strong>Vulnerability Assessment</strong>: Dựa trên dữ liệu thu thập được trong bước đầu tiên, người ta có thể tìm ra điểm yếu bảo mật trong hệ thống của mục tiêu. Điều này giúp người kiểm tra thâm nhập có thể khởi động các cuộc tấn công bằng cách sử dụng các điểm yếu đã được xác định trong hệ thống.</li>
<li><strong>Actual Exploit</strong>: Đây là bước quan trọng nhất trong quá trình này. Nó đòi hỏi các kỹ năng và kỹ thuật đặc biệt để phát động một cuộc tấn công vào hệ thống của mục tiêu. Những người kiểm tra thâm nhập có kinh nghiệm có thể sử dụng kỹ năng của họ để khởi động một cuộc tấn công vào hệ thống.</li>
<li><strong>Result analysis and report preparation</strong>: Sau khi hoàn thành quá trình Kiểm Tra Thâm Nhập, các báo cáo chi tiết sẽ được chuẩn bị để thực hiện các hành động khắc phục. Tất cả các lỗ hổng đã được xác định và các phương pháp khắc phục được khuyến nghị sẽ được liệt kê trong các báo cáo này. Bạn có thể tùy chỉnh định dạng báo cáo lỗ hổng bảo mật (HTML, XML, MS Word hoặc PDF) theo yêu cầu từ tổ chức của bạn.</li>
</ul>
<p><img class="article-img" src="https://raw.githubusercontent.com/minhgiau998/image/develop/2020/10/16/penetration-testing-10.png#full" alt="penetration-testing-10.png"></p>
<h1 id="mot-so-vi-du-ve-cac-truong-hop-kiem-tra-tham-nhap">Một số ví dụ về các trường hợp Kiểm Tra Thâm Nhập:</h1>
<p>Các bạn hãy nhớ rằng đây không phải là các phương thức bắt buộc. Trong Pentest, mục tiêu của bạn là tìm ra các lỗ hổng bảo mật trong hệ thống. Dưới đây là một số trường hợp Kiểm Tra Thâm Nhập chung và không nhất thiết phải áp dụng cho tất cả các ứng dụng.</p>
<ol>
<li>Kiểm tra xem ứng dụng web có thể xác định các vụ tấn công spam trên các contact forms được sử dụng trên trang web.</li>
<li>Máy chủ Proxy - Kiểm tra xem lưu lượng truy cập mạng do các thiết bị proxy theo dõi. Máy chủ Proxy gây khó khăn cho các hacker để có được các thông tin chi tiết nội bộ của mạng do đó bảo vệ hệ thống khỏi các cuộc tấn công bên ngoài.</li>
<li>Hệ thống lọc spam email - Xác minh nếu lưu lượng truy cập email trong và ngoài mạng được lọc và không bị chặn. Nhiều ứng dụng email đi kèm với bộ lọc spam được tích hợp sẵn cần được cấu hình theo nhu cầu của bạn. Các quy tắc cấu hình này có thể được áp dụng cho các tiêu đề email, chủ đề hoặc nội dung.</li>
<li>Tường lửa – Đảm bảo toàn bộ mạng hoặc máy tính được bảo vệ bằng tường lửa. Tường lửa có thể là phần mềm hoặc phần cứng để chặn truy cập trái phép vào hệ thống. Tường lửa có thể ngăn việc gửi dữ liệu bên ngoài mạng mà không được phép.</li>
<li>Cố gắng khai thác lỗi ở tất cả các máy chủ, hệ thống máy tính để bàn, máy in và thiết bị mạng.</li>
<li>Kiểm tra rằng tất cả tên người dùng và mật khẩu được mã hóa và chuyển qua kết nối bảo mật như https.</li>
<li>Kiểm tra thông tin lưu trữ của cookies trong trang web. Không nên có định dạng dễ đọc.</li>
<li>Kiểm tra các lỗ hổng cũ để kiểm tra xem có hiệu quả không.</li>
<li>Kiểm tra xem có port nào đang mở trong mạng.</li>
<li>Kiểm tra tất cả các thiết bị điện thoại.</li>
<li>Kiểm tra bảo mật mạng wifi.</li>
<li>Kiểm tra tất cả các phương thức của HTTP. Phương thức PUT và DELETE không nên được bật trên máy chủ web.</li>
<li>Kiểm tra nếu mật khẩu đáp ứng các tiêu chuẩn bắt buộc của tổ chức. Mật khẩu phải dài ít nhất 8 ký tự chứa ít nhất một số và một ký tự đặc biệt.</li>
<li>Tên người dùng không phải là <strong>admin</strong> hoặc <strong>administrator</strong>.</li>
<li>Trang đăng nhập ứng dụng phải được khóa sau vài lần đăng nhập không thành công.</li>
<li>Thông báo lỗi phải là chung và không đề cập chi tiết lỗi cụ thể như &quot;tên người dùng không hợp lệ&quot; hoặc &quot;mật khẩu không hợp lệ&quot;.</li>
<li>Kiểm tra các ký tự đặc biệt, thẻ html và script được xử lý đúng như giá trị đầu vào.</li>
<li>Chi tiết hệ thống nội bộ không nên được tiết lộ trong bất kỳ lỗi hoặc thông báo nào.</li>
<li>Thông báo lỗi tùy chỉnh phải được hiển thị cho người dùng trong trường hợp lỗi trang web.</li>
<li>Kiểm tra khi sử dụng registry. Thông tin nhạy cảm không nên được giữ trong registry.</li>
<li>Mọi files phải được quét trước khi tải lên máy chủ.</li>
<li>Dữ liệu nhạy cảm không được thông qua trong url khi giao tiếp với các modules nội bộ khác nhau của ứng dụng web.</li>
<li>Không nên có tên người dùng hoặc mật khẩu nào trong hệ thống.</li>
<li>Kiểm tra tất cả các trường input với chuỗi đầu vào dài và không có khoảng trống.</li>
<li>Kiểm tra chức năng đặt lại mật khẩu phải được bảo mật.</li>
<li>Kiểm tra ứng dụng bằng kiểu tấn công <strong>SQL Injection</strong>.</li>
<li>Kiểm tra ứng dụng bằng kiểu tấn công <strong>Cross Site Scripting</strong>.</li>
<li>Validate các trường input phải được thực hiện ở phía server thay vì kiểm tra javascript ở phía ứng dụng.</li>
<li>Các tài nguyên quan trọng trong hệ thống cần được phân cho những người được ủy quyền.</li>
<li>Tất cả các nhật ký truy cập phải được duy trì với quyền truy cập đúng.</li>
<li>Kiểm tra session người dùng khi đăng xuất.</li>
<li>Kiểm tra việc duyệt thư mục bị vô hiệu hóa trên máy chủ.</li>
<li>Kiểm tra rằng tất cả các ứng dụng và phiên bản cơ sở dữ liệu đều được cập nhật mới nhất.</li>
<li>Kiểm tra trên thanh địa chỉ url để xem ứng dụng web có hiển thị bất kỳ thông tin không mong muốn nào không.</li>
<li>Kiểm tra rò rỉ bộ nhớ và tràn bộ đệm.</li>
<li>Kiểm tra lưu lượng truy cập mạng từ ngoài vào để tìm kiếm các cuộc tấn công bằng <strong>trojan</strong>.</li>
<li>Kiểm tra xem hệ thống có an toàn không.</li>
<li>Kiểm tra hệ thống hoặc mạng có được bảo đảm trong các cuộc tấn công từ <strong>DOS (Denial-of-Service)</strong>. Hacker có thể nhắm mục tiêu mạng hoặc một máy tính duy nhất có request liên tục do tài nguyên trên hệ thống mục tiêu bị quá tải dẫn đến việc các dịch vụ bị tê liệt.</li>
<li>Kiểm tra các cuộc tấn công HTML script.</li>
<li>kiểm tra các cuộc tấn công <strong>COM &amp; ActiveX</strong>.</li>
<li>Kiểm tra các cuộc tấn công <strong>spoofing</strong>. Spoofing có thể làm giả địa chỉ ip, giả mạo id email, giả mạo <strong>ARP</strong>, giả mạo các liên kết, giả mạo id người gọi.</li>
<li>Kiểm tra các cuộc tấn công khi chuỗi định dạng không được kiểm soát - một cuộc tấn công có thể gây ra ứng dụng này bị crash hoặc thực hiện một đoạn script độc hại trên đó.</li>
<li>Kiểm tra cuộc tấn công <strong>XML Injection</strong>.</li>
<li>kiểm tra cuộc tấn công <strong>Directory Traversal</strong>.</li>
<li>Kiểm tra các trang lỗi đang hiển thị bất kỳ thông tin nào có thể hữu ích cho một hacker để xâm nhập vào hệ thống.</li>
<li>Kiểm tra nếu có dữ liệu quan trọng như mật khẩu được lưu trữ trong các tập tin bí mật trên hệ thống.</li>
<li>Kiểm tra xem ứng dụng có trả lại nhiều dữ liệu hơn không.</li>
</ol>
<h1 id="cac-chuan-kiem-tra-tham-nhap">Các chuẩn Kiểm Tra Thâm Nhập</h1>
<ul>
<li><a href="https://www.pcisecuritystandards.org/">Payment Card Industry Data Security Standard (PCI DSS)</a></li>
<li><a href="https://owasp.org/">Open Web Application Security Project (OWASP)</a></li>
<li><a href="http://www.iso.org/iso/catalogue_detail?csnumber=50297">ISO/IEC 27002</a>- <a href="https://www.isecom.org/OSSTMM.3.pdf">The Open Source Security Testing Methodology Manual (OSSTMM)</a></li>
</ul>
<h1 id="cac-chung-chi-trong-kiem-tra-tham-nhap">Các chứng chỉ trong Kiểm Tra Thâm Nhập</h1>
<ul>
<li><a href="https://www.eccouncil.org/programs/certified-ethical-hacker-ceh/">EC-Council Certified Ethical Hacker (CEH)</a></li>
<li><a href="https://www.eccouncil.org/programs/licensed-penetration-tester-lpt-master/">EC-Council Licensed Penetration Tester (LPT) - Master</a></li>
<li><a href="https://www.iacertification.org/cpt_certified_penetration_tester.html">IACRB Certified Penetration Tester (CPT)</a></li>
<li><a href="https://www.iacertification.org/cept_certified_expert_penetration_tester.html">Certified Expert Penetration Tester (CEPT)</a></li>
<li><a href="https://www.iacertification.org/cmwapt_certified_moible_and_web_app_penetration_tester.html#:~:text=The%20Certified%20Mobile%20and%20Web,mobile%20and%20web%20application%20field.">Certified Mobile and Web Application Penetration Tester (CMWAPT)</a></li>
<li><a href="https://www.iacertification.org/crtop_certified_red_team_operations_professional.html">Certified Red Team Operations Professional (CRTOP)</a></li>
<li><a href="https://www.comptia.org/certifications/pentest">CompTIA PenTest+</a></li>
<li><a href="https://www.giac.org/certification/penetration-tester-gpen">GIAC Penetration Tester (GPEN)</a></li>
<li><a href="https://www.giac.org/certification/exploit-researcher-advanced-penetration-tester-gxpn">GIAC Exploit Researcher and Advanced Penetration Tester (GXPN)</a></li>
<li><a href="https://www.youracclaim.com/org/offensive-security/badge/offensive-security-certified-professional-oscp">Offensive Security Certified Professional (OSCP)</a></li>
</ul>
<h1 id="ket-luan">Kết luận</h1>
<p>Với vai trò là người kiểm tra thâm nhập, bạn nên thu thập và ghi lại tất cả các lỗ hổng trong hệ thống. Đừng bỏ qua bất kỳ tình huống nào nếu xét rằng nó sẽ không được thực thi bởi người dùng cuối.</p>
