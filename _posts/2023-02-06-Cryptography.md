---
layout: post
title: "[FIA] Intro Cryptography"
summary: "Mật mã"
author: technical
date: '2023-02-06 9:00:00'
category: CTF
thumbnail: assets/img/thumbnail/crypto.jpeg
keywords: Hacking, CTF
permalink: /blog/Cryptography/
usemathjax: true
---

# Cryptography

# Tổng quan

Cryptography hay còn được gọi mật mã học là một bộ môn nghiên cứu và ứng dụng các thuật toán hoặc phương thức mã hóa khác nhau để phục vụ cho việc bảo mật các loại thông tin khác nhau. Những challenge CTF thuộc mảng Cryptography này cũng được mang đến với mục đích giúp người chơi tìm hiểu thêm về các thuật toán để ứng dụng chúng giải ra flag. Lịch sử phát triển của mật mã học có tuổi đời rất lâu, vì thế mà các bài CTF có thể trải rộng từ những mã hóa sơ khai nhất có thể kể đến như **Caesar Cipher** cho đến tái hiện lại các cuộc tấn công mật mã ngày nay. Các challenge này thường nặng về kỹ năng toán học.

# Đặc điểm

## Mã hóa là gì ? Sự khác nhau giữa Encrypt và Encode ?

Có thể nói hai thuật ngữ này nếu dịch sang tiếng Việt đều có nghĩa là mã hóa, đều có chung một mục đích là đem thông tin của người gửi (**plain text**) chuyển hóa sang một dạng thông tin khác hoặc làm phức tạp nó (**cipher text**) để không một ai ngoài người nhận có thể hiểu được nội dung của người gửi.

Điểm khác biệt lớn nhất giữa hai thuật ngữ này đến từ khái niệm "khóa" (**key**). Key ở đây giống như password của thuật toán vậy, nó có thể là một con số hoặc một chuỗi ký tự để mà khi mã hóa cùng một plain text, cùng một thuật toán nhưng key khác nhau thì ta sẽ có các cipher text khác nhau.:

- **Encode**: đem plain text của người gửi, mã hóa bằng một thuật toán hoặc quy ước nào đó thành cipher text rồi gửi cho người nhận. Người nhận biết được thuật toán mã hóa qua người gửi rồi dùng chính thuật toán đó để giải ra plain text đọc được nội dung. Tuy nhiên các thuật toán mã hóa thì lại thường được public rộng rãi, các hacker có kiến thức về mật mã có thể phán đoán được đó là loại mã hóa nào dựa vào cipher text và dễ dàng sử dụng các công cụ hỗ trợ để decode => không bảo mật.

<p><img class="article-img" src="/assets/img/CTF/Intro/EncodingDecoding.jpg" alt="Header.png" width="1053" height="606"></p>

- **Encrypt**: đem plain text của người gửi, kết hợp nó với "key" rồi mã hóa bằng thuật toán tạo ra cipher text. Các thuật toán mã hóa cũng được public những vẫn đề nằm ở chỗ cái "key", thứ mà chỉ có người gửi và người nhận biết nên cho dù các hacker có lấy được cipher text, biết được thuật toán mã hóa mà không biết key thì cũng không đọc được plain text => bảo mật hơn.

<p><img class="article-img" src="/assets/img/CTF/Intro/Encrypt.jpg" alt="Header.png" width="1053" height="606"></p>

## Một số mã hóa thường gặp

Cryptography có thể được chia ra làm hai loại: mật mã cổ điển (classical cryptography) và mật mã hiện đại (modern cryptography).

### Classical Cryptography

Quá trình mã hóa và giải mã của các mã hóa cổ điển thường phụ thuộc vào sự sáng tạo của người tạo ra nó, nó có thể là bất kỳ quy ước gì được tác giả sử dụng để làm phức tạp quá trình giải mã. Một số Classical Cryptography có thể kể đến như:

- **Monoalphabetic Cipher**: mỗi ký tự trong plain text sẽ được ánh xạ thành một ký tự khác dựa trên key để tạo thành cipher text.
    
    VD: các loại Substitution Cipher (Affine Cipher, Caesar Cipher, ROT13, etc)
    
- **Polyalphabetic Cipher**: cũng là một loại Substitution Cipher nhưng thay vì mã hóa từng ký tự thành một ký tự khác thì loại mã hóa này biến đổi từng ký tự thành một cụm ký tự khác.
    
    VD: Vignere Cipher, Enigma machine, etc
    
- **Strange Cipher:** các loại mã hóa lạ chủ yếu dựa trên sự sáng tạo của tác giả.
    
    VD: Brainfuck
    

**Một số writeup tham khảo:**

- Affine, Caesar Cipher: [WRITE-UP KMA-CTF-2021 LAN 2](https://kcsc-club.github.io/2021/08/30/kma-ctf-02-2021/#CRYPTOGRAPHY)
- Vignere Cipher: [PicoCTF-2021](https://github.com/HHousen/PicoCTF-2021/blob/master/Cryptography/New%20Vignere/README.md)

### Modern Cryptography

Đến với các loại mật mã học hiện đại, chúng bắt đầu áp dụng nhiều hơn các lý thuyết toán học vào trong các thuật toán mã hóa. Năm 1949, bài báo “Communication Theory of Secrecy Systems” của **Claude Shannon** được xuất bản đã đánh dấu cho sự ra đời của mật mã hiện đại. Một số Modern Cryptography có thể kể đến như:

- **Symmetric Cryptography**: mã hóa đối xứng là loại mã hóa mà trong đó quá trình mã hóa và giải mã sử dụng chung một key.
    
    VD: AES, RC4, DES, etc
    
- **Asymmetric Cryptography**: mã hóa bất đối xứng là loại mã hóa sử dụng một public key để mã hóa và dùng một private key khác để giải mã nó.
    
    VD: RSA, ElGammal, etc
    
- **Hash function:** hàm băm mật mã khác với hai loại mã hóa trên do nó có đặc tính một chiều, tức là plain text sau khi cho qua hàm băm mật mã thành cipher text thì không thể decrypt quay lại plain text.
    
    VD: MD5, SHA, etc
    

**Một số writeup tham khảo:**

- AES: [zer0pts CTF 2021 - Crypto](https://jsur.in/posts/2021-03-07-zer0pts-ctf-2021-crypto-writeups#three-aes)
- AES: [lUcgryy/CTF-Write-up](https://github.com/lUcgryy/CTF-Write-up/tree/main/ASCIS%202022%20Quals/Cryptography/Leaky%20AES)
- RSA: [lUcgryy/CTF-Write-up](https://github.com/lUcgryy/CTF-Write-up/tree/main/FPTUHacking_CTF_2022/V%C3%B2ng%20lo%E1%BA%A1i/Cryptography)
- ElGammal: [zer0pts CTF 2021](https://hackmd.io/@theoldmoon0602/r1cMHWCzd)

# Một số công cụ thường dùng

- [CyberChef](https://gchq.github.io/CyberChef/): công cụ thông dụng thường dùng để giải mã các mã hóa đơn giản
- [Dcode](https://www.dcode.fr/cipher-identifier): dùng để phát hiện các loại cipher cũng như decode chúng
- [FeatherDuster](https://github.com/nccgroup/featherduster): phát hiện và khai thác các mã hóa yếu
- [RsaCtfTool](https://github.com/RsaCtfTool/RsaCtfTool): giải mã RSA
- [FactorDB](http://factordb.com/index.php?query=): tìm tích thừa số nguyên tố

# Những điều cần có khi chơi Cryptography

- Có niềm đam mê mãnh liệt với toán học :D
- Có nền tảng lập trình, hiểu biết một số loại ngôn ngữ để có thể đọc hiểu các đoạn mã hóa của chương trình diễn ra như thế nào cũng như viết script để giải mã chúng (Python và các module liên quan là một thế mạnh)

# Nguồn tự học và luyện tập

- [Crypto Hack](https://cryptohack.org/): Nơi các bạn có thể học từ cơ bản mọi kiến thức liên quan đến Crypto và các bài CTF để thực hành
- [PicoCTF](https://picoctf.org/): Nơi chứa các bài CTF từ dễ đến trung bình rất thích hợp cho người mới bắt đầu chơi CTF
- [CtfTime](https://ctftime.org/): tìm kiếm các giải đấu trên nền tảng này và rèn luyện qua từng ngày kèm theo đọc writeup
- [Ebook](https://mrajacse.files.wordpress.com/2012/01/applied-cryptography-2nd-ed-b-schneier.pdf)

### Bài viết được tham khảo từ các nguồn

- [Geeksforgeeks](https://www.geeksforgeeks.org/difference-between-encryption-and-encoding/)
- [CTF-Wiki](https://ctf-wiki.mahaloz.re/crypto/introduction/)