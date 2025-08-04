---
title: 마이그레이션 - Hiworks 시스템 전자결재 데이터 추출 가이드
author: dkdk.kr
date: 2025-07-29 16:32:00 +0800
categories:
  - 마이그레이션 전자결재
tags:
  - 마이그레이션
  - 전자결재
render_with_liquid: true
mermaid: true
math: true
---
타 ERP에서 VCWorks로 전자결재 데이터를 이전하기 위한 데이터 추출 가이드 입니다.

### 1. 하이웍스 오피스에 로그인합니다.
> 전자결재 관리 권한을 가진 계정으로 로그인 해주셔야합니다.
{: .prompt-info }

### 2. 로그인 후 오피스 홈에서 [전자결재] 메뉴로 이동합니다.
![step2](/assets/img/Hiworks_전자결재_1.png)

### 3. [관리자 설정 > 문서백업]을 클릭 후 백업할 항목 우측에 있는 [백업] 글씨를 클립합니다.
![step3](/assets/img/Hiworks_전자결재_2.png)

### 4. 기간 선택, 기안자 전체로 선택, 백업 안내 메일 주소를 입력 후 확인을 클릭합니다. 백업이 완료 되면 기입하신 메일로 백업 완료 안내 메일이 발송됩니다.
![step4](/assets/img/Hiworks_전자결재_3.png)

### 5. 문서 백업이 완료되면  [전자결재 > 관리자 설정 > 문서 백업 > 백업 확인 > 다운로드]를 클릭해 사용자 PC에 다운로드하여 압축해제 후 이용할 수 있습니다.
![step5](/assets/img/Hiworks_전자결재_4.png)

### 6. 아래와 같은 형식으로 다운로드 된 파일을 똑똑에게 전달합니다.
```plaintext
C:.
│  backup-index.html
│  
├─data
│      data.js
│      data_info.js
│
└─static
    ├─css
    │      ...
    ├─images
    │      ...
    └─scripts
            approval.js
            common.js
            jquery-1.11.2.min.js
            jquery-ui.min.js
            jquery.tmpl.min.js
            print.js
            template.js
            underscore-min.js
```



