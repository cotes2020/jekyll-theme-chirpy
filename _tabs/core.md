---
icon: fas fa-compass
order: 1
mermaid: true
---
VCworks.kr은 똑똑[^dkdk] 주식회사에서 만든 대한민국 Venture Capital ERP Solution입니다.   

## VC업무의 일반 흐름
- 다음의 항목을 클릭하여 관련있는 항목으로 바로 이동할 수 있습니다.

```mermaid
flowchart TD
 subgraph s1["설정"]
        hr0001["구성원 등록"]
        hr0002["구성원 정보 수정"]
        hr0007["조직도 관리"]
  end
 subgraph s2["결성"]
        fd0001["조합생성(개요등록)"]
        fd0010["투자의무달성현황 등록"]
        fd0011["조합원명부 등록"]
        fm0010["재원별 회계원장 등록"]
        fd0009a["금융정보 등록(계좌등록)"]
        fd0013["필수 필요서류 등록"]
        fd0012["보수정보 등록"]
        fd0000(("결성완료"))
        fd0006["출자/배분 등록"]
        oi0002["출자 운용지시서 등록(전자결재)"]
        fm0001a(("자동전표"))
  end
 subgraph s3["투심"]
        wr0003["주간회의(딜소싱)"]
        vs0003["투자정보입력"]
        vs0006["투자심의(표결)"]
        ed0001a["계약품의(전자결재)"]
        oi0003["투자 운용지시(전자결재)"]
        vs0009["투자완료(첨부파일등록)"]
  end
 subgraph s4["가치평가"]
        pm0100["포트폴리오 정보"]
        pm0300["가치평가"]
        pm0301["투자유형별 등록"]
        pm0302["재원별 등록"]
        pm0303["가치평가 조회"]
  end
 subgraph s5["영업보고"]
        br0001["템플릿 관리(작성 및 확인)"]
        br0012["포트폴리오 연락망"]
        br0004["영업보고 요청"]
        br0007["영업보고 검수"]
        br0009["심사역 의견 작성"]
        br0011["영업보고서 생성"]
  end
 subgraph s6["거래원장"]
        pm0001["거래원장관리"]
        pm0004["거래등록"]
        fm0002["전표입력"]
  end
 subgraph s7["회수"]
        ex0001["회수위원회(표결)"]
        oi0001["출고 운용지시(전자결재)"]
        ex0007["회수관리"]
        ex0009["회수 전자결재"]
        ex0009a["회수 운용지시(전자결재)"]
        fm0002a["전표입력"]
  end
 subgraph s8["의결"]
        sa0002["의결 응답"]
        sa0003["의결 정보 반영"]
  end
    hr0001 --> hr0002
    hr0002 --> hr0007
    hr0007 --> fd0001
    fd0001 --> fd0010 & fd0011 & fd0009a
    fm0010 --> fd0009a
    fd0009a --> fd0000
    fd0010 --> fd0000
    fd0013 --> fd0000
    fd0012 --> fd0000
    fd0011 --> fd0006
    fd0006 --> oi0002
    oi0002 --> fm0001a & fd0000
    fd0000 --> wr0003
    wr0003 --> vs0003
    vs0003 --> vs0006
    vs0006 --> ed0001a
    ed0001a --> oi0003
    oi0003 --> vs0009
    vs0009 --> pm0100 & ex0001 & sa0002
    pm0100 --> pm0300 & br0012
    pm0300 --> pm0301 & pm0302
    pm0301 --> pm0303
    pm0302 --> pm0303
    pm0303 --> br0011
    br0001 --> br0004
    br0012 --> br0004
    br0004 --> br0007
    br0007 --> br0009
    br0009 --> br0011
    oi0003 -. 선택적 .-> pm0001
    pm0001 --> fm0002 & pm0004 & ex0001
    pm0004 --> fm0002
    br0011 --> ex0001
    ex0001 --> oi0001
    oi0001 --> ex0007
    ex0007 --> ex0009
    ex0009 --> ex0009a
    ex0009a --> fm0002a
    sa0002 -. 선택적 .-> sa0003
    sa0003 --> pm0001

    click hr0001 "{% post_url 2024-07-02-hr0001 %}"
    click hr0002 "{% post_url 2024-07-02-hr0002 %}"
    click hr0007 "{% post_url 2024-07-02-hr0007 %}"
    click fd0001 "{% post_url 2024-07-04-fd0001 %}"
    click fd0010 "{% post_url 2024-07-04-fd0010 %}"
    click fd0011 "{% post_url 2024-07-04-fd0011 %}"
    click fm0010 "{% post_url 2024-07-04-fm0010 %}"
    click fd0009a "{% post_url 2024-07-05-fd0009a %}"
    click fd0013 "{% post_url 2024-07-05-fd0013 %}"
    click fd0012 "{% post_url 2024-07-06-fd0012 %}"
    click fd0006 "{% post_url 2024-07-07-fd0006 %}"
    click oi0002 "{% post_url 2024-07-07-oi0002 %}"
    click wr0003 "{% post_url 2024-07-09-wr0003 %}"
    click vs0003 "{% post_url 2024-07-10-vs0003 %}"
    click vs0006 "{% post_url 2024-07-11-vs0006 %}"
    click ed0001a "{% post_url 2024-07-12-ed0001a %}"
    click oi0003 "{% post_url 2024-07-13-oi0003 %}"
    click vs0009 "{% post_url 2024-07-14-vs0009 %}"
    click pm0300 "{% post_url 2024-09-01-pm0300 %}"
    click pm0301 "{% post_url 2024-09-02-pm0301 %}"
    click pm0302 "{% post_url 2024-09-03-pm0302 %}"
    click pm0303 "{% post_url 2024-09-04-pm0303 %}"
    click br0001 "{% post_url 2024-08-23-br0001 %}"
    click br0012 "{% post_url 2024-08-22-br0012 %}"
    click br0004 "{% post_url 2024-08-24-br0004 %}"
    click br0007 "{% post_url 2024-08-25-br0007 %}"
    click br0009 "{% post_url 2024-08-26-br0009 %}"
    click br0011 "{% post_url 2024-08-27-br0011 %}"
    click pm0001 "{% post_url 2024-07-15-pm0001 %}"
    click pm0004 "{% post_url 2024-07-16-pm0004 %}"
    click fm0002 "{% post_url 2024-07-17-fm0002 %}"
    click ex0001 "{% post_url 2024-07-18-ex0001 %}"
    click oi0001 "{% post_url 2024-07-19-oi0001 %}"
    click ex0007 "{% post_url 2024-07-20-ex0007 %}"
    click ex0009 "{% post_url 2024-07-21-ex0009 %}"
    click ex0009a "{% post_url 2024-07-22-ex0009a %}"
    click fm0002a "{% post_url 2024-07-23-fm0002a %}"
    click sa0002 "{% post_url 2024-09-11-sa0002 %}"
    click sa0003 "{% post_url 2024-09-12-sa0003 %}"

```

버그 및 문의 사항은 다음 이메일로 보내주세요: **[we@dkdk.kr](mailto:we@dkdk.kr)**


---

[^dkdk]:똑똑(dkdk.kr)은 대한민국 벤처투자전문회사인 DSC인베스트먼트가 VC업계의 업무 방식을 혁신하고자 만든 IT자회사입니다. 
