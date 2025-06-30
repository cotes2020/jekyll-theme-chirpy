---
title: "[DS/Product DS/Marketing] 이커머스 산업에서 꼭 알아야 하는 지식들! - 검색/성과 지표 개념"
author: eun
date: 2025-06-29 22:26:00 +0900
categories: [Data Science, Product DS]
tags: [Intern, Marketing]
render_with_liquid: true
image:
  path: https://ems.elancer.co.kr/99_upload/Append/T_Blog/editor/2024031401122245482.jpg
  # lqip: data:image/webp;base64,UklGRpoAAABXRUJQVlA4WAoAAAAQAAAADwAABwAAQUxQSDIAAAARL0AmbZurmr57yyIiqE8oiG0bejIYEQTgqiDA9vqnsUSI6H+oAERp2HZ65qP/VIAWAFZQOCBCAAAA8AEAnQEqEAAIAAVAfCWkAALp8sF8rgRgAP7o9FDvMCkMde9PK7euH5M1m6VWoDXf2FkP3BqV0ZYbO6NA/VFIAAAA
  alt: Dashboard
---

DS팀에서는 고객의 행동 데이터를 분석해 대시보드를 제작하고, 이에 대한 성과를 올릴 수 있는 인사이트를 도출하는데요! 마케팅이나 영업 등 고객의 패턴을 추적하고 확인하는 데 있어서는 꼭꼭!!! 알아야 하는 개념들입니다. 회사에서 과제를 하나 수행하게 되었는데, 특정 조건에 대한 지표를 제작하는 것이었습니다. 생소한 내용이기 때문에 공부할 겸, 익숙해지고자 정리해보았습니다 ヾ(≧▽≦*)o

![](https://ems.elancer.co.kr/99_upload/Append/T_Blog/editor/2024031401122245482.jpg)

<br>

## 지표 설명

---

- **UV(Unique Visitor):**
    - 순 방문자 수
    - 중복되는 방문자를 제외하고, 실제 방문한 고유한 사용자 수
- **QC(Query Count):**
    - 검색 시도 수
- **NCR:**
    - 검색 시도 대비 클릭으로 **전환되지 않는** 비율(한 제품에 대한)
    - 상품 상세 클릭 or 장바구니 바로 담기 클릭
- **CTR(Click Through Rate):**
    - 검색 시도 대비 클릭으로 **전환되는** 비율(클릭 전환율)(한 제품에 대한)
    - 클릭수/노출수X100
    - CTR+NCR = 100%
- **qCTR**:
    - 클릭 했는지 안했는지에 대한 0/1 이진 값으로 표현
- **ATC CVR: 검색 시도 대비 장바구니 담기로 이어지는 비율**
    - ATC: add to cart( 보기 → 담기로 전환)
- **CVR(Click Conversion Rate)**
    - 검색 수 대비 구매로 이어지는 비율
    - 전환수/검색수X100
- **ASP(Average Selling Price)**
    - 검색을 통한 평균 거래 단가(=검색을 통한 평균 판매 구매 금액)
- **GMV**
    - 총 매출 금액
- **Median Result Count**
    - 검색 결과로 나온 상품 개수의 중앙값
        - ex) 평균적으로 계란을 검색하면 90개 정도 결과가 나온다.
            - (사람마다 배송지역이 다 다르기 때문에 배송 가능한 상품에 대한 관련 검색어로 계산이 됨)
- **Median Click Position**
    - 몇 번째에 있는 상품을 선택했는지
    - 지표를 통한 분석 접근 예)
        - 치즈 키워드를 검색했을 때, 소비자들은 50번째에 해당하는 상품을 가장 많이 구매한다. 그렇다면, 50번째에 있는 치즈를 위로 올리면 구매 전환율이 높아지지 않을까?

<br><br><br><br>

개념들만 간단하게 작성해봤는데요.. 사실 코드로 직접 짜보는게 이해하기 더 쉽습니다! SQL문으로 열심히 끄적하고 있긴 한데 워낙 생소한 용어이다 보니, 아직은 어색하고 어렵네요 ^^.. 다음에는 직접 코드로 수식을 작성해보는 리뷰를 진행해보겠습니다! φ(*￣0￣)