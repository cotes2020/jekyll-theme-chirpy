---
icon: fas fa-chart-line 
order: 7
mermaid: true
---

<div class="bi-guide-cards">
  <a href="#chart-fundamentals" class="bi-guide-card">
    <span><span class="bi-card-ko">1. 차트 기본사항</span><span class="bi-card-en">(Chart Fundamentals)</span></span>
  </a>
  <a href="#dataset" class="bi-guide-card">
    <span><span class="bi-card-ko">2. 데이터셋 설명</span><span class="bi-card-en">(Dataset)</span></span>
  </a>
  <a href="#bar-chart" class="bi-guide-card">
    <span><span class="bi-card-ko">3. 막대차트</span><span class="bi-card-en">(Bar Chart)</span></span>
  </a>
  <a href="#mixed-chart" class="bi-guide-card">
    <span><span class="bi-card-ko">4. 혼합 차트</span><span class="bi-card-en">(Mixed Chart)</span></span>
  </a>
  <a href="#pie-chart" class="bi-guide-card">
    <span><span class="bi-card-ko">5. 원형 차트</span><span class="bi-card-en">(Pie Chart)</span></span>
  </a>
  <a href="#dashboard" class="bi-guide-card">
    <span><span class="bi-card-ko">6. 대시보드</span><span class="bi-card-en">(Dashboard)</span></span>
  </a>
  <a href="#appendix1" class="bi-guide-card">
    <span><span class="bi-card-ko">7. 부록</span><span class="bi-card-en">설립일 3년 이내 투자건 필터링 방법</span></span>
  </a>
  <a href="#appendix2" class="bi-guide-card">
    <span><span class="bi-card-ko">8. 부록</span><span class="bi-card-en">제안사 현황에서 기준일자 선택 필터 구성</span></span>
  </a>

</div>

<style>
.bi-guide-cards {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 1rem;
  margin-bottom: 2rem;
}
.bi-guide-card {
  display: flex;
  align-items: center;
  justify-content: center;
  min-height: 124px;
  padding: 1rem 1rem;
  background: var(--card-bg, #f8f9fa);
  border: 1px solid var(--card-border-color, #dee2e6);
  border-radius: 0.5rem;
  color: var(--text-color, #333);
  text-decoration: none;
  transition: transform 0.2s, box-shadow 0.2s;
}
.bi-guide-card:hover {
  transform: translateY(-2px);
  box-shadow: 0 4px 12px rgba(0,0,0,0.15);
  background: var(--card-hovor-bg, #e2e2e2);
  text-decoration: none;
}
.content a.bi-guide-card:hover {
  color: var(--text-color, #333) !important;
  border-bottom: none !important;
}
.bi-guide-card > span {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  row-gap: 0.7rem;
  text-align: center;
}
.bi-guide-card .bi-card-ko {
  font-size: 1.3rem;
  font-weight: 500;
  line-height: 1.25;
}
.bi-guide-card .bi-card-en {
  font-size: 0.9rem;
  font-weight: 400;
  line-height: 1.25;
}
@media (max-width: 768px) {
  .bi-guide-cards {
    grid-template-columns: repeat(2, minmax(0, 1fr));
  }
}
#chart-fundamentals,
#dataset,
#bar-chart,
#mixed-chart,
#pie-chart,
#dashboard,
#appendix1,
#appendix2 {
  scroll-margin-top: 2rem;
}
</style>

<br>

## BI 란?

BI (Business Intelligence) 기능은 VCworks를 사용하며 축적된 데이터를 손쉽게 가공하여 시각화할 수 있는 툴입니다. BI의 가장 기본적인 단위는 **차트** 이고, 이 차트가 모여 **대시보드** 를 이루고 있습니다. 즉, 대시보드는 다수의 차트를 동시에 볼 수 있는 화면이고, 이 화면을 VCworks 내의 메뉴로 추가하실 수 있습니다.

<br>

---
### 1. 차트 기본사항 (Chart Fundamentals) {#chart-fundamentals}

가장 기본적인 차트는 테이블 차트입니다. 따라서 테이블 차트를 기준으로 차트 기본사항을 안내드립니다.

<div style="position: relative; box-sizing: content-box; max-height: 80vh; max-height: 80svh; width: 100%; aspect-ratio: 1.78; padding: 40px 0 40px 0;">
  <iframe src="https://app.supademo.com/embed/cmov6ezla0fho9rr9vjiu9fk5?embed_v=2&utm_source=embed" loading="lazy" title="차트 기본사항" allow="clipboard-write" frameborder="0" webkitallowfullscreen="true" mozallowfullscreen="true" allowfullscreen style="position: absolute; top: 0; left: 0; width: 100%; height: 100%;"></iframe>
</div>

<div style="position: relative; box-sizing: content-box; max-height: 80vh; max-height: 80svh; width: 100%; aspect-ratio: 1.78; padding: 40px 0 40px 0;">
  <iframe src="https://app.supademo.com/embed/cmpw5wv9w5o1cqmy7ydo65vsi?embed_v=2&utm_source=embed" loading="lazy" title="2. 테이블 차트" allow="clipboard-write" frameborder="0" webkitallowfullscreen="true" mozallowfullscreen="true" allowfullscreen style="position: absolute; top: 0; left: 0; width: 100%; height: 100%;"></iframe>
</div>

<br>

---

### 2. 데이터셋 설명 (Dataset) {#dataset}

차트 생성 화면에서 "분석 대상"에 해당하는 각각의 데이터셋을 별도의 페이지에서 상세하게 설명드립니다.

<div style="position: relative; box-sizing: content-box; max-height: 80vh; max-height: 80svh; width: 100%; aspect-ratio: 1.78; padding: 40px 0 40px 0;">
  <iframe src="https://app.supademo.com/embed/cmq0nrttd18heqmgh9ye0zwsp?embed_v=2&utm_source=embed" loading="lazy" title="6. 데이터셋 설명" allow="clipboard-write" frameborder="0" webkitallowfullscreen="true" mozallowfullscreen="true" allowfullscreen style="position: absolute; top: 0; left: 0; width: 100%; height: 100%;"></iframe>
</div>



---
### 3. 막대차트 (Bar Chart) {#bar-chart}

테이블 차트에서 두 개의 컬럼을 뽑아 하나는 x축, 다른 하나는 막대의 높이로 나타낸 것이 막대 차트입니다.


<div style="position: relative; box-sizing: content-box; max-height: 80vh; max-height: 80svh; width: 100%; aspect-ratio: 1.78; padding: 40px 0 40px 0;">
  <iframe src="https://app.supademo.com/embed/cmpyqh8fy9z0gqmy7epdwdhj4?embed_v=2&utm_source=embed" loading="lazy" title="3. 막대 차트 (Bar Chart)" allow="clipboard-write" frameborder="0" webkitallowfullscreen="true" mozallowfullscreen="true" allowfullscreen style="position: absolute; top: 0; left: 0; width: 100%; height: 100%;"></iframe>
</div>

---
### 4. 혼합 차트 (Mixed Chart) {#mixed-chart}

혼합 차트는 막대 차트와 선 차트 등 서로 다른 유형의 차트를 하나의 그래프에 함께 표현하여, 단위나 스케일이 다른 두 지표를 한눈에 비교할 수 있는 차트입니다.

<div style="position: relative; box-sizing: content-box; max-height: 80vh; max-height: 80svh; width: 100%; aspect-ratio: 1.78; padding: 40px 0 40px 0;">
  <iframe src="https://app.supademo.com/embed/cmpysrtt9a11vqmy7rds0zlhe?embed_v=2&utm_source=embed" loading="lazy" title="4. 혼합 차트" allow="clipboard-write" frameborder="0" webkitallowfullscreen="true" mozallowfullscreen="true" allowfullscreen style="position: absolute; top: 0; left: 0; width: 100%; height: 100%;"></iframe>
</div>

---
### 5. 원형 차트 (Pie Chart) {#pie-chart}

<div style="position: relative; box-sizing: content-box; max-height: 80vh; max-height: 80svh; width: 100%; aspect-ratio: 1.78; padding: 40px 0 40px 0;">
  <iframe src="https://app.supademo.com/embed/cmpyz3y3ha5v2qmy7y5mwro8b?embed_v=2&utm_source=embed" loading="lazy" title="4. 원형차트" allow="clipboard-write" frameborder="0" webkitallowfullscreen="true" mozallowfullscreen="true" allowfullscreen style="position: absolute; top: 0; left: 0; width: 100%; height: 100%;"></iframe>
</div>

---
### 6. 대시보드 (Dashboard) {#dashboard}

대시보드에서는 다수의 차트를 한 번에 볼 수 있고, 필터를 추가하여 일괄로 적용할 수 있습니다.

<div style="position: relative; box-sizing: content-box; max-height: 80vh; max-height: 80svh; width: 100%; aspect-ratio: 1.78; padding: 40px 0 40px 0;">
  <iframe src="https://app.supademo.com/embed/cmpz3gmsxacp4qmy70qnpi008?embed_v=2&utm_source=embed" loading="lazy" title="5. 대시보드 생성 및 임베딩" allow="clipboard-write" frameborder="0" webkitallowfullscreen="true" mozallowfullscreen="true" allowfullscreen style="position: absolute; top: 0; left: 0; width: 100%; height: 100%;"></iframe>
</div>

<div style="position: relative; box-sizing: content-box; max-height: 80vh; max-height: 80svh; width: 100%; aspect-ratio: 1.78; padding: 40px 0 40px 0;">
  <iframe src="https://app.supademo.com/embed/cmq0cjvih0rl3qmghd2zauqsr?embed_v=2&utm_source=embed" loading="lazy" title="5. 대시보드 필터링" allow="clipboard-write" frameborder="0" webkitallowfullscreen="true" mozallowfullscreen="true" allowfullscreen style="position: absolute; top: 0; left: 0; width: 100%; height: 100%;"></iframe>
</div>

<br>

---
## 7. Appendix - 설립일 3년 이내 투자건 필터링 방법 (SQL 구문 삽입) {#appendix1}

초기창업 투자 건만을 필터링하고 싶다면 다음 가이드를 참고하여 아래의 SQL 구문을 붙여넣으실 수 있습니다.

<div style="position: relative; box-sizing: content-box; max-height: 80vh; max-height: 80svh; width: 100%; aspect-ratio: 1.78; padding: 40px 0 40px 0;">
  <iframe src="https://app.supademo.com/embed/cmq0s3hsd1h6yqmghatlc81a2?embed_v=2&utm_source=embed" loading="lazy" title="부록. 초기창업 여부" allow="clipboard-write" frameborder="0" webkitallowfullscreen="true" mozallowfullscreen="true" allowfullscreen style="position: absolute; top: 0; left: 0; width: 100%; height: 100%;"></iframe>
</div>


- 아래의 SQL 구문을 그대로 복사하여 이용하면 됩니다.
  - 만약 연도를 바꾸고 싶은 경우 `years` 앞 숫자를 조절하세요.
  
- 설립일 3년 이내 사용자 컬럼 추가용
```sql
CASE
  WHEN
    "납입일자" <= "설립일" + INTERVAL '3 years'
  THEN 'O'
  ELSE 'X'
END
```

- 설립일 3년 이내 필터용
```sql
"납입일자" <= "설립일" + INTERVAL '3 years'
```

<br>

※ 초기기업 여부 컬럼 이외에 다른 컬럼을 직접 정의하고 싶으신 경우, "2. 데이터셋" 세션에서 안내하는 페이지의 데이터셋 설명과 컬럼 정보를 ChatGPT, Claude 등의 AI에게 컨텍스트로 제공하시면서 원하는 컬럼을 설명하시면 그에 맞는 SQL문을 받으실 수 있습니다.


<br>

---
## 8. Appendix - 제안사 현황에서 기준일자 선택 필터 구성 {#appendix2}

<div style="position: relative; box-sizing: content-box; max-height: 80vh; max-height: 80svh; width: 100%; aspect-ratio: 1.60; padding: 40px 0 40px 0;">
  <iframe src="https://app.supademo.com/embed/cmpnfv511099yqmxc3g132g4s?embed_v=2&utm_source=embed" loading="lazy" title="Tips. 제안사 현황 작성 시, 기준일자 선택 방법" allow="clipboard-write" frameborder="0" webkitallowfullscreen="true" mozallowfullscreen="true" allowfullscreen style="position: absolute; top: 0; left: 0; width: 100%; height: 100%;"></iframe>
</div>

---


