--- 
title: "Schema Matching" 
description: 진행 중인 프로젝트에 대한 조사 
author: cylanokim 
date: 2025-09-19 20:00:00 +0800
categories: [Schema_Matching_Project, 자료 조사]
tags: [schema, 조사]
pin: true
math: true
mermaid: true
---

현재 나는 대규모 언어 모델(LLM)을 활용한 스키마 매칭(Schema Matching) API를 개발 중이다. 이를 위하여 우선 스키마 매칭은 무엇인지, 그리고 관련된 논문을 우선 찾아보기로 하였다. 생각보다 스키마 매칭은 데이터 분석에 있어 흥미로운 주제였다. 이번 포스팅은 대규모 언어 모델 등장 이전에 선행되었던 스키마 매칭 방법론을 간략하게 정리하였다. 

## schema matching 이란?
- 스키마 매칭은 서로 다른 두 데이터 셋의 스키마(테이블 구조, 속성명, 의미)를 자동으로 매핑해주는 과정을 말한다. 
- 스키마 매칭은 서로 다른 관계형 데이터 통합, 마이그레이션 및 비교를 위해서 매우 중요하다. 

## 전통적인 schema matching 방법론
1. 문자열 기반 해석 
- 속성의 이름 자체를 비교
- `edit_distance`, `Jaccard Similarity`, `n-gram`과 같은 문자열 유사도를 이용하여 비교
- 단점: 문자 구조만으로 의미론적 해석이 안되는 경우도 많다. 

2. 스키마 구조 기반 기법
- 스키마의 계층적 구조를 통해 부모/자식 관계를 고려하여 추론
- 단점: 만약 데이터 셋의 스키마에 계층적 구조가 없다면?

3. 데이터의 type과 분포 기반으로 매칭
- 데이터의 실제 값의 type과 샘플링을 통한 분포를 통해 매칭
- 단점: 데이터가 충분히 있어야함. 그리고 서로 다른 데이터인데, 값의 범위가 비슷한 경우도 많음

4. 사전/orthology 기반 기법
- WordNet등을 활용하여 속성명을 의미론적으로 매칭
- 단점: 도메인 특화 용어(의학, 반도체...)에 대응이 어려움

5. 머신러닝 기반 기법
- 문자열 유사도, 데이터 타입, 값의 분포 등을 통해 supervised/unsupervised 방식으로 학습
- 대표 연구: COMA, Cupid, LSD 
- 단점: 라벨링된 데이터가 필요하거나 데이터셋이 많아야함. 

---

## 관련 schema matching 연구 논문 요약
###  1.Generic Schema Matching with Cupid
- 본 논문의 특징
    - 새로운 범용 스키마 매칭 알고리즘 Cupid 제안 논문
    - 스키마 요소의 이름, 데이터 타입, 스키마 구조를 분석하여 매핑. 
    - 언어적(linguistic), 구조적(structural) 매칭 두 단계로 구성
    - 언어적 매칭: 스키마 요소의 이름을 정규화, 범주화, 비교하여 유사도(linguistic similarity coefficient, lsim)을 계산 
    - 구조적 매칭: 요소 주변의 맥락 정보를 바탕으로 구조적 유사도(ssim)를 계산
- 예상 문제점
    - 전문 영역의 데이터 셋은 약어로 구성된 경우가 많다. 이는 도메인 지식 기반으로 추론하지 

### **2. Schema Matching Using neural network** 
- 본 논문의 특징
    - Y. Li, D.-B. Liu and W.-M. Zhang, "Schema Matching Using Neural Network"
    - 문제 정의: 스키마 매칭은 서로 다른 데이터 소스의 스키마를 의미론적 이질성을 찾는 것 (Semantic Heterogeneity)
    - 기존 방식은 시간이 많이 소요, 오류가 발생하기 쉽고, 비용이 많이 듬
    - 위 논문은 데이터 분포에 기반한 스키마 배칭 방법 (SMDD, Schema Matching method based on Data Distribution)을 소개한다. 
    - SMDD는 데이터 자체의 유사성을 기반으로 cluster를 형성하여 범주화 함. unsupervised 러닝으로 추가의 라벨링 작업이 필요하지 않음 

- SMDD 방법의 예상 문제점
    - SMDD는 데이터 분포와 특징을 통해 분류한다. 그러나 이는 데이터 분포는 비슷하지만 서로 다른 스키마에 대해 잘못 매칭 작업이 진행될 가능성이 있다. 
    - 만약 매칭을 하려는 데이터 셋의 양이 작으면 효과적인 매칭이 안될 가능성이 높다. 