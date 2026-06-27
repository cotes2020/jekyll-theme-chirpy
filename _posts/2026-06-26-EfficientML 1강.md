---
title: MIT EfficientML.ai 1강 Introduction
date: 2026-06-07 16:00:00 +0900
categories: [AI, MIT EfficientML.ai Lecture]
tags: [ai, tinyml, pruning, knowledge_distillation, quantization, hardware_acceleration, llm]     # TAG names should always be lowercase
toc: true
math: true
# image: /assets/img/posts/
---

# Introduction
오늘부터 MIT에서 Song Han 교수의 EfficientML.ai Course를 1강부터 정리해보겠다. 이 글에서 정리하는 2023년 강의의 정식 명칭은 TinyML and Efficient Deep Learning Computing이다. 당시 강의에서는 추론과 학습을 하드웨어 효율적으로 수행하는 방법론뿐 아니라 application별 최적화 기법과 QuantumML도 다뤘다. 2026년 가을 강의는 [TinyML and Efficient AI Computing](https://hanlab.mit.edu/courses/2026-fall-65940)으로 이름과 구성이 갱신되어 LLM post-training, long-context LLM과 agent 등의 주제까지 포함한다. 강의 영상은 [MIT Han Lab 유튜브](https://www.youtube.com/@MITHANLab)에서도 확인할 수 있다. 

1강은 Introduction으로, Song Han 교수님에 대한 배경과, HAN Lab에 대한 간단한 소개, 그리고 왜 하드웨어 가속화가 필요하고, 어떠한 연구들이 있었는지를 다룬다. 2023년 강의에서는 최신 연구를 다루지 않아 강의 자료에 있는 내용들과 함께 내가 따로 조사한 내용들도 같이 작성해보겠다.

# Prof. Song Han

[Song Han 교수](https://hanlab.mit.edu/songhan)는 MIT EECS(Electrical Engineering and Computer Science)의 종신 부교수이자 MIT HAN Lab의 PI이다. 칭화대학교에서 Microelectronics Engineering 학사 학위를 받고, Stanford에서 Bill Dally 교수의 지도를 받아 Electrical Engineering 석사와 박사 학위를 받았다. 박사 졸업 후에는 Google Brain에서 Research Scientist로 일했으며, 2018년에 MIT EECS 교수로 합류했다. 현재는 MIT에서 연구와 교육을 수행하는 동시에 NVIDIA Research의 Efficient AI 팀도 이끌고 있다.

Song Han 교수의 연구를 가장 잘 설명하는 문장은 **"먼저 모델을 작게 만들고, 그 작아진 모델을 실제로 빠르게 실행할 시스템과 하드웨어를 함께 설계한다"**라고 생각한다. 박사 과정에서 발표한 [Deep Compression](https://hanlab.mit.edu/projects/deep-compression)은 pruning, trained quantization, Huffman coding을 순서대로 적용해 AlexNet과 VGG-16의 저장 공간을 정확도 저하 없이 각각 35배, 49배 줄였다. 이어서 발표한 EIE(Efficient Inference Engine)는 이렇게 압축된 sparse model을 그대로 실행하는 전용 가속기였다. 즉, 알고리즘으로 weight와 연산을 줄이는 데서 끝나지 않고 그 sparsity를 실제 속도와 에너지 절감으로 바꾸는 하드웨어까지 연결했다.

이후의 연구도 같은 문제의식을 따른다. Hardware-aware NAS인 [ProxylessNAS](https://hanlab.mit.edu/projects/proxylessnas)와 Once-for-All, microcontroller에서 딥러닝을 실행하는 MCUNet, LLM을 위한 SmoothQuant와 AWQ, long-context inference를 위한 StreamingLLM 등으로 연구 대상은 계속 바뀌었지만, 모델의 정확도는 유지하면서 메모리 사용량, data movement, latency와 energy를 줄인다는 목표는 일관적이다.

대표적인 수상 경력으로는 ICLR 2016, FPGA 2017, MLSys 2024 Best Paper Award, NSF CAREER Award, MIT Technology Review의 35 Innovators Under 35, IEEE의 AI's 10 to Watch, Sloan Research Fellowship이 있다. EIE는 ISCA 50년 역사에서 인용 수 상위 5개 논문 중 하나로 선정되었다. 또한 효율적인 AI 기술을 실제 제품으로 연결하기 위해 DeePhi Tech와 OmniML을 공동 창업했으며, 두 회사는 각각 Xilinx(현재 AMD)와 NVIDIA에 인수되었다.

# MIT HAN Lab
[MIT HAN Lab](https://hanlab.mit.edu/)은 Song Han 교수가 이끄는 Efficient AI 연구실이다. 현재 연구실 홈페이지는 연구 목표를 강력하지만 계산량, 에너지, 확장 비용이 큰 foundation model을 algorithm-system co-design으로 더 효율적이고 실용적으로 만드는 것이라고 설명한다. 과거의 model compression과 TinyML에서 출발해, 현재는 LLM, VLM/VLA, diffusion model과 같은 generative AI까지 연구 범위를 확장했다.

HAN Lab의 연구는 크게 두 축으로 볼 수 있다.

- **Efficient AI Algorithm**: pruning과 sparsity, quantization, knowledge distillation, neural architecture search, efficient attention, 새로운 model architecture 등을 이용해 필요한 연산과 메모리를 줄인다.
- **Efficient AI Hardware & System**: 압축된 모델을 실제 장치에서 빠르게 실행할 수 있도록 kernel, inference/training engine, compiler와 accelerator를 함께 설계한다.

# 왜 하드웨어 가속과 Efficient ML이 필요한가?

## 1. AI가 요구하는 연산량이 하드웨어의 공급보다 빠르게 증가한다

![alt text](https://1drv.ms/i/c/01f9a177b0d453f2/IQR1dVBYLzAUTbhDdI9HKAHWAUmnlwCV2mIwUPLxNCorcQs?width=1231&height=624)
위 그림은 model size와 GPU memory의 증가 속도를 비교한다. Transformer 이후 모델의 parameter 수는 빠르게 증가했지만, 단일 가속기의 메모리 용량은 같은 속도로 커지지 않았다. 모델이 메모리에 들어가지 않으면 여러 GPU로 나눠야 하고, 이때 장치 사이의 통신 비용까지 발생한다. 따라서 더 큰 하드웨어만 기다리는 것으로는 부족하며, model compression을 통해 AI computing의 수요와 hardware computing의 공급 사이 간격을 줄여야 한다.

## 2. 연산뿐 아니라 data movement가 비싸다

신경망은 MAC 연산만 수행하는 것이 아니라 weight, activation과 KV cache를 메모리에서 반복해서 읽고 써야 한다. Song Han 교수는 [MIT 인터뷰](https://news.mit.edu/podcast/podcast-curiosity-unbounded-episode-18-inside-efficient-ai-gpus-gpts)에서 AI의 에너지 소비 원인을 compute와 data movement로 나누며, 특히 GPU와 GPU 사이, DRAM과 cache 사이에서 데이터를 옮기는 비용이 더 클 수 있다고 설명한다. Pruning은 불필요한 weight와 연산을 제거하고, quantization은 각 값을 표현하는 bit 수를 줄인다. 두 방법 모두 계산량뿐 아니라 메모리 사용량과 전송량을 함께 줄일 수 있다.

## 3. Edge와 TinyML 환경의 자원은 훨씬 제한적이다

Cloud GPU와 달리 휴대전화, 자율주행 장치, IoT sensor와 MCU에는 전력, 발열, 메모리와 저장 공간의 강한 제약이 있다. 강의 자료에서는 cloud, mobile, tiny device로 갈수록 사용 가능한 메모리 차이가 매우 커진다는 점을 보여준다. 이런 장치에서 AI를 실행하려면 모델을 작게 만드는 것뿐 아니라 제한된 SRAM 안에서 activation을 재사용하고, 장치가 잘 처리하는 연산으로 모델을 구성해야 한다.

그럼에도 on-device AI가 필요한 이유는 분명하다. 네트워크 연결 없이 동작할 수 있고, cloud 왕복 시간을 없애 실시간 응답이 가능하며, 사용자의 민감한 데이터를 장치 밖으로 보내지 않아도 된다. [MCUNet](https://hanlab.mit.edu/projects/mcunet)은 TinyNAS와 TinyEngine을 공동 설계해 상용 microcontroller에서 ImageNet-scale inference를 가능하게 만든 대표적인 예다.

## 4. 알고리즘의 이론적 효율을 실제 성능으로 바꿔야 한다

Efficient ML에서 목표는 단순히 parameter 수나 FLOPs가 작은 논문 속 모델을 만드는 것이 아니다. 실제 target hardware에서 latency, throughput, peak memory와 energy가 줄어야 한다. 같은 연산량이라도 병렬화하기 쉬운지, memory access가 연속적인지, 해당 precision과 sparsity를 하드웨어가 지원하는지에 따라 속도가 달라진다. 이것이 hardware-aware optimization과 algorithm-system co-design이 필요한 이유다.

# HAN Lab의 대표 연구 흐름

| 시기      | 대표 연구                                                                                                                                                                                         | 핵심 아이디어                                                                                                                                                                        |
| --------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| 2015-2016 | [Deep Compression](https://hanlab.mit.edu/projects/deep-compression), [EIE](https://hanlab.mit.edu/projects/retrospective-eie-efficient-inference-engine-on-sparse-and-compressed-neural-network) | Pruning과 quantization으로 model을 압축하고, sparse·low-precision model을 직접 가속하는 hardware를 설계했다.                                                                         |
| 2018-2020 | [ProxylessNAS](https://hanlab.mit.edu/projects/proxylessnas), [Once-for-All](https://hanlab.mit.edu/projects/ofa)                                                                                 | 측정된 hardware latency를 설계 과정에 넣고, 한 번 학습한 network에서 장치별 제약에 맞는 sub-network를 선택하도록 했다.                                                               |
| 2020-2022 | [MCUNet](https://hanlab.mit.edu/projects/mcunet), on-device training                                                                                                                              | TinyNAS와 TinyEngine의 공동 설계로 MCU의 작은 SRAM과 Flash에서도 inference와 training이 가능하도록 했다.                                                                             |
| 2023-2024 | [SmoothQuant](https://hanlab.mit.edu/projects/smoothquant), [AWQ](https://hanlab.mit.edu/projects/awq), [StreamingLLM](https://hanlab.mit.edu/projects/streamingllm)                              | LLM의 weight와 activation을 quantization하거나 attention sink와 최근 token만 KV cache에 유지해 memory bottleneck과 inference latency를 줄였다. AWQ는 MLSys 2024 Best Paper를 받았다. |
| 2025-2026 | Efficient video generation, reasoning LLM training, VLA                                                                                                                                           | Quantization과 sparsity의 대상을 weight에서 attention, KV cache, long video token과 RL rollout으로 확장하고 있다.                                                                    |

이 흐름을 보면 초기에는 weight sparsity와 low precision, TinyML에서는 network architecture와 memory scheduling, 최근 generative AI에서는 attention과 KV cache, token, training rollout이 주요 최적화 대상이 되었다.

# 2026년의 최신 연구

2026년 6월 기준 HAN Lab 홈페이지에 공개된 최근 연구들은 efficient AI의 범위가 inference compression을 넘어 training, video generation, robotics까지 확장되었음을 보여준다.

- [QeRL](https://hanlab.mit.edu/projects/qerl-beyond-efficiency----quantization-enhanced-reinforcement-learning-for-llms) (ICLR 2026)은 NVFP4 quantization과 LoRA를 결합해 LLM reinforcement learning의 rollout memory와 시간을 줄인다. 32B 모델을 H100 80GB 한 장에서 RL training할 수 있게 했으며, quantization noise를 단순한 오차가 아니라 exploration을 돕는 요소로 활용한다.
- [TLT](https://hanlab.mit.edu/projects/tlt) (ASPLOS 2026)은 reasoning model의 RL training에서 일부 sample의 매우 긴 generation이 전체 GPU를 기다리게 하는 long-tail 문제를 다룬다. 유휴 자원으로 adaptive draft model을 계속 학습하고 speculative decoding에 사용해 model quality를 유지하면서 end-to-end training을 1.7배 이상 가속한다.
- [SANA-Video](https://hanlab.mit.edu/projects/sana-video) (ICLR 2026 Oral)는 quadratic attention 대신 linear attention을 사용하고, 길이가 늘어나도 고정된 크기를 유지하는 KV cache를 설계해 minute-long video generation의 계산량과 메모리를 줄인다.
- [LongLive](https://hanlab.mit.edu/projects/longlive) (ICLR 2026)은 frame-level autoregressive 구조, short-window attention과 frame sink, KV-recache를 결합해 사용자가 prompt를 바꾸며 상호작용할 수 있는 장시간 video를 단일 H100에서 실시간으로 생성한다.
- [ForeAct](https://hanlab.mit.edu/projects/foreact) (CVPR 2026 Highlight)는 VLA(Vision-Language-Action) 모델 앞에 future observation을 예측하는 planner를 붙인다. 기존 VLA의 구조를 바꾸지 않고도 다음 시각 상태를 0.33초에 생성해 robot의 multi-step planning을 돕는다.

# 정리

Song Han 교수와 MIT HAN Lab이 말하는 Efficient ML은 단순히 작은 모델을 만드는 기술이 아니다. 모델의 정확도를 유지하면서 계산량과 데이터 이동을 줄이고, 그 구조를 실제로 활용할 수 있는 software와 hardware까지 함께 설계하는 분야다. 이를 통해 cloud에서는 같은 자원으로 더 많은 요청을 처리하고, edge에서는 privacy와 real-time성을 지키며, MCU처럼 매우 작은 장치에서도 AI를 사용할 수 있다.

앞으로 강의에서 다룰 pruning, quantization, knowledge distillation, neural architecture search는 각각 다른 방법처럼 보이지만, 모두 **AI가 요구하는 자원과 hardware가 제공할 수 있는 자원 사이의 차이를 줄인다**는 하나의 목표로 연결된다.
