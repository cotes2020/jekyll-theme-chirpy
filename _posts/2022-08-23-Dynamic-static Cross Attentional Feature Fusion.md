---
title: Dynamic-static Cross Attentional Feature Fusion Method for Speech Emotion Recognition
author: adria
date: 2022-9-3 11:33:00 +0800
categories: [Publications]
tags: [Speech Emotion Recognition, Attention Mechanism, Feature Fusion]
math: true
mermaid: true
image:
  path: /commons/acoustic.png
  width: 800
  height: 500
  alt: Figure 2. Extraction of Acoustic Features
---

## Abustruct


The dynamic-static fusion features play an important role in speech emotion recognition (SER). However, the fusion methods of dynamic features and static features generally are simple addition or serial fusion, which might cause the loss of certain underlying emotional information. To address this issue, we proposed a dynamic-static cross attentional feature fusion method (SD-CAFF) with a cross attentional feature fusion mechanism (Cross AFF) to extract superior deep dynamic-static fusion features. To be specific, the Cross AFF is utilized to parallel fuse the deep features from the CNN/LSTM feature extraction module, which can extract the deep static features and the deep dynamic features from acoustic features (MFCC, Delta, and Delta-delta). In addition to the SD-CAFF framework, we also employed muti-label auxiliary learning in the training process to further improve the accuracy of emotion recognition. The experimental results on IEMOCAP demonstrated the WA and UA of SD-CAFF are 75.78% and 74.89%, respectively, which outperformed the current SOTAs. Furthermore, SD-CAFF achieved competitive performances (WA: 56.77%; UA: 56.30%) in the comparison experiments of cross-corpus capability on MSP-IMPROV. Finally, we employed confusion matrix, t-SNE, and a series of ablation experiments to verify the effectiveness of SD-CAFF and the necessity of each module.

## Display

<img src="/commons/sd-caff/ser.png" alt="ser" title="The process of SER">
<img src="/commons/sd-caff/structure.png" alt="structure" title="The structure of SD-CAFF">
<img src="/commons/sd-caff/cross att.png" alt="cross att" title="The cross att">
<img src="/commons/sd-caff/res1.png" alt="res1" title="res1">
<img src="/commons/sd-caff/res2.png" alt="res2" title="res2">

## Links
https://link.springer.com/chapter/10.1007/978-3-031-27818-1_29

## Reference
Dong, Ke, Hao Peng, and Jie Che. "Dynamic-Static Cross Attentional Feature Fusion Method for Speech Emotion Recognition." International Conference on Multimedia Modeling. Cham: Springer Nature Switzerland, 2023.

## Download
<a href="/commons/pubs/camera_ready.pdf">SMB.pdf</a>
