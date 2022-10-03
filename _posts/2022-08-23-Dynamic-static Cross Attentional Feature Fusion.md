---
title: Dynamic-static Cross Attentional Feature Fusion Method for Speech
Emotion Recognition
author: adria
date: 2022-08-12 11:33:00 +0800
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


## Links
This paper is under peer review.

## Reference
NULL.
