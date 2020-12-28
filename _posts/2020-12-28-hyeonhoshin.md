---
title: Deep-Learning feature extractor&descriptor based Visual Odometry
author: Hyeonho, Shin
date: 2020-12-28 12:22:00 +0900
categories: [Exhibition,2020년]
tags: [post,hyeonho-shin,deep-learning,computer-science]
---

------------------------------------------ 

## Abstract

Welcome to brief description of DeepPoint VO, final assignment program designed by Hyeonho shin.

Our submitted program is designed to know to how much accuracy can be improved using the latest VO algorithms for beating SOTA in Visual monocular odometry algorithm, BVO[1].

To begin with, we implemented six algorithms and various parameters to compare. As a result, the key point algorithm was ‘LFNet’ with soft ratio-test.

Without additional training about KITTI dataset, our VO beats SOTA in median. Also we found the counter-well-known facts about ratio test.

Source code repository : <a href="https://github.com/hyeonhoshin/tanuki-pyslam">https://github.com/hyeonhoshin/tanuki-pyslam </a>

## Design
We reused the vanilla visual odometry framework except for Deep Learning based key point extractor and descriptor.

In detail, we used Brute Force feature matcher, because there is no elapsed time difference between FLANN matcher. Also, FLANN matcher has a danger to fall into local minima.

For reducing outliers of matching results, we used ratio test and RANSAC algorithm. The ratio test is similar to Voting algorithm. If 2nd good matching pair’s distance is too different with 1st one, we neglect this result. The RANSAC eliminates outliers, by probabilistic finding.

## Experiment condition
### Hardware
* CPU : Intel Xeon Silver 4214
* RAM : 256GB
* Graphics : NVIDIA RTX2080ti

### Software
* Ubuntu 18.04 (20 not supported)
* Bash shell
* Anaconda

### Parameters
* Feature number limit : 1000
* Ratio test coef : 0.99
* Non maxima suppression radius : 7 pixels
* No train with KITTI sets. Used the pretrained model.

## Result
### Accuracy comparison among Key point algorithms
<table>
    <tr>
        <td></td>
        <td>ORB2</td>
        <td>DELF</td>
        <td>R2D2</td>
        <td>D2NET</td>
        <td>LFNET</td>
    </tr>
    <tr>
        <td>Translation error(%)</td>
        <td>16.80</td>
        <td>83.09</td>
        <td>1.06</td>
        <td>1.69</td>
        <td>0.68</td>
    </tr>
    <tr>
        <td>Rotation error(%)</td>
        <td>5.49</td>
        <td>37.18</td>
        <td>0.41</td>
        <td>0.70</td>
        <td>0.59</td>
    </tr>
</table>

With above table 1, we can see the LFNET has the best performance among the famous SOTA key point extraction and description algorithms.

### Speed

<table>
    <tr>
        <td>The number of features</td>
        <td>100</td>
        <td>250</td>
        <td>500</td>
        <td>1000</td>
        <td>2000</td>
        <td>3000</td>
    </tr>
    <tr>
        <td>Estimated FPS</td>
        <td>31.16</td>
        <td>25.95</td>
        <td>18.72</td>
        <td>13.69</td>
        <td>8.81</td>
        <td>3.70</td>
    </tr>
    <tr>
        <td>Translation error in Seq 07(%)</td>
        <td>14.43</td>
        <td>5.644</td>
        <td>5.479</td>
        <td>5.07</td>
        <td>4.96</td>
        <td>3.99</td>
    </tr>
</table>

Because of the Python’s multi-thread limitation and IO bottleneck by using libraries(Torch-Caffe-NumPy), the total speed is under-estimated significantly. So we used the pre-made tool for estimating the FPS which considers feature key points extraction, description and matching. Because they take the most proportion of visual odometry, we thought it would be valid and reliable.
We picked the key points 1000 only for analysis. Because it has more than 10FPS, which makes it possible to do real-time with KITTI’s dataset. The KITTI’s dataset has 10fps, and in the 1000 key points case the fps is higher than KITTI’s one.

### Accuracy
<table>
    <tr>
        <td>Trans err (%)</td>
        <td>00</td>
        <td>01</td>
        <td>02</td>
        <td>03</td>
        <td>04</td>
        <td>05</td>
        <td>06</td>
        <td>07</td>
        <td>08</td>
        <td>09</td>
        <td>10</td>
        <td>median</td>
    </tr>
    <tr>
        <td>Ours</td>
        <td>1.15</td>
        <td>3.72</td>
        <td>1.90</td>
        <td>0.86</td>
        <td>1.32</td>
        <td>0.68</td>
        <td>5.07</td>
        <td>1.75</td>
        <td>1.39</td>
        <td>0.85</td>
        <td>1.32</td>
        <td></td>
    </tr>
    <tr>
        <td>BVO</td>
        <td>1.99</td>
        <td>2.14</td>
        <td>1.47</td>
        <td>0.93</td>
        <td>0.88</td>
        <td>0.92</td>
        <td>1.85</td>
        <td>1.61</td>
        <td>1.38</td>
        <td>1.24</td>
        <td>1.27</td>
        <td>1.38</td>
    </tr>
</table>

As you can see above table, we can beat the SOTA BVO in median. In detail, we beated BVO in sequece 00, 03, 04, 06, 10. Specifically, in 06, we can outperform 3 times in the error. On the other hand, we get the bad result in 07, about 5 times.
Our analysis is guessed that our algorithm is more fragile to speed variation of the car. The sequence 06 keep speed consistently, but luminance change is large. In contract, in sequence 07, speed change is more variant. It includes waiting, stopping, accelearating, slowing.

## Conclusion
In this project, we designed the visual odometry algorithm assisted by Deep-Learning based Key point detection and description. And it outperforms in some sequences by accuracy without additional traing about KITTI dataset.
What we revealed in this project is the fact that ratio test is needed for implementing VO. By many authors including LFNet’s authors, without ratio test, the VO’s stablity is degraded. And by trials and errors, we present our the best parameter for ratio test also. The best results occurs at 0.99.
In addition, we revealed the fact that do not set scale factor. That makes the Neural Networks distribution, and make degradation for it.
Also among 6 SoTA Kp detection & description(LFNet, D2Net, SuperPoint, R2D2, DELF), The best performance algorithm is **LFNet**. But the fastest algorithm with reasonable performance is SuperPoint.

## How to set Pre-requisites
Because our alogrithms depends on a lot of external library. I’ll give you a program which can make same environments wit ours.
- Install anaconda (https://www.anaconda.com/products/individual)
- Download our program using git recursively ( “git clone --recursive
https://github.com/hyeonhoshin/tanuki-pyslam”, because it needs to download sub-libraries )
- Change directory to the downloaded ‘tanuki-pyslam’
- In terminal at the ‘~’ directory, type “. install_all_conda.sh”
- If the setting program demands the sudo password, type it.
- All done.

## How to run our program
- Change directory into ‘tanuki-pyslam’
- Activate virtual Python environment by typing ‘conda activate pyslam’
- Input your dataset directory and output directory in config.ini
(In KITTI dataset, base_path=/home/user/....../dataset, output_path=/as/you/want but in default the
results are recorded in outputs folder)
- Specify which sequence to test in config.ini (For example, name=06 means 6th sequence in KITTI set)
- Execute our program by typing ‘python main_vo.py’
- Check the generated file “XX.txt” which includes R, t

## How to see real-time progress
If you want to see real-time progress same as included video file, just modify gui_on as True. But, if you cannot use Monitor, it makes our code do not operate.

## How to evaluate my program
- Change directory into ‘~’
- Clone the KITTI Visual odometry evaluation tool by typing “git clone
https://github.com/Huangying-Zhan/kitti-odom-eval”
- Change directory to the cloned directory, by typing “cd kitti-odom-eval”
- For installing dependencies, type “conda env create -f requirement.yml –n kitti_eval”
- Activate the installed virtual environment by typing, “conda activate kitti_eval”
- For executing the evaluation program, type “python eval_odom.py —result RESULT_PATH”, which includes result files from the step of “How to run our program”

## References
[1] Fabio Irigon Pereira, 2017 Workshop of Computer Vision, “Backward Motion for Estimation Enhancement in Sparse Visual
Odometry”
[2] Y. Ono, E. Trulls, P. Fua, K.M. Yi, "LF-Net: Learning Local Features from Images"
[3] Luigifreda, github, Pyslam, https://github.com/luigifreda/pyslam.
[4] Dusmanu, CVPR 2019, “D2-Net: A Trainable CNN for Joint Detection and Description of Local Features”
[5] Axel Barroso-Laguna, ICCV 2019, "Key.Net: Keypoint Detection by Handcrafted and Learned CNN Filters“
[6] Zhan, arXiv 2019, “Visual Odometry Revisited: What Should Be Learnt?”
