---
title: Performance of Symbolic & Connectionist AI in Image Recognition
author: Max van Dijck
date: 2021-06-17 08:55:00 +1200
categories: [Research]
tags: [DeepLearning, FastAI, Research]
---
Neural network approaches in the field of artificial intelligence (AI) have accelerated in popularity over the past decade in the academic field. This surge has arguably left behind import concerns in intelligent machines such as interpretability and robustness. This raises questions around the trade-off of performance and aforementioned concerns when choosing between symbolic and connectionist approaches to solving AI problems. This study explores three different models, Deep learning, K-Nearest Neighbour and Support Vector Machines across two separate image datasets. The intent of this is to evaluate the performance of image recognition approaches across methods of different interpretability when approaching classification challenges of high-dimensional data. The study found that symbolic approaches can compete well within the range of Deep Learning capabilities on datasets with a large volume of training data and lower dimensionality. However, when faced with little training data a transfer learning approach significantly outperformed the other methods. A downside to K-NN was exhibited in its inference time where it took significantly longer than the other methods to label the test data.

## Introduction

The purpose of this research is to provide a foundation argument for why the consideration of a range of approaches to data driven problems should be a necessary step for machine learning practitioners and data scientists to take. In the first instance one may look at a machine learning prediction problem and choose the highest accuracy approach, this is fostered by platforms such as Kaggle (Kaggle Inc, 2021) and competitions such as the ImageNet Classification Challenge (Stanford Image Lab, 2020). While this encourages the development and use of State-of-the-Art techniques little is done to question the interpretability of the results. While this is not necessarily a bad thing, one may argue that the reliance upon black-box models may prove consequential for the field in the event of a paradigm shift or regulation. 

This study compares multiple instances of AI approaches and aims to identify possible trajectories that should be explored in situations where AI is not merely used as a tool to produce the most accurate model but where the model may have several drawbacks regarding the impact on its environment or users. Three types of techniques, Deep Learning, Support Vector Machines and K-Nearest Neighbour will be evaluated by their performance across two image datasets of two separate complexities and sizes. This aims to highlight the ability of each model on challenging high-dimensional data and evaluate how each model’s performance drops when the number of training samples lower and the dimensionality further increases. 

## Literature Review

Machine learning has become a highly popular field in recent years due to factors not limited to but including; the abundance of cheap computing, the mass generation and availability of data and the creation of open access tools and methods (Brownlee, 2020) (Dean, 2019).  This growth is also commonly attributed to artificial intelligences high exposure in the media with claims that AI is either “Impossible” or “Just around the corner” (Allen, 1998) alongside its increase in usage in a wide range of disciplines including science, business, commercial settings, industry, engineering, entertainment and government (Bunz, 2019). However, rushed deployment of AI has resulted in cases where “the technology (has) not being tested thoroughly before-hand, which is leading to cases of injustice often related to machine bias” – Bunz. This is clear and harmful in multiple instances. For example, Colorado previously implemented more than 900 incorrect rules into its public benefits system causing pregnant woman to be denied Medicaid. Or in 2011 where Idaho implemented an algorithm for allocating home care and community integration funds that then dropped funds for the severely disabled community by as much as 42 percent (Institute for Healthcare Policy & Innovation, 2018).

Due to the wide usage and commonality of connectionist AI in society, it becomes concerning that these algorithms are becoming increasingly complicated and uninterpretable over time (Christian Berghoff, 2020) but that these changes are driven by creating solutions to further uninterpretable problems such as Adversarial Image Attacks (Ian J. Goodfellow, 2015) and Trojaning Atta¬¬cks (Yingqui Liu, 2018). Solutions to these problems include creating Sparse Representation Networks (Yiwem Gui, 2018) and Network Ensembles (Tao, 2019) which are comprised of larger or multiple networks which further complicate the issue of interpretability and explainability. The concerns of society towards these algorithms is evident as international organisation, The United Nations, has acted to restrict unexplainable automated decision-making of which the user will be significantly affected (Brynce Goodman, 2016). The law creates a “right of explanation” where users can ask for explanations of decisions made about them. Due to this demand of algorithmic transparency certain use cases of machine learning approaches and other high-dimensionality algorithms are in a grey area when it comes to widespread use cases especially in medical and governmental use cases. Given the pressure from society and the combination of governments and organizations towards implementers of these algorithms. One must begin to look at the trade-off between creating interpretable, explainable AI with lower performance and accuracy compared to their complex unexplainable counterparts.

## Datasets
### MNIST Handwritten Digits

The MNIST handwritten digit dataset (Yann LeCun, n.d.) is a common benchmark model for comparing algorithms. It consists of 70,000 28x28 pixel greyscale images of the digits 0-9. For this model evaluation there will be 60,000 images for the training set and 10,000 remaining images for the test set. This dataset was chosen for its low dimensionality compared to other image datasets and it’s high volume of training and testing samples.

### NZ Native Birds

The NZ Native Birds dataset is a self-collated image dataset of varying bird species. The dataset is purposefully created to be a challenging high-resolution, low training data dataset containing 10 categories with approximately 150 images per category. This high complexity is designed to challenge the different approaches of each model and provide a platform to easily analyse shortcomings of said models. The images were collected using Microsoft Azures Bing Image Search and cleaned by initially training a deep learning network and using said network to identify images with higher loss. These images with higher loss were then inspected and removed or recategorized accordingly. Dataset is available here: [**NZ Native Birds**](https://drive.google.com/file/d/1-DNgRR16OJsoJDbPxiW87Ja-SklRh-mW/view?usp=sharing)

## Image Recognition Models

The focus of this research is to compare different AI models on the aforementioned datasets. As such, a short description of each model followed by an in-depth explanation into their respective architectures used in this study will be provided.

### Deep Learning & Transfer Learning

Deep learning comes from the connectionist approach of recreating the biological process the human brain uses for learning. This is achieved through creating layers of artificial neurons which are connected to previous and subsequent layers. Each neuron has its own weight and bias which is applied to the output of the previous layer, this process continues through all layers of the network until it reaches the final layer where each neurons output represents a probability of a certain outcome. This approach is reliant upon the Universal Approximation Theorem proven in 1991 by Kurt Hornik (Hornik, 1991) which states that the result of the first layer of a neural network can approximate any well-behaved function, this function can also be approximated by a network of greater depth. 

There are two Deep Learning models used in this study. Both models use the same ResNet50 (residual network) architecture first published in 2015 (Kaiming He, 2016). The ResNet architecture is designed in a way to prevent vanishing and exploding gradients, this is caused by the exponential increase or decrease of gradients due to training with stochastic gradient descent. These gradients can cause the model to either become unable to learn or cause the model to get stuck in a local minimum unable to find the global minimum to approximate the target function correctly (Seshapanpu, 2020). The introduction of skip connections in the ResNet architecture greatly reduces these gradient issues and allows the training of much more complex, deeper networks. The effect of these skip layers is best illustrated in Hao Li et al.’s  "Visualizing the Loss Landscape of Neural Nets" (Hao Li, 2017). Li shows that using skip connections smooths the loss function which makes training easier as it avoids many local minimums.
![Training Landscape Graphs](/assets/img/2021-06-17/training_landscape.png)
_Fig 1. Visualised CNN training landscape (left) ResNet training landscape (right) (Hao Li, 2017)_
The difference between the two models is that the first model will have used transfer learning and the second model will be trained completely from scratch. Transfer learning is the act of taking a pretrained model, in this case, on a large dataset such as ImageNet and adjusting the model to fit to a new dataset. This adjustment is achieved by preserving all the high-quality edge detection in the early layers using a lower learning rate and retrains the ‘deepest’ layers of the network with a higher learning rate.

### K-Nearest Neighbour

The K-nearest Neighbour algorithm (k-NN) was first introduced by Fix & Hodges as a classification method in 1951 (Evenlyn Fix, 1951). The method works by calculating the distance between datapoints in the classification target and the training data, the classification target can then be labelled according the surrounding data’s labels taking in account a certain number of surrounding data (k = 𝑥). This algorithm has a few notable advantages. Firstly, there is no need to build a model with many parameters to tune. Secondly, k-NN is simple to implement and interpret the choices the model makes. A drawback of the k-NN algorithm arises in high-dimensional data whereas the number of dimensions and classification options increase the distances become close to equal lowering the reliability and accuracy of the classifier.

For image detection we will be taking the difference between values of each pixel in the training-set image and the classification image to then find the average pixel difference for the image. This process is repeated across the training set to find the most similar image(s). In this study the closest one, three and five (k = 1, 3, 5) to classify images will be used.

### Support Vector Machines

Support Vector Machines (SVMs) for non-linear classification were developed in 1992 when Vladimir Vapnik introduced kernel methods to maximum-margin hyperplanes (Vladimir Vapnik, 1992). This kernel trick allows the mapping of training data to be translated into higher dimensions and for them to be calculated in a method which allows for a hyperplane to be implemented which divides the different categories in these new dimensions. These SVMs generally contain tens or hundreds of kernel tricks in high dimensionality data such as images. Support Vector Machines are used in this study to demonstrate the performance of a fair middle ground between symbolic and connectionist approaches to solving large datasets.  

## Results

The results of this study are judged on model performance on the two styles of datasets, lower or higher dimensionality and higher or lower training data respectively alongside their respective training and inference time.

### Handwritten Digits

The MNIST handwritten digits showed that Symbolic methods can compete reliably with more data. The standard deviation of the accuracy of the models was 8.60e-3 and had a ±0.7% margin of error given a 95% confidence level.

Despite deep learning and transfer learning still coming across with the highest accuracy of the methods, one should consider the trade-off of using a Symbolic method in areas where algorithmic explainability is important. Notably, KNN still performed well within a competitive range for this dataset.
![Handwritten Accuracies](/assets/img/2021-06-17/handwritten_accuracy.png)
_Fig 2. Model accuracies on the MNIST dataset_

### NZ Native Birds

Due to the lack of training data, the NZ Native Birds dataset highlights the strengths of having a transfer learning approach. The standard deviation of this dataset’s accuracies is 0.240, assuming a 95% confidence level the models had a ±44.4% margin of error.

As shown in Figure 3, Transfer Learning outperformed the rest of the models by more than 40%. Since machine learning is such a data driven practice it becomes important to have an already robust model before training to a such specific dataset. One must also look at the results with the understanding that no image augmentations or dimensionality reduction had taken place as it is outside the scope of this research.
![Bird Accuracies](/assets/img/2021-06-17/birds_accuracy.png)
_Fig 3. Model accuracies on the Birds dataset_

### Summary
Overall, the performance of Symbolic approaches on data that is of a high concentration and of a reasonable dimensionality remains relevant. Data science and machine learning practitioners should remain diligent and responsible by considering all approaches to solving problems rather than choosing models with many parameters especially in areas where users are greatly affected e.g., Healthcare or Government.

One major drawback of KNN however is its inference time. Generally, machine learning models are expected to have high training costs but can then be deployed onto low-cost infrastructure or directly onto low powered devices. Since, at inference time, KNN compares each training set image to the test image the inference time is directly proportional to the size of the training dataset. This causes issues on high-bandwidth data such as video streams. This effect is shown in Fig 6, training time is not shown due to KNN not requiring any changes to the training dataset or creating a prediction model off the data.

![Both Dataset Accuracies](/assets/img/2021-06-17/overall_accuracy.png)
_Fig 4. Model results comparatively across datasets_
![Training Times](/assets/img/2021-06-17/training_time.png)
_Fig 5. Model training time in seconds_
![Inference Times](/assets/img/2021-06-17/inference_time.png)
_Fig 5. Model inference time in milliseconds_

## Further Research
Artificial Intelligence is an ever-expanding field, while the popularity grows the importance of data ethics, privacy and algorithmic integrity increases with it. This research has attempted to outline the importance of these issues and to look at multiple approaches of different complexities and interpretability. However, in comparison to the wide range of techniques in the field of artificial intelligence the study is rather limited by the three models and the two datasets used. Further research may look at expanding the scope of algorithms and datasets, for example moving towards other types of data e.g., tabular, text or a combination of types. Additionally, there is clear evidence that concern over AI is growing and regulatory measures are starting to expand to limit the use of certain approaches, this suggests the importance of revisiting more traditional approaches to problems and further improving algorithms from a symbolic perspective.

Future research may also look at integrating both Symbolic and Connectionist in a Neuro-symbolic approach such as combining decision trees and neural networks such as the approach that Animesh Garg took towards creating generalizable autonomy for robotic applications (Garg, 2020). This has the potential to bring together the strong performance of deep learning with the clear desired outputs that symbolic methods provide (Garcez & C. Lamb, 2020).

## References

Allen, J. F. (1998, December 15). AI Growing Up: The changes and Opportunities. AI Magazine, pp. 13-23.

Brownlee, J. (2020, December 10). Machine Learning is Popular Right Now. Retrieved from Machine Learning Mastery: https://machinelearningmastery.com/machine-learning-is-popular/#:~:text=Machine%20learning%20is%20popular%20because,capability%20of%20machine%20learning%20methods.&text=There%20is%20an%20abundance%20of%20data%20to%20learn%20from.

Brynce Goodman, S. F. (2016, August 31). European Union Regulations on Algorithmic Decision-Making and a “Right to Explanation”. AI Magazine, pp. 50-57. doi:10.1609/aimag.v38i3.2741

Bunz, M. (2019). The calculation of meaning: on the misunderstanding of new artificial intelligence as culture. Culture, Theory and Critique, 60, pp. 264-278. doi:10.1080/14735784.2019.1667255

Christian Berghoff, M. N. (2020, July 22). Vunerabilities of Connectionist AI Applications: Evaluation and Defense. Front. Big Data. doi:10.3389/fdata.2020.00023

Dean, J. (2019). The Deep Learning Revolution and Its Implications for Computer Architecture. CoRR, abs/1911.05289. Retrieved from http://arxiv.org/abs/1911.05289

Evenlyn Fix, J. H. (1951, February). Nonparametric Discrimination Consistency Properties. International Statistical Review / Revue Internationale de Statistique, 57, pp. 238-247. doi:https://doi.org/10.2307/1403797

Garcez, A. d., & C. Lamb, L. (2020). Neurosymbolic AI: The 3rd Wave. arXiv(arXiv:2012.05876).

Garg, A. (2020, March 28). MIT 6.S191 (2020): Generalizable Autonomy for Robot Manipulation. Retrieved from Youtube: https://www.youtube.com/watch?v=8Kn4Gi8iSYQ&list=PLtBw6njQRU-rwp5__7C0oIVt26ZgjG9NI&index=20&ab_channel=AlexanderAminiAlexanderAmini

Hao Li, Z. X. (2017). Visualizing the Loss Landscape of Neural Nets. CoRR, abs/1712.09913. Retrieved from http://arxiv.org/abs/1712.09913
Hornik, K. (1991). Approximation Capabilities of Multilayer Feedforward Networks. Neural Networks, 4(0893-6080), pp. 251-257. doi:https://doi.org/10.1016/0893-6080(91)90009-T

Ian J. Goodfellow, J. S. (2015, March 20). Explaining and Harnessing Adversarial Examples. CoRR, abs/1412.6572.

Institute for Healthcare Policy & Innovation. (2018, March 21). What happens when an algorithm cuts your health care.

Kaggle Inc. (2021). Kaggle: Your Machine Learning and Data Science Community. Retrieved from Kaggle: https://www.kaggle.com/

Kaiming He, X. Z. (2016). Deep Residual Learning for Image Recognition. 2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pp. 770-778. Retrieved from http://arxiv.org/abs/1512.03385

Seshapanpu, J. (2020, April 16). Vanishing and Exploding Gradients in Neural Networks. Retrieved from Medium: https://medium.com/@sjakki/vanishing-and-exploding-gradients-in-neural-networks-5b3ee108a568

Stanford Image Lab. (2020). ImageNet Large Scale Visual Recognition Challenge (ILSVRC). Retrieved from ImageNet: https://www.image-net.org/challenges/LSVRC/#:~:text=The%20ImageNet%20Large%20Scale%20Visual,image%20classification%20at%20large%20scale.&text=Another%20motivation%20is%20to%20measure,indexing%20for%20retrieval%20and%20annotation.

Tao, S. (2019, August 13). Deep Neural Network Ensembles. LOD.

Vladimir Vapnik, B. B. (1992, July). A Training Algorithm for Optimal Margin Classifiers. Proceedings of the fifth annual workshop on Computational learning theory, pp. 144–152. doi:10.1145/130385.130401
Yann LeCun, C. C. (n.d.). The MNIST Database. Retrieved from Lecun: http://yann.lecun.com/exdb/mnist/index.html

Yingqui Liu, S. M.-C. (2018, January). Trojaning Attack on Neural Networks. doi:10.14722/ndss.2018.23300

Yiwem Gui, C. Z. (2018). Sparse DNNs with Improved Adversarial Robustness. NeurIPS.
