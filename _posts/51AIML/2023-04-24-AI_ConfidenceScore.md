---
title: AIML - Confidence score for ML model
date: 2023-04-24 11:11:11 -0400
description:
categories: [51AIML]
# img: /assets/img/sample/rabbit.png
tags: [AIML]
---

- [Confidence score for ML model](#confidence-score-for-ml-model)
  - [overall](#overall)
  - [Confidence estimation techniques](#confidence-estimation-techniques)
  - [Data labeling](#data-labeling)
    - [蒸馏法](#蒸馏法)
    - [LLM data labeling benchmark](#llm-data-labeling-benchmark)
      - [Exam 1: Test Methodology](#exam-1-test-methodology)
        - [Setup](#setup)
        - [Results](#results)
        - [Qualitative Evaluation](#qualitative-evaluation)
      - [Exam 2: Language Models (Mostly) Know What They Know](#exam-2-language-models-mostly-know-what-they-know)
      - [Estimating Confidence with Autolabel](#estimating-confidence-with-autolabel)
      - [Confidence estimation techniques](#confidence-estimation-techniques-1)
        - [P(True)](#ptrue)
        - [Prompting for Confidence Score](#prompting-for-confidence-score)
        - [Token probabilities](#token-probabilities)
        - [Entropy 熵](#entropy-熵)



---

# Confidence score for ML model

---

## overall

> The main purpose of any confidence indicator, be it a quantitative score or a qualitative signal, is to highlight potential uncertainties for human review.

Confidence scoring mechanisms for language models (LLMs) refer to methods used to estimate the model's level of certainty or confidence in its predictions. Here are some common confidence scoring mechanisms for LLMs:

**Probability Scores**:
- LLMs often provide probability scores or confidence scores for each prediction.
- These scores represent the `model's estimated probability that a particular output is correct`.
- High probability scores indicate high confidence, while lower scores suggest lower confidence.

**Entropy**:
- Entropy is a measure of `uncertainty or disorder` in a set of probabilities.
- In the context of LLMs, entropy can be used to quantify how confident the model is in its predictions.
- Lower entropy values indicate higher confidence, as the model is more certain about its predictions.

**Margin-based Confidence**:
- Margin-based confidence considers `the difference in probability between the top prediction and the second-best prediction`.
- A larger margin implies higher confidence.
- This approach is common in classification tasks, where the model selects the class with the highest probability.

**Calibration**:
- Calibration ensures that `the predicted probabilities align with the actual likelihood of correctness`.
- Well-calibrated models provide accurate confidence estimates.
- Calibration plots can be used to visualize the relationship between predicted probabilities and actual outcomes.

**Out-of-Distribution Detection**:
- Confidence scores can be used to detect out-of-distribution (OOD) samples.
- If a model encounters data significantly different from its training distribution, it might output lower confidence scores, indicating uncertainty.

**Uncertainty Quantification**:
- Some methods explicitly aim to quantify uncertainty.
- Bayesian approaches, for example, use probability distributions over model parameters to represent uncertainty in predictions.
- `Variational Inference` and `Monte Carlo Dropout` are techniques that fall into this category.

**Auxiliary Training Objectives**:
- Training objectives can be designed to encourage the model to output more informative confidence scores.
- For example, models can be trained to maximize the expected calibration error.

**Self-Assessment**:
- Some models incorporate self-assessment mechanisms, allowing the model to evaluate its own performance.
- This can involve comparing its predictions to the ground truth during training.



Traditional machine learning (ML) or layout-based models rely on quantitative confidence scores, which are based on fixed domains, use confidence metric to assess the accuracy of the extraction process.
- For example, a key-value pair extraction is scored by how certain the model is that the extracted value for the `first_name` field is the closest adjacent name, considering the character recognition and the document’s layout.

For LLM
- the natural language prompts are open-ended and lack a clear output type or domain.
- For instance, if you query `calculate document word count`, there is no specific location where the LLM can find that information, as it performs the word count itself.
- Assigning a numerical confidence score would be asking the model to rate its own performance, which is inherently subjective 主觀的.
- Instead, instructs the LLM to identify any common sources of uncertainty present in the answer, and the LLM responds with the corresponding signal.



LLM's' confidence about an answer, with an exhaustive list of considerations, including:

- `Partial answer found`: an answer is produced, but the LLM isn’t confident that it fully addresses the query
  - To return multiple answers, use the List method.
  - To return a single answer, ensure the context contains a single answer using Advanced prompt configuration.

- `Multiple answers found`: an answer is produced, but the LLM has identified multiple answers that could work
  - Simplify the prompt, for example, break it up into multiple prompts.

- `No answer found, query too ambiguous`: the LLM is unable to identify an answer because of the prompt’s ambiguity
  - Advanced prompt configuration.

- `Answer found`: the LLM is confident about the produced answer, and will be able to successfully reproduce the extraction across varying document types

- `No answer found`: an answer cannot be produced from the context

---

## Confidence estimation techniques

different techniques for estimating confidence of LLM generated labels:

- **Token-level generation probabilities**
  - commonly referred to as “logprobs”
  - by far the most accurate technique for estimating LLM confidence.

- Explicitly prompting the LLM to output a confidence score, while popular, is highly unreliable.
  - This technique had the lowest accuracy and the highest standard deviation across datasets.
  - More often than not the LLM just hallucinates some number.

- Autolabel, open-sourced library


---


## Data labeling


数据标注
- 这是需要投入最多时间和资源的步骤。

数据标注可以使用多种方法(或方法组合)来完成，包括:
- 内部：利用现有的人员和资源。虽然可以更好地控制结果，但可能既耗时又昂贵，特别是如果需要从头开始雇用和培训注释者。
- 外包雇佣临时的自由职业者来标记数据：将能够评估这些承包商的技能，但对工作流组织的控制将会减少。
- 众包：可以选择使用可信的第三方数据合作伙伴众包的数据标签需求，如果没有内部资源，这是一个理想的选择。数据合作伙伴可以在整个模型构建过程中提供专业知识，并提供对大量贡献者的访问，这些贡献者可以快速处理大量数据。对于那些希望大规模部署的公司来说，众包是理想的选择。
- 用机器：数据标注也可由机器完成。应该考虑机器学习辅助的数据标记，特别是当必须大规模准备训练数据时。它还可以用于自动化需要数据分类的业务流程。

质量保证(QA)
- 质量保证是数据标注过程中经常被忽视的关键组成部分。
- 数据上的标签必须满足许多特征;它们必须信息量大、独特、独立。
- 标签也应该反映出准确的真实程度。
- 例如，在为自动驾驶汽车标记图像时，必须在图像中正确标记所有行人、标志和其他车辆，以使模型成功工作。

培训和测试
- 一旦为训练标记了数据，并且通过了QA，那么就是时候使用这些数据来训练的AI模型了。
- 从那里，在一组新的未标记数据上测试它，看看它做出的预测是否准确。
- 根据模型的需求，将对准确性有不同的期望。如果的模型正在处理放射学图像以识别感染，则精度级别可能需要高于用于识别在线购物体验中的产品的模型，因为这可能是生死攸关的问题。相应地设置的自信阈值。

利用Human-in-the-loop
- 当测试的数据时，人类应该参与到提供地面真相监测的过程中。
- 利用human-in-the-loop允许检查的模型是否正在做出正确的预测，识别训练数据中的差距，向模型提供反馈，并在做出低置信度或不正确的预测时根据需要重新训练它。

规模
- 创建灵活的数据标记流程，使能够进行扩展。随着的需求和用例的发展，期望对这些过程进行迭代。


Next generation data labeling tools
- like `Autolabel`
- leverage LLMs to create `large, diverse labeled datasets`.
- In an application domain like labeling where correctness is critical, it is imperative to be able to accurately estimate the model’s level of confidence in its own knowledge and reasoning.
- Doing so enables us to automatically reject low confidence labels, ensemble LLMs optimally, and learn more about strengths and weaknesses of any given LLM.

- example
  - Toxic comments classification dataset [^Datasets_civil_comments] from the labeling benchmark.
  - If are able to estimate the LLM’s confidence level alongside the labels it generates, can calibrate the model’s label quality (% agreement with ground truth labels) at any given confidence score.
  - then decide an operating point (`confidence threshold`) for the LLM, and reject all labels below this threshold.

[^Datasets_civil_comments]: Datasets:civil_comments, https://huggingface.co/datasets/civil_comments

![Screenshot 2023-11-13 at 16.46.44](/assets/img/Screenshot%202023-11-13%20at%2016.46.44.png)

---

### 蒸馏法

What do LLMs Know about Financial Markets? A Case Study on Reddit Market Sentiment Analysis [^What_do_LLMs_Know_about_Financial_Markets?]

[^What_do_LLMs_Know_about_Financial_Markets?]: What do LLMs Know about Financial Markets? A Case Study on Reddit Market Sentiment Analysis
- WWW 2023论文，介绍使用LLM标注的一种最简单方法：蒸馏法。

- 论文摘要：
  - 在金融市场情感分析任务上，经prompt调优后的PaLM-540B可远超Baseline的结果，在Reddit上ACC=72%（+22%）；
  - 论文实践出的`最佳prompt`组合技：`manual few-shot COT + self-consistency`；
  - 论文提出2种蒸馏方式：
    - 基于分类的蒸馏（CLS）、
    - 基于回归的蒸馏（REG），
    - 最后选择了P-R曲线更平滑的REG；
  - 蒸馏到task model（backbone：Charformer[3]的encoder部分）之后，在Reddit上ACC=69%（只降了3个点），并具备迁移相似任务的能力。

- 技术框架

  - 第一步：利用LLM来标注unlabel data，通过 `样本 -> LLM -> hard/soft label`，得到weakly labeled data；

  - 第二步：更小的任务模型（T5/BERT等）直接从weakly labeled data中进行监督学习。

![Screenshot 2023-11-13 at 22.20.24](/assets/img/Screenshot%202023-11-13%20at%2022.20.24.png)

- 用LLM标注的原因

  - 任务: 社交媒体中的金融市场情感分析。
  - 任务定义如下：
    - 给定一篇reddit帖子，模型判断这篇帖子表达出的、针对某公司的financial sentiment，
    - 具体是要进行3分类，候选标签集合为：positive、negative和neutral。

  - 用LLM来标注的理由有二：

    - 标注难。该任务需要同时具有金融 + 社交媒体的知识，对标注员的专业性要求高，在论文作者的实验中，即使是人类也只能达到70%的标注一致性，仅通过人类难以获得大量的高质量标注样本；

    - LLM标注效果好。试验使用LLM + In-Context Learning来标注，在进行了promp工程之后发现效果不错，在Reddit上ACC达到72%，考虑到任务的难点，LLM标注效果符合预期，于是采用LLM来标注。

- LLM的标注效果

  - 首先，论文作者把`情感分析任务`改成了`预测股票涨跌的任务`，positive 对应 看涨、negative 对应 看跌、neutral 对应 不确定。

  - 笔者认为，这个调整让任务更加具体，对LLM以及人类来说，判断股票涨跌 要比 判断抽象的金融情绪 更好理解。`“具体”本就是prompt的原则之一`。
  - 在`PaLM-540B COT` * 8（8代表self-consistency中sample的次数）的设置下，在各项数据集上可以取得远超Baseline的结果，在Reddit上Acc=72%，而`FinBERT-HKUST`仅为50%。

![Screenshot 2023-11-13 at 22.28.59](/assets/img/Screenshot%202023-11-13%20at%2022.28.59.png)


- 需要着重说明的是，作者进行了prompt enginerring之后，LLM才被逐步调优到最佳效果。

  - 这套组合技是：**manual few-shot COT + self-consistency**。

  - **manual few-shot COT**[^Chain-of-Thought_Prompting_Elicits_Reasoning_in_LLMs]
    - 在prompt中人工加入包含解题步骤的examples，是few-shot learning和COT的结合。
    - 论文中，使用了6个examples（每个类别随机挑2个），COT则是先总结对股票涨跌的opinion，然后再给出最终答案。
    - 生成opinion是为了让LLM给自己引入金融领域的知识，这种方法对特定domain的任务有启发性；

    - ![Screenshot 2023-11-13 at 22.33.11](/assets/img/Screenshot%202023-11-13%20at%2022.33.11.png)

  - **self-consistency**[^Self_Consistency_Improves_Chain_of_Thought_Reasoning_in_LLMs]
    - 多次sample LLM的结果（进行sampling，而非greedy decoding），再将最频繁出现的结果作为最终结果（即所谓`majority vote`）。
    - 论文中，temperature设置为0.5，最佳sample次数为8次。
    - ![Screenshot 2023-11-13 at 22.33.56](/assets/img/Screenshot%202023-11-13%20at%2022.33.56.png)

> [^Chain-of-Thought_Prompting_Elicits_Reasoning_in_LLMs]: Chain-of-Thought Prompting Elicits Reasoning in Large Language Models: https://proceedings.neurips.cc/paper_files/paper/2022/file/9d5609613524ecf4f15af0f7b31abca4-Paper-Conference.pdf

> [^Self_Consistency_Improves_Chain_of_Thought_Reasoning_in_LLMs]: Self-Consistency Improves Chain of Thought Reasoning in Language Models: https://openreview.net/pdf?id=1PL1NIMMrw


- 根据作者的消融实验，有以下发现：

  - COT、self-consistency的提升效果都很大，ACC从平平无奇的50%提升到了72%；
  - 对比PaLM-62B和540B，self-consistency对“小”模型也有帮助，但COT对“小”模型的帮助不大；
  - 随机打乱example的顺序，variance问题仍然比较明显。
  - 笔者认为，
    - `In-Context Learning`是LLM迅速adapt到下游任务的关键
    - 更大的LLM + 更好的prompt技巧（如few-shot、COT、self-consistency）又是提效果的关键。


两种蒸馏方法

- 首先，在实验中作者仅保留了用于self-consistency的8次sample中，一致次数>=5次的样本（即丢弃了置信度低的样本，这些样本通常准确率也较低）；

- 然后，采取以下2种方式来进行蒸馏：

  - CLS：每个样本得到hard label（即最频繁出现的label），直接通过正常的分类loss（cross entropy）来学习；

  - REG：每个样本得到soft label（把sample 8次结果的agreement ratio，转换为label分布），通过regression loss（MSE）来学习。

  - ![Screenshot 2023-11-13 at 22.37.19](/assets/img/Screenshot%202023-11-13%20at%2022.37.19.png)

- 根据实验结果，CLS和REG的最佳效果接近（80.5 + 68.0 vs 84.2 + 65.5），但两种方法有不同的特性：

  - CLS需要更准确的数据。随着agreement减小，尽管数据多了，但precision会下降，当agreement=8（即8次预测完全一样）时，效果最佳，但此时仅使用了31%的数据（用作蒸馏的数据共20000）；
  - REG的包容性更强。可以使用更多的、更难的（LLM预测更不一致）数据，在agreement=5时，效果最佳，可以使用85%的数据。
  - 最终作者选择了REG。一方面，REG用了更多的、更难的数据；另一方面，REG的P-R曲线更平滑一些（在部署时，需要根据预期presion来选择threshold，更平滑的话选点的效果更好）。
  - ![Screenshot 2023-11-13 at 22.38.24](/assets/img/Screenshot%202023-11-13%20at%2022.38.24.png)

  - 笔者认为，从知识蒸馏的研究[^Distilling_the_Knowledge_in_a_Neural_Network]来看，从soft label中学习的确是更好的方式，本论文的实验也证明要稍优一些；用self-consistency来产生soft label，进而蒸馏的思想，具有启发性

[^Distilling_the_Knowledge_in_a_Neural_Network]Distilling the Knowledge in a Neural Network, https://arxiv.org/pdf/1503.02531.pdf



- 蒸馏模型的效果


  - 在Baseline对比实验中，使用了3份测试数据集：

    - FiQA News，来自于FiQA benchmark，任务为 `新闻标题 -> 情感`二分类，训练/验证/测试比例 = 80/10/10；
    - FiQA Post，来自于FiQA benchmark，任务为 `推特和Stocktwits的博文 -> 情感`二分类，训练/验证/测试比例 = 80/10/10；
    - Reddit，任务为 `reddit帖子 -> 情感`三分类，人工标注100条用作测试，随机采样20000条用于任务模型的蒸馏。



  - Baseline如下：

    - 在FiQA上finetune后的Charformer-encoder（任务模型的backbone）。对应下图第一列模型；
    - 两个Pretrained model，FinBERT-ProsusAI、FinBERT-HKUST，未再做微调。对应下图第二列模型；
    - 用于标注的LLM（PaLM COT * 8）、蒸馏后的任务模型。对应下图第三列模型。

    - ![Screenshot 2023-11-13 at 22.40.29](/assets/img/Screenshot%202023-11-13%20at%2022.40.29.png)


- 根据对比实验，结论如下：

  - LLM效果最佳。在三个测试集上，LLM的效果都是显著最好的；
  - 蒸馏效果不错。在reddit上训练任务模型后，acc可以达到69%，只比LLM低3个点；
  - 任务模型泛化性不错。仅在reddit数据上蒸馏后，任务模型在FiQA News和FiQA Post上也具备迁移能力，说明LLM的标注的确让任务模型学到了判定“金融情绪”的较通用方式。


- 错误分析

  - 更进一步，论文作者对蒸馏后的任务模型进行了错误分析，绘制了混淆矩阵。
  - ![Screenshot 2023-11-13 at 22.43.58](/assets/img/Screenshot%202023-11-13%20at%2022.43.58.png)

  - 通过结果可以看出，任务模型主要是误判或漏判了Neural类别，作者观察数据后发现是因为模型对于包含了矛盾观点的帖子、包含了更高级的投资动作的帖子难以准确分类，因此有两个针对性的优化点：

    - 更好的处理包含矛盾观点的复杂帖子；
    - 考虑动态引入金融知识，以避免在COT过程中，LLM没有引入相关的金融知识。


- LLM与人类的协作标注
  - 人类标注者，能否不再是简单地标注数据，而是帮助设计一种domain-knowledge-injected prompt，帮助LLM来执行任务或者是与LLM进行协同？
  - 笔者认为，LLM作为一个工具，如何适当地使用它，让整个标注系统更加的低成本、高质量、高效率，才是人类设计者的最终目的。


---

### LLM data labeling benchmark

- LLMs can label data as well as humans, but 100x faster [^labeling_benchmark]
- a benchmark for evaluating performance of LLMs for labeling text datasets

[^labeling_benchmark]: labeling benchmark, https://www.refuel.ai/blog-posts/llm-labeling-technical-report

Key takeaways and learnings:
- State of the art LLMs can label text datasets at the same or better quality compared to skilled human annotators, but ~20x faster and ~7x cheaper.
- For achieving the `highest quality labels`, GPT-4 is the best choice among out of the box LLMs (88.4% agreement with ground truth, compared to 86% for skilled human annotators).
- For achieving the `best tradeoff between label quality and cost`, GPT-3.5-turbo, PaLM-2 and open source models like FLAN-T5-XXL are compelling.
- `Confidence based thresholding` can be a very effective way to mitigate impact of hallucinations and ensure high label quality.
‍

**LLMs for data labeling** [^LLMs_for_data_labeling]

[^LLMs_for_data_labeling]: Labeling with Confidence https://www.refuel.ai/blog-posts/labeling-with-confidence

- When leveraging LLMs for data labeling, it is important to be able to accurately estimate the `model’s level of confidence` in its own knowledge and reasoning.
- Doing so enables us to `automatically reject` low confidence labels and ensemble LLMs optimally.
- We examine different techniques for estimating confidence of LLM generated labels, in the context of data labeling for NLP tasks.
- **Key takeaways**:
  - `Token-level generation probabilities` (commonly referred to as `logprobs`) are by far the most accurate technique for estimating LLM confidence.
  - Explicitly prompting the LLM to output a confidence score, while popular, is `highly unreliable`.
    - This technique had the lowest accuracy and the highest standard deviation across datasets.
    - More often than not the LLM just hallucinates some number.

- used `Autolabel`, recently open-sourced library, to run all the experiments that are a part of this report.
  - Next generation data labeling tools like Autolabel leverage LLMs to create `large, diverse labeled datasets`.

- As a motivating example, consider the `Toxic comments classification dataset` from the labeling benchmark. If are able to estimate the LLM’s confidence level alongside the labels it generates, can
  - calibrate the model’s label quality (% agreement with ground truth labels) at any given confidence score.
  - then decide an `operating point` (**confidence threshold**) for the LLM, and reject all labels below this threshold.

![Screenshot 2023-12-04 at 05.02.35](/assets/img/Screenshot%202023-12-04%20at%2005.02.35.png)


---

#### Exam 1: Test Methodology

##### Setup

![Screenshot 2023-11-13 at 21.55.35](/assets/img/Screenshot%202023-11-13%20at%2021.55.35.png)

1. For an `input x`, generate a `label y` by prompting the Labeling LLM (GPT-4).
1. Next, estimate the confidence `c`, given x and y using a Verifier LLM (FLAN-T5-XXL or GPT-4). This score quantifies the Verifier LLM’s confidence in the given `label y` being correct. We describe four confidence calculation methods that benchmark in the section below.
1. Then, compare y with the ground truth label gt to decide whether the prediction is a “true positive” (y == gt) or “false positive” (y != gt).
1. Finally, compute AUROC (Area Under the Receiver Operating Characteristic) to understand how “good” a specific confidence estimation method is. For building the ROC curve, first compute true positive rate and false positive rates at various score thresholds. We use all distinct values of c as thresholds.

For generating the labels themselves, use GPT-4 as the labeling LLM in this evaluation.
- Ideally we’d like to use the same LLM for verification as well, but token-level generation probabilities (logprobs) are currently not available for OpenAI chat engine models (including GPT-4).
- Hence use FLAN-T5-XXL as the verifier LLM. For techniques that don’t rely on logprobs, also report the AUROC numbers with GPT-4 as the verifier LLM.

**Metric**

The **AUROC** is a scalar value that summarizes the overall performance of the classifier by measuring the area under the **ROC curve**.
- It provides a measure of the classifier's ability to distinguish between the positive and negative classes across all possible classification thresholds.
- The confidence score that are calculating can be thought of as a binary classifier where the labels are whether the model got a specific label `correct` (true positive) or `incorrect` (false positive).

![Screenshot 2023-12-04 at 04.43.33](/assets/img/Screenshot%202023-12-04%20at%2004.43.33.png) [^Metric]

[^Metric]: https://vitalflux.com/roc-curve-auc-python-false-positive-true-positive-rate/
‍
- Using the plot above as a reference, can see that a random classifier would have an AUROC of 0.5.

![Screenshot 2023-12-04 at 04.45.30](/assets/img/Screenshot%202023-12-04%20at%2004.45.30.png)


**Datasets**

We included the following datasets for this evaluation. These datasets are available for download [here](https://docs.refuel.ai/guide/resources/refuel_datasets/).

List of datasets used for labeling in this report

![Screenshot 2023-12-04 at 05.08.34](/assets/img/Screenshot%202023-12-04%20at%2005.08.34.png)


##### Results

![Screenshot 2023-12-04 at 04.45.30](/assets/img/Screenshot%202023-12-04%20at%2004.45.30.png)

> evaluated the performance of five different confidence estimation techniques across different NLP tasks and datasets.

- **Explicitly prompting the LLM to output a confidence score**
  - the least accurate.
  - This technique had the `lowest AUROC (0.58)` and the `highest standard deviation  (+/- 0.13)` across datasets.
  - More often than not the LLM just hallucinates some number.

- **Token probability**
  - still by far the most accurate and reliable technique for estimating LLM confidence.

  - Across all the 4 datasets, this technique achieves the `highest AUROC (0.832)`, and the `lowest standard deviation 偏差 (+/- 0.07)`.

  - Even with token probabilities, do see some variability in how well it can work for a given labeling task. From early exploration internally with public and proprietary datasets, have found that fine tuning the verifier LLM on the target labeling task improves the calibration significantly. We hope to share more details about this exploration in a future blog post.

Here’s a breakdown of the performance by dataset:

![Screenshot 2023-12-04 at 08.00.06](/assets/img/Screenshot%202023-12-04%20at%2008.00.06.png)



##### Qualitative Evaluation

In order to get a more intuitive understanding for which kinds of inputs LLMs are more confident, vs less confident about, here’s what did:

- Compute embeddings for all inputs in the dataset using [sentence transformers](https://github.com/UKPLab/sentence-transformers)
- Project all inputs into a 2D embedding map using [UMAP](https://umap-learn.readthedocs.io/en/latest/)
- Overlay this projection map with useful metadata such as confidence score and “whether the LLM label is correct i.e. matches ground truth label” for each point.


We illustrate this with the Banking complaints classification dataset below.
- clusters of low confidence inputs correspond q unite well to incorrect LLM labels
- Overlaying confidence score and whether `the LLM label (y) == Ground truth label (gt)` for the Banking Dataset

![Screenshot 2023-12-04 at 08.03.07](/assets/img/Screenshot%202023-12-04%20at%2008.03.07.png)



Further, examine a few rows with low confidence LLM labels (<= 0.2).
- Many of these inputs are ambiguous 模稜兩可, and it is hard even for a human annotator to tell which of the labels (the one provided as ground truth in the dataset, or the LLM generated one) is “correct”.

![Screenshot 2023-12-04 at 08.12.58](/assets/img/Screenshot%202023-12-04%20at%2008.12.58.png)


---

#### Exam 2: Language Models (Mostly) Know What They Know

We study whether language models can evaluate the validity of their own claims and predict which questions they will be able to answer correctly. [^Language_Models_Know_What_They_Know]

[^Language_Models_Know_What_They_Know]: Language Models (Mostly) Know What They Know, https://arxiv.org/pdf/2207.05221.pdf

- We first show that `larger models are well-calibrated on diverse multiple choice and true/false question` when they are provided in the right format.

- Thus we can approach **self-evaluation** on open-ended sampling tasks
  - asking models to first propose answers, and then to evaluate the `probability "P(True)"` that their answers are correct.
  - We find encouraging performance, calibration, and scaling for P(True) on a diverse array of tasks.
  - Performance at self-evaluation further improves when we allow models to consider many of their own samples before predicting the validity of one specific possibility.

- Next, we investigate whether models can be trained to predict `"P(IK)"`
  - `"P(IK)"`: the probability that "I know" the answer to a question, without reference to any particular proposed answer.
  - Models perform well at predicting P(IK) and partially generalize across tasks, though they struggle with calibration of P(IK) on new tasks.
  - The predicted P(IK) probabilities also increase appropriately in the presence of relevant source materials in the context, and in the presence of hints towards the solution of mathematical word problems.
- We hope these observations lay the groundwork for training more honest models, and for investigating how honesty generalizes to cases where models are trained on objectives other than the imitation of human writing.


---

#### Estimating Confidence with Autolabel

- Autolabel library relies on `token level generation probabilities` to estimate LLM label confidence.
- Generating confidence scores alongside labels is a simple config change - setting the key `compute_confidence = True` should initiate confidence score computation:
- Enabling confidence estimation in the library is a one line config change
![Screenshot 2023-12-04 at 08.15.00](/assets/img/Screenshot%202023-12-04%20at%2008.15.00.png)

However, very few LLM providers today support extraction of token level generation probabilities alongside the completion.

For all other models, Refuel provides access to a hosted Verifier LLM (currently a FLAN T5-XXL model) via an API to estimate `logprobs` as a post-processing step after the label is generated, regardless of the LLM that was originally used to generate the label.

---


#### Confidence estimation techniques

**Techniques**

![Screenshot 2023-12-04 at 05.09.23](/assets/img/Screenshot%202023-12-04%20at%2005.09.23.png)

>Is the first option still a good method to try?

We benchmark the following four methods for confidence estimation:


##### P(True)
1. The labeling LLM (GPT-4) generates a label (llm_label) given an input prompt.
2. Using this, prompt the verifier LLM to complete the following sentence.
3. The token generation probability of `“Yes”` is used as the confidence score

![Screenshot 2023-12-04 at 05.11.26](/assets/img/Screenshot%202023-12-04%20at%2005.11.26.png)


##### Prompting for Confidence Score

- The labeling LLM (GPT-4) generates a label (`llm_label`) given an input (`prompt`).
- Using this, prompt the verifier LLM to complete the following sentence.
- The value output by the verifier LLM is parsed as a float and used as the confidence score.
- If parsing is unsuccessful, give the sample a confidence score of 0.0 by default.

![Screenshot 2023-12-04 at 05.13.40](/assets/img/Screenshot%202023-12-04%20at%2005.13.40.png)

##### Token probabilities

- For classification-like and QA tasks, this is simply the probability of the first token in the generated label output produced.
  - For NER, probabilities are first generated for all tokens and then the probability of tokens for each entity is averaged to `compute confidence scores per entity`.

- use the Verifier LLM for estimating token probabilities of a prediction
  - first, generate the prediction logits for all tokens in the vocabulary, for the length of the output sequence.
  - Then, compute the softmax over the token probability distribution for each index and use the probabilities corresponding to respective tokens in the prediction as token probabilities.

##### Entropy 熵

- Exhaustive Entropy
  - This method is used to calculate entropy for classification-like tasks. First, calculate the probability of each of the possible labels and then calculate the 2-bit shannon entropy of the probability distribution over possible labels.

- ‍Semantic Entropy
  - This method is used to calculate entropy for generation-like tasks. We prompt the LLM to produce N predictions at a temperature of 0.5 and then group them using a pairwise ROGUE score. Then, will calculate the average probability of each group of predictions and calculate the entropy of that prediction distribution.
‍
![Screenshot 2023-12-04 at 07.53.10](/assets/img/Screenshot%202023-12-04%20at%2007.53.10.png)
