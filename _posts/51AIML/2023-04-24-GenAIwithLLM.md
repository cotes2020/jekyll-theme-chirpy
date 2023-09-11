---
title: AIML - Generative AI with Large Language Models
date: 2023-04-24 11:11:11 -0400
description:
categories: [51AIML]
# img: /assets/img/sample/rabbit.png
tags: [AIML]
---

# AIML - Generative AI with Large Language Models

- [AIML - Generative AI with Large Language Models](#aiml---generative-ai-with-large-language-models)
  - [Overall](#overall)
  - [Traditional AIML vs GenAI](#traditional-aiml-vs-genai)
  - [Attention is all you need](#attention-is-all-you-need)
  - [subject thoery](#subject-thoery)
  - [Hugging Face](#hugging-face)
    - [Generative AI Time Series Forecasting](#generative-ai-time-series-forecasting)
    - [Multivariate Time Series Forecasting](#multivariate-time-series-forecasting)
    - [Generative AI Transformers for Time Series Forecasting](#generative-ai-transformers-for-time-series-forecasting)
    - [Falcon-Chat demo](#falcon-chat-demo)
  - [Juptyper](#juptyper)

---

## Overall

**OpenSource**
- hugging face
- OpenAI
- Generative AI (answers for everything)

**Program**
- python
- panda

**AI modeling**
- pyTorch
- Tensor flow (Google)

**ML platforms**
- Jupyter Notebooks

**Time series**
- Forecasting and predictive Analytics

**Use case**
- Supply Chain Management with GenAI

OpenSource -> fine tuning -> custom result

---

## Traditional AIML vs GenAI

Traditional AIML
- good at **identify pattern**
- learning from the pattern
- limit success with close supervised learning of very large amount of data
- must have human involved


GenAI
- produces 'content' (text, image, music, art, forecasts, etc...)
- use 'transformers' (Encoders/Decoders) based on pre-trained data using small amount of fine tuning data
  - encode and decode at the same time
    - less data and faster
    - GenAI use small data and uses encoders and decoders and Transformers to take that smaller data and be able to use it for other types of models. (Pre training)
    - then add on top of it small amounts of fine tuning data
    - and then get a a training model.
  - As perceptions, not neurons.

---

## Attention is all you need

- high alignment

![Screenshot 2023-09-06 at 22.55.58](/assets/img/post/Screenshot%202023-09-06%20at%2022.55.58.png)

![Screenshot 2023-09-06 at 22.56.59](/assets/img/post/Screenshot%202023-09-06%20at%2022.56.59.png)

- Multi-Headed Attention

![Screenshot 2023-09-06 at 23.10.28](/assets/img/post/Screenshot%202023-09-06%20at%2023.10.28.png)

---

## subject thoery

GPU
- cloud class instance (from NVIDIA)
  - google colab
  - kaggle
  - amazon sagemaker
  - gradient
  - microsoft azure

> Tesla is graphics cards from NVIDIA for AI



pyTorch
- it does these heavy mathematical computation very easily with libraries
- it got a whole set of APIs and utilities that let you manipulate all of these different tensors



tensors
- a tensor is a computer data object (data structure) that represents numeric data,
- it could be floating point data values or data objects within data objects.
- 1d tensors (column)
- 2d tensors (xy)
- 3d tensors (xyz)
- 4d tensors (**cube**)
- 5d tensors
- 6d tensors


---


## Hugging Face

> like github repo

- search for an AI model


```bash
# +++++ Getting started with our git and git-lfs interface
# If you need to create a repo from the command line (skip if you created a repo from the website)
pip install huggingface_hub
# You already have it if you installed transformers or datasets
huggingface-cli login
# Log in using a token from huggingface.co/settings/tokens
# Create a model or dataset repo from the CLI if needed
huggingface-cli repo create repo_name --type {model, dataset, space}


# +++++ Clone your model or dataset locally
# Make sure you have git-lfs installed
# (https://git-lfs.github.com)
git lfs install
git clone https://huggingface.co/username/repo_name


# +++++ Then add, commit and push any file you want, including larges files
# save files via `.save_pretrained()` or move them here
git add .
git commit -m "commit from $USER"
git push


# +++++ In most cases, if you're using one of the compatible libraries, your repo will then be accessible from code, through its identifier: username/repo_name
# For example for a transformers model, anyone can load it with:
tokenizer = AutoTokenizer.from_pretrained("username/repo_name")
model = AutoModel.from_pretrained("username/repo_name")
```


### Generative AI Time Series Forecasting

> [Generative AI Time Series Forecasting](https://huggingface.co/blog/time-series-transformers)



### Multivariate Time Series Forecasting

> [Multivariate Time Series Forecasting](https://huggingface.co/blog/informer)



### Generative AI Transformers for Time Series Forecasting

> [Generative AI Transformers for Time Series Forecasting](https://huggingface.co/blog/autoformer)


---


### Falcon-Chat demo

> [Falcon-Chat demo](https://huggingface.co/spaces/HuggingFaceH4/falcon-chat)





---

## Juptyper

> To run a shell command from within a notebook cell, you must put a ! in front of the command:
> !pip install hyperopt

```py
!nvidia-smi --list-gpus


!pip install --upgrade pip

!pip uninstall -y git+https://github.com/openai/CLIP.git \
  urllib3==1.25.10 \
  sentence_transformers \
  torch torchvision pytorch-lightning lightning-bolts

# install supporting puthon packages for Data Frame processing
# and for Progress Bar
!pip install numpy pandas matplotlib tqdm scikit-learn

# install only the older version of Torch
!pip install --ignore-installed \
    urllib3==1.25.10 \
    torch torchvision pytorch-lightning lightning-bolts

# install latest (Upgrade) sentence transformers for fine-tuning
!pip install --ignore-installed \
  urllib3==1.25.10 \
  pyyaml \
  sentence_transformers

# Use CLIP model from OpenAI
!pip install git+https://github.com/openai/CLIP.git

# load the python package to run Pandas in parallel for better speed
!pip install pandarallel

!pip install torchaudio

!pip uninstall -y nvidia_cublas_cu11
```


.
