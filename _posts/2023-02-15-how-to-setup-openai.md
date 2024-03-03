---
title: How to set up OpenAI and use their APIs
author: yuanjian
date: 2024-02-15 20:25:00 -0500
categories: [Blogging, Tutorial]
tags: [OpenAI, GPT, DALLE-2, ChatGPT]
pin: true
---

We will start with OpenAI the company, then introduce the practical setup of OpenAI account, and the actual usage of their APIs.

OpenAI Brief History
======

OpenAI was founded in 2015 as a non-profit organization. Elon Musk and Sam Altman annouced the creation of OpenAI at the end of NIPS (Neural Information Processing Systems) conference in Montreal. It is said that they were worried about Google dominating the AI market after the acquisition of Deepmind in 2014. The company transitioned to a 'capped-for-profit' company in 2019 to attract an investment from Microsoft.

> Sam Altman: Dropped out of Stanford to create a location-based social network called Loopt in 2005. Loopt was acquired in 2012 for $43.4 million. In 2011, Altman started as a part-time partner at startup accelerator Y Combinator (YC). In 2014, he became the president of YC. It wasn't until 2019, Altman transitioned away from YC to focus on OpenAI. Gary Tan is now the president of YC.

> Elon Musk: In 1995, he started Zip2 with his brother Kimbal Musk and Greg Kouri. Zip2 was sold to Compaq in 1999 for (US dollar) 307 million (Elon's stake was worth about 7 million). In 1999, he starts an online financial services company X.com, which is merged with Confinity to form PayPal. In 2002, Paypal is acquired by eBay for 1.5 billion. Musk's stake was about 176 million. In 2004, Musk invested in Tesla, becoming its largest shareholder. In 2022, Musk acquires Twitter for $43 billion. Musk is also involved in Solar City, Boring Company, Neuralink, Starlink, and OpenAI.

Early major investors in OpenAI include Reid Hoffman, Jessica Livingston, Peter Thiel, Infosys, Khosla Ventures, YC Research.

In 2016, OpenAI released the Gym library, which allowed for an easy-to-use environment for reinforcement learning. In 2018, OpenAI announced the first version of GPT (Generative Pre-Trained Transformer). In 2019, OpenAI announced a new model called GPT-2. In 2020, GPT-3 was announced. In 2021, OpenAI announced the creation of DALL-E, a model capable of producing images from text. In 2022, **DALLE-2** and **ChatGPT** are announced. DALLE-2 creates much higher fidelity images from text prompts. ChatGPT is an optimized version of GPT for dialogue, trained on human feedback. At the start of 2023, OpenAI announced that Microsoft made a new $10 billion investment for OpenAI. Azure is now the exclusive provider of OpenAI model API calls.

GPT Brief Intro
======

### Words to Vectors

- We cannot calculate an error loss between words, but we can calcualte difference between two vectors.
- Text is converted to tokens, and tokens are encoded as vectors.
- An embedding neural network is used to convert the tokens into a vector, GPT-3 initially used a 12,288 dimension vector.



DALLE-2 Brief Intro
======
Here is a simplified high level illustration of how it looks inside DALLE. There is one important question, why not directly decode the text embedding but going through a Prior and a Image Embedding? The authors noted much better results whhen using the additional Prior model using CLIP.

![DALLE illustration](https://i.ibb.co/kMw0g80/diffusionmodel.jpg)

An infinite number of images could be consistent with a given caption, so the outputs of the two encoders will not perfectly coincide. Hence a separate prior model is needed to "translate" the text embedding into an image embedding that could plausibly match it.

Contrastive Language-Image Pre-training (CLIP) only incentivized to learn the features of an image that are sufficient to match it up with the correct caption (as opposed to any of the others in the list). This makes CLIP not ideal for learning about certain aspects of images, like relative positions of objects.

![Contrastive Language-Image Pre-training](https://i.ibb.co/47v8T8v/language-image-pretraining.png)

Prior Stage:
  - Generate the CLIP image embedding from the given caption.

Decoder Stage:
  - A diffusion model is trained to undo the steps of a fixed corruption process. Each step of the corruption process adds a small amount of gaussian noise to the image and will make the final image pure noise, as shown below.
  - unCLIP receives both a corrupted version of the image it is trained to reconstruct, as well as the CLIP image embedding of the clean image.

![Diffusion model](https://i.ibb.co/qYZH2kF/inside-dalle.png)

DALLE-2 was trained on 512x512 images, and thus any higher resolution output is actually upscaled from 512x512. One interesting fact is that DALLE-2 can run on *edge-devices* such as an iPhone.

The First OpenAI API Program
=======

We need to get a personal API key from [openai.com](https://openai.com). They use the API key to track our API usage and charge us. There are $18 free credits for the first three months. A common practice is to set the API key as an environment variable `OPENAI_API_KEY`, and then we can retrieve it in our python code. A very simple usage of their Completion API is shown below.

```python
import openai
import os
openai.api_key = os.getenv('OPENAI_API_KEY')
response = openai.Completion.create(model = 'text-davinci-003',
prompt='Give me two reasons to learn OpenAI API with Python', max_tokens=300)
```

We asked the API with a prompt "Give me two reasons to learn OpenAI API with Python", and specified the model to be `text-davinci-003`, and the response is a JSON object shown below.

```javascript
{
  "choices": [
    {
      "finish_reason": "stop",
      "index": 0,
      "logprobs": null,
      "text": "\n\n1. OpenAI API with Python offers many advantages such as access to high-performance, easy-to-use, and open source libraries. This makes it much easier to focus on developing solutions rather than worrying about coding up the solutions from scratch.\n\n2. Similarly, Python can be used to take advantage of the many APIs that OpenAI has to offer such as natural language processing, image recognition and image classification, generative models, and more. With OpenAI APIs, developers can quickly and easily develop, train, and deploy machine learning solutions to all types of problems."
    }
  ],
  "created": 1676499417,
  "id": "cmpl-6kKN7UFQWLzX3l05rLoetP4D12kMT",
  "model": "text-davinci-003",
  "object": "text_completion",
  "usage": {
    "completion_tokens": 119,
    "prompt_tokens": 11,
    "total_tokens": 130
  }
}
```

If we retrieve the text by `response['choices'][0]['text']`, we can get the actual result as below.

> 1. OpenAI API with Python offers many advantages such as access to high-performance, easy-to-use, and open source libraries. This makes it much easier to focus on developing solutions rather than worrying about coding up the solutions from scratch.
> 2. Similarly, Python can be used to take advantage of the many APIs that OpenAI has to offer such as natural language processing, image recognition and image classification, generative models, and more. With OpenAI APIs, developers can quickly and easily develop, train, and deploy machine learning solutions to all types of problems.

This completes the first basic usage of OpenAI API.