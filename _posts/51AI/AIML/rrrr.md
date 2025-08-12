
- [AI Canon 教规](#ai-canon-教规)
- [A gentle introduction](#a-gentle-introduction)
  - [Foundational learning: neural networks, backpropagation, and embeddings](#foundational-learning-neural-networks-backpropagation-and-embeddings)
    - [Explainers](#explainers)
    - [Courses](#courses)
  - [Tech deep dive: understanding transformers and large models](#tech-deep-dive-understanding-transformers-and-large-models)
    - [Explainers](#explainers-1)
    - [Courses](#courses-1)
    - [Reference and commentary](#reference-and-commentary)
  - [Practical guides to building with LLMs](#practical-guides-to-building-with-llms)
    - [Reference](#reference)
    - [Courses](#courses-2)
    - [LLM benchmarks](#llm-benchmarks)
  - [Market analysis](#market-analysis)
    - [a16z thinking](#a16z-thinking)
    - [Other perspectives](#other-perspectives)
- [Landmark research results](#landmark-research-results)
    - [**Large language models**](#large-language-models)
      - [New models](#new-models)
      - [Model improvements (e.g. fine-tuning, retrieval, attention)](#model-improvements-eg-fine-tuning-retrieval-attention)
    - [**Image generation models**](#image-generation-models)
    - [**Agents**](#agents)
    - [**Other data modalities**](#other-data-modalities)
      - [Code generation](#code-generation)
      - [Video generation](#video-generation)
      - [Human biology and medical data](#human-biology-and-medical-data)
      - [Audio generation](#audio-generation)
      - [Multi-dimensional image generation](#multi-dimensional-image-generation)



# AI Canon 教规

> Research in artificial intelligence is increasing at an exponential rate. It’s difficult for AI experts to keep up with everything new being published, and even harder for beginners to know where to start.

“AI Canon”
- a curated list of resources we’ve relied on to get smarter about modern AI
- because these papers, blog posts, courses, and guides have had an outsized impact on the field over the past several years.


---


# A gentle introduction

> get up to speed quickly on the most important parts of the modern AI wave.


* **[Software 2.0](https://karpathy.medium.com/software-2-0-a64152b37c35)**: Andrej Karpathy was one of the first to clearly explain (in 2017!) why the new AI wave really matters. His argument is that AI is a new and powerful way to program computers. As LLMs have improved rapidly, this thesis has proven prescient, and it gives a good mental model for how the AI market may progress.

* **[State of GPT](https://build.microsoft.com/en-US/sessions/db3f4859-cd30-4445-a0cd-553c3304f8e2)**: Also from Karpathy, this is a very approachable explanation of how ChatGPT / GPT models in general work, how to use them, and what directions R&D may take.
*   **[What is ChatGPT doing … and why does it work?](https://writings.stephenwolfram.com/2023/02/what-is-chatgpt-doing-and-why-does-it-work/)**: Computer scientist and entrepreneur Stephen Wolfram gives a long but highly readable explanation, from first principles, of how modern AI models work. He follows the timeline from early neural nets to today’s LLMs and ChatGPT.

* **[Transformers, explained](https://daleonai.com/transformers-explained)**: This post by Dale Markowitz is a shorter, more direct answer to the question “what is an LLM, and how does it work?” This is a great way to ease into the topic and develop intuition for the technology. It was written about GPT-3 but still applies to newer models.

* **[How Stable Diffusion works](https://mccormickml.com/2022/12/21/how-stable-diffusion-works/)**: This is the computer vision analogue to the last post. Chris McCormick gives a layperson’s explanation of how Stable Diffusion works and develops intuition around text-to-image models generally. For an even _gentler_ introduction, check out this [comic](https://www.reddit.com/r/StableDiffusion/comments/zs5dk5/i_made_an_infographic_to_explain_how_stable/) from r/StableDiffusion.


---


## Foundational learning: neural networks, backpropagation, and embeddings

> base understanding of fundamental ideas in machine learning and AI, from the basics of deep learning to university-level courses from AI experts.

### Explainers


* **[Deep learning in a nutshell: core concepts](https://developer.nvidia.com/blog/deep-learning-nutshell-core-concepts/)**: This four-part series from Nvidia walks through the basics of deep learning as practiced in 2015, and is a good resource for anyone just learning about AI.

* **[Practical deep learning for coders](https://course.fast.ai/)**: Comprehensive, free course on the fundamentals of AI, explained through practical examples and code.

* **[Word2vec explained](https://towardsdatascience.com/word2vec-explained-49c52b4ccb71)**: Easy introduction to embeddings and tokens, which are building blocks of LLMs (and all language models).

* **[Yes you should understand backprop](https://karpathy.medium.com/yes-you-should-understand-backprop-e2f06eab496b)**: More in-depth post on back-propagation if you want to understand the details. If you want even more, try the [Stanford CS231n lecture](https://www.youtube.com/watch?v=i94OvYb6noo) on Youtube.

### Courses


* **[Stanford CS229](https://www.youtube.com/playlist?list=PLoROMvodv4rMiGQp3WXShtMGgzqpfVfbU)**: Introduction to Machine Learning with Andrew Ng, covering the fundamentals of machine learning.

* **[Stanford CS224N](https://www.youtube.com/playlist?list=PLoROMvodv4rOSH4v6133s9LFPRHjEmbmJ)**: NLP with Deep Learning with Chris Manning, covering NLP basics through the first generation of LLMs.


---

## Tech deep dive: understanding transformers and large models

> countless resources—some better than others—attempting to explain how LLMs work.

### Explainers


* **[The illustrated transformer](https://jalammar.github.io/illustrated-transformer/)**: A more technical overview of the transformer architecture by Jay Alammar.

* **[The annotated transformer](https://nlp.seas.harvard.edu/annotated-transformer/)**: In-depth post if you want to understand transformers at a source code level. Requires some knowledge of PyTorch.

* **[Let’s build GPT: from scratch, in code, spelled out](https://www.youtube.com/watch?v=kCc8FmEb1nY)**: For the engineers out there, Karpathy does a video walkthrough of how to build a GPT model.

* **[The illustrated Stable Diffusion](https://jalammar.github.io/illustrated-stable-diffusion/)**: Introduction to latent diffusion models, the most common type of generative AI model for images.

* **[RLHF: Reinforcement Learning from Human Feedback](https://huyenchip.com/2023/05/02/rlhf.html)**: Chip Huyen explains RLHF, which can make LLMs behave in more predictable and human-friendly ways. This is one of the most important but least well-understood aspects of systems like ChatGPT.

* **[Reinforcement learning from human feedback](https://www.youtube.com/watch?v=hhiLw5Q_UFg)**: Computer scientist and OpenAI cofounder John Shulman goes deeper in this great talk on the current state, progress and limitations of LLMs with RLHF.

### Courses


* **[Stanford CS25](https://www.youtube.com/watch?v=P127jhj-8-Y)**: Transformers United, an online seminar on Transformers.

* **[Stanford CS324](https://stanford-cs324.github.io/winter2022/)**: Large Language Models with Percy Liang, Tatsu Hashimoto, and Chris Re, covering a wide range of technical and non-technical aspects of LLMs.

### Reference and commentary


* **[Predictive learning, NIPS 2016](https://www.youtube.com/watch?v=Ount2Y4qxQo&t=1072s)**: In this early talk, Yann LeCun makes a strong case for unsupervised learning as a critical element of AI model architectures at scale. Skip to [19:20](https://youtu.be/Ount2Y4qxQo?t=1160) for the famous cake analogy, which is still one of the best mental models for modern AI.

* **[AI for full-self driving at Tesla](https://www.youtube.com/watch?v=hx7BXih7zx8)**: Another classic Karpathy talk, this time covering the Tesla data collection engine. Starting at [8:35](https://youtu.be/hx7BXih7zx8?t=515) is one of the great all-time AI rants, explaining why long-tailed problems (in this case stop sign detection) are so hard.

* **[The scaling hypothesis](https://gwern.net/scaling-hypothesis)**: One of the most surprising aspects of LLMs is that scaling—adding more data and compute—just keeps increasing accuracy. GPT-3 was the first model to demonstrate this clearly, and Gwern’s post does a great job explaining the intuition behind it.

* **[Chinchilla’s wild implications](https://www.lesswrong.com/posts/6Fpvch8RR29qLEWNH/chinchilla-s-wild-implications)**: Nominally an explainer of the important Chinchilla paper (see below), this post gets to the heart of the big question in LLM scaling: are we running out of data? This builds on the post above and gives a refreshed view on scaling laws.

* **[A survey of large language models](https://arxiv.org/pdf/2303.18223v4.pdf)**: Comprehensive breakdown of current LLMs, including development timeline, size, training strategies, training data, hardware, and more.

* **[Sparks of artificial general intelligence: Early experiments with GPT-4](https://arxiv.org/abs/2303.12712)**: Early analysis from Microsoft Research on the capabilities of GPT-4, the current most advanced LLM, relative to human intelligence.

* **[The AI revolution: How Auto-GPT unleashes a new era of automation and creativity](https://pub.towardsai.net/the-ai-revolution-how-auto-gpt-unleashes-a-new-era-of-automation-and-creativity-2008aa2ca6ae)**: An introduction to Auto-GPT and AI agents in general. This technology is very early but important to understand—it uses internet access and self-generated sub-tasks in order to solve specific, complex problems or goals.

* **[The Waluigi Effect](https://www.lesswrong.com/posts/D7PumeYTDPfBTp3i7/the-waluigi-effect-mega-post)**: Nominally an explanation of the “Waluigi effect” (i.e., why “alter egos” emerge in LLM behavior), but interesting mostly for its deep dive on the theory of LLM prompting.


---

## Practical guides to building with LLMs


> A new application stack is emerging with LLMs at the core. While there isn’t a lot of formal education available on this topic yet, we pulled out some of the most useful resources we’ve found.

### Reference


* **[Build a GitHub support bot with GPT3, LangChain, and Python](https://dagster.io/blog/chatgpt-langchain)**: One of the earliest public explanations of the modern LLM app stack. Some of the advice in here is dated, but in many ways it kicked off widespread adoption and experimentation of new AI apps.

* **[Building LLM applications for production](https://huyenchip.com/2023/04/11/llm-engineering.html)**: Chip Huyen discusses many of the key challenges in building LLM apps, how to address them, and what types of use cases make the most sense.

* **[Prompt Engineering Guide](https://www.promptingguide.ai/)**: For anyone writing LLM prompts—including app devs—this is the most comprehensive guide, with specific examples for a handful of popular models. For a lighter, more conversational treatment, try [Brex’s prompt engineering guide](https://github.com/brexhq/prompt-engineering).

* **[Prompt injection: What’s the worst that can happen?](https://simonwillison.net/2023/Apr/14/worst-that-can-happen/)** Prompt injection is a potentially serious security vulnerability lurking for LLM apps, with no perfect solution yet. Simon Willison gives the definitive description of the problem in this post. Nearly everything Simon writes on AI is outstanding.

* **[OpenAI cookbook](https://github.com/openai/openai-cookbook/tree/main)**: For developers, this is the definitive collection of guides and code examples for working with the OpenAI API. It’s updated continually with new code examples.

* **[Pinecone learning center](https://www.pinecone.io/learn/)**: Many LLM apps are based around a vector search paradigm. Pinecone’s learning center—despite being branded vendor content—offers some of the most useful instruction on how to build in this pattern.

* **[LangChain docs](https://python.langchain.com/en/latest/index.html)**: As the default orchestration layer for LLM apps, LangChain connects to just about all other pieces of the stack. So their docs are a real reference for the full stack and how the pieces fit together.

### Courses


* **[LLM Bootcamp](https://fullstackdeeplearning.com/llm-bootcamp/)**: A practical course for building LLM-based applications with Charles Frye, Sergey Karayev, and Josh Tobin.

* **[Hugging Face Transformers](https://huggingface.co/learn/nlp-course/chapter1/1)**: Guide to using open-source LLMs in the Hugging Face transformers library.

### LLM benchmarks


* **[Chatbot Arena](https://lmsys.org/blog/2023-05-03-arena/)**: An Elo-style ranking system of popular LLMs, led by a team at UC Berkeley. Users can also participate by comparing models head to head.

* **[Open LLM Leaderboard](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard)**: A ranking by Hugging Face, comparing open source LLMs across a collection of standard benchmarks and tasks.


---

## Market analysis


We’ve all marveled at what generative AI can produce, but there are still a lot of questions about _what it all means_. Which products and companies will survive and thrive? What happens to artists? How should companies use it? How will it affect literally jobs and society at large? Here are some attempts at answering these questions.

### a16z thinking


* **[Who owns the generative AI platform?](https://a16z.com/who-owns-the-generative-ai-platform/)**: Our flagship assessment of where value is accruing, and might accrue, at the infrastructure, model, and application layers of generative AI.

* **[Navigating the high cost of AI compute](https://a16z.com/2023/04/27/navigating-the-high-cost-of-ai-compute/)**: A detailed breakdown of why generative AI models require so many computing resources, and how to think about acquiring those resources (i.e., the right GPUs in the right quantity, at the right cost) in a high-demand market.

* **[Art isn’t dead, it’s just machine-generated](https://a16z.com/art-isnt-dead-its-just-machine-generated/)**: A look at how AI models were able to reshape creative fields—often assumed to be the last holdout against automation—much faster than fields such as software development.

* **[The generative AI revolution in games](https://a16z.com/the-generative-ai-revolution-in-games/)**: An in-depth analysis from our Games team at how the ability to easily create highly detailed graphics will change how game designers, studios, and the entire market function. [This follow-up piece](https://a16z.com/the-generative-ai-revolution-will-enable-anyone-to-create-games/) from our Games team looks specifically at the advent of AI-generated content vis à vis user-generated content.

* **[For B2B generative AI apps, is less more?](https://a16z.com/for-b2b-generative-ai-apps-is-less-more/)**: A prediction for how LLMs will evolve in the world of B2B enterprise applications, centered around the idea that summarizing information will ultimately be more valuable than producing text.

* **[Financial services will embrace generative AI faster than you think](https://a16z.com/2023/04/19/financial-services-will-embrace-generative-ai-faster-than-you-think/)**: An argument that the financial services industry is poised to use generative AI for personalized consumer experiences, cost-efficient operations, better compliance, improved risk management, and dynamic forecasting and reporting. 

* **[Generative AI: The next consumer platform](https://a16z.com/generative-ai-the-next-consumer-platform/)**: A look at opportunities for generative AI to impact the consumer market across a range of sectors from therapy to ecommerce.

* **[To make a real difference in health care, AI will need to learn like we do](https://time.com/6274752/ai-health-care/)**: AI is poised to irrevocably change how we look to prevent and treat illness. However, to truly transform drug discovery to care delivery, we should invest in creating an ecosystem of “specialist” AIs—that learn like our best physicians and drug developers do today.

* **[The new industrial revolution: Bio x AI](https://a16z.com/2023/05/17/the-new-industrial-revolution-bio-x-ai/)**: The next industrial revolution in human history will be biology powered by artificial intelligence.

### Other perspectives


* **[On the opportunities and risks of foundation models](https://arxiv.org/abs/2108.07258)**: Stanford overview paper on Foundation Models. Long and opinionated, but this shaped the term.

* **[State of AI Report](https://www.stateof.ai/)**: An annual roundup of everything going on in AI, including technology breakthroughs, industry development, politics/regulation, economic implications, safety, and predictions for the future.

* **[GPTs are GPTs: An early look at the labor market impact potential of large language models](https://arxiv.org/abs/2303.10130)**: This paper from researchers at OpenAI, OpenResearch, and the University of of Pennsylvania predicts that “around 80% of the U.S. workforce could have at least 10% of their work tasks affected by the introduction of LLMs, while approximately 19% of workers may see at least 50% of their tasks impacted.”

* **[Deep medicine: How artificial intelligence can make healthcare human again](https://www.amazon.com/Deep-Medicine-Eric-Topol-audiobook/dp/B07PJ21V5N/ref=sr_1_1?hvadid=580688888836&hvdev=c&hvlocphy=9031955&hvnetw=g&hvqmt=e&hvrand=13698160037271563598&hvtargid=kwd-646099228782&hydadcr=15524_13517408&keywords=eric+topol+deep+medicine&qid=1684965845&sr=8-1)**: Dr. Eric Topol reveals how artificial intelligence has the potential to free physicians from the time-consuming tasks that interfere with human connection. The doctor-patient relationship is restored. ([a16z podcast](https://a16z.com/podcast/a16z-podcast-ai-and-your-doctor-today-and-tomorrow/))


---

# Landmark research results


Most of the amazing AI products we see today are the result of no-less-amazing research, carried out by experts inside large companies and leading universities. Lately, we’ve also seen impressive work from individuals and the open source community taking popular projects into new directions, for example by creating automated agents or porting models onto smaller hardware footprints. 

Here’s a collection of many of these papers and projects, for folks who really want to dive deep into generative AI. (For research papers and projects, we’ve also included links to the accompanying blog posts or websites, where available, which tend to explain things at a higher level. And we’ve included original publication years so you can track foundational research over time.)

### **Large language models**

#### New models


* **[Attention is all you need](https://arxiv.org/abs/1706.03762)** (2017): The original transformer work and research paper from Google Brain that started it all. ([blog post](https://ai.googleblog.com/2017/08/transformer-novel-neural-network.html))

* **[BERT: pre-training of deep bidirectional transformers for language understanding](https://arxiv.org/abs/1810.04805)** (2018): One of the first publicly available LLMs, with many variants still in use today. ([blog post](https://ai.googleblog.com/2018/11/open-sourcing-bert-state-of-art-pre.html))

* **[Improving language understanding by generative pre-training](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf)** (2018): The first paper from OpenAI covering the GPT architecture, which has become the dominant development path in LLMs. ([blog post](https://openai.com/research/language-unsupervised))

* **[Language models are few-shot learners](https://arxiv.org/abs/2005.14165)** (2020): The OpenAI paper that describes GPT-3 and the decoder-only architecture of modern LLMs.

* **[Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155)** (2022): OpenAI’s paper explaining InstructGPT, which utilizes humans in the loop to train models and, thus, better follow the instructions in prompts. This was one of the key unlocks that made LLMs accessible to consumers (e.g., via ChatGPT). ([blog post](https://openai.com/research/instruction-following))

* **[LaMDA: language models for dialog applications](https://arxiv.org/abs/2201.08239)** (2022): A model form Google specifically designed for free-flowing dialog between a human and chatbot across a wide variety of topics. ([blog post](https://blog.google/technology/ai/lamda/)) 

* **[PaLM: Scaling language modeling with pathways](https://arxiv.org/abs/2204.02311)** (2022): PaLM, from Google, utilized a new system for training LLMs across thousands of chips and demonstrated larger-than-expected improvements for certain tasks as model size scaled up. ([blog post](https://ai.googleblog.com/2022/04/pathways-language-model-palm-scaling-to.html)). See also the [PaLM-2 technical report](https://arxiv.org/abs/2305.10403).

* **[OPT: Open Pre-trained Transformer language models](https://arxiv.org/abs/2205.01068)** (2022): OPT is one of the top performing fully open source LLMs. The release for this 175-billion-parameter model comes with code and was trained on publicly available datasets. ([blog post](https://ai.facebook.com/blog/democratizing-access-to-large-scale-language-models-with-opt-175b/))

* **[Training compute-optimal large language models](https://arxiv.org/abs/2203.15556)** (2022): The Chinchilla paper. It makes the case that most models are data limited, not compute limited, and changed the consensus on LLM scaling. ([blog post](https://www.deepmind.com/blog/an-empirical-analysis-of-compute-optimal-large-language-model-training))

* **[GPT-4 technical report](https://arxiv.org/abs/2303.08774)** (2023): The latest and greatest paper from OpenAI, known mostly for how little it reveals! ([blog post](https://openai.com/research/gpt-4)). The [GPT-4 system card](https://cdn.openai.com/papers/gpt-4-system-card.pdf) sheds some light on how OpenAI treats hallucinations, privacy, security, and other issues.

* **[LLaMA: Open and efficient foundation language models](https://arxiv.org/abs/2302.13971)** (2023): The model from Meta that (almost) started an open-source LLM revolution. Competitive with many of the best closed-source models but only opened up to researchers on a restricted license. ([blog post](https://ai.facebook.com/blog/large-language-model-llama-meta-ai/))

* **[Alpaca: A strong, replicable instruction-following model](https://crfm.stanford.edu/2023/03/13/alpaca.html)** (2023): Out of Stanford, this model demonstrates the power of instruction tuning, especially in smaller open-source models, compared to pure scale.

#### Model improvements (e.g. fine-tuning, retrieval, attention)


* **[Deep reinforcement learning from human preferences](https://proceedings.neurips.cc/paper_files/paper/2017/file/d5e2c0adad503c91f91df240d0cd4e49-Paper.pdf)** (2017): Research on reinforcement learning in gaming and robotics contexts, that turned out to be a fantastic tool for LLMs.

* **[Retrieval-augmented generation for knowledge-intensive NLP tasks](https://arxiv.org/abs/2005.11401)** (2020): Developed by Facebook, RAG is one of the two main research paths for improving LLM accuracy via information retrieval. ([blog post](https://ai.facebook.com/blog/retrieval-augmented-generation-streamlining-the-creation-of-intelligent-natural-language-processing-models/))

* **[Improving language models by retrieving from trillions of tokens](https://arxiv.org/abs/2112.04426)** (2021): RETRO, for “Retrieval Enhanced TRansfOrmers,” is another approach—this one by DeepMind—to improve LLM accuracy by accessing information not included in their training data. ([blog post](https://www.deepmind.com/blog/improving-language-models-by-retrieving-from-trillions-of-tokens))

* **[LoRA: Low-rank adaptation of large language models](https://arxiv.org/abs/2106.09685) (2021): This research out of Microsoft introduced a more efficient alternative to fine-tuning for training LLMs on new data. It’s now become a standard for community fine-tuning, especially for image models.

* **[Constitutional AI (2022)](https://arxiv.org/abs/2212.08073)**: The Anthropic team introduces the concept of reinforcement learning from AI Feedback (RLAIF). The main idea is that we can develop a harmless AI assistant with the supervision of other AIs.

* **[FlashAttention: Fast and memory-efficient exact attention with IO-awareness](https://arxiv.org/abs/2205.14135)** (2022): This research out of Stanford opened the door for state-of-the-art models to understand longer sequences of text (and higher-resolution images) without exorbitant training times and costs. ([blog post](https://ai.stanford.edu/blog/longer-sequences-next-leap-ai/))

* **[Hungry hungry hippos: Towards language modeling with state space models](https://arxiv.org/abs/2212.14052)** (2022): Again from Stanford, this paper describes one of the leading alternatives to attention in language modeling. This is a promising path to better scaling and training efficiency. ([blog post](https://hazyresearch.stanford.edu/blog/2023-01-20-h3))

### **Image generation models**


* **[Learning transferable visual models from natural language supervision](https://arxiv.org/abs/2103.00020)** (2021): Paper that introduces a base model—CLIP—that links textual descriptions to images. One of the first effective, large-scale uses of foundation models in computer vision. ([blog post](https://openai.com/research/clip))

* **[Zero-shot text-to-image generation](https://arxiv.org/abs/2102.12092)** (2021): This is the paper that introduced DALL-E, a model that combines the aforementioned CLIP and GPT-3 to automatically generate images based on text prompts. Its successor, DALL-E 2, would kick off the image-based generative AI boom in 2022. ([blog post](https://openai.com/research/dall-e))

* **[High-resolution image synthesis with latent diffusion models](https://arxiv.org/abs/2112.10752)** (2021): The paper that described Stable Diffusion (after the launch and explosive open source growth).

* **[Photorealistic text-to-image diffusion models with deep language understanding](https://arxiv.org/abs/2205.11487)** (2022): Imagen was Google’s foray into AI image generation. More than a year after its announcement, the model has yet to be released publicly as of the publish date of this piece. ([website](https://imagen.research.google/))

* **[DreamBooth: Fine tuning text-to-image diffusion models for subject-driven generation](https://arxiv.org/abs/2208.12242)** (2022): DreamBooth is a system, developed at Google, for training models to recognize user-submitted subjects and apply them to the context of a prompt (e.g. \[USER\] smiling at the Eiffel Tower). ([website](https://dreambooth.github.io/))

* **[Adding conditional control to text-to-image diffusion models](https://arxiv.org/abs/2302.05543)** (2023): This paper from Stanford introduces ControlNet, a now very popular tool for exercising fine-grained control over image generation with latent diffusion models.

### **Agents**


* **[A path towards autonomous machine intelligence](https://openreview.net/pdf?id=BZ5a1r-kVsf) (2022): A proposal from Meta AI lead and NYU professor Yann LeCun on how to build autonomous and intelligent agents that truly understand the world around them.

* **[ReAct: Synergizing reasoning and acting in language models](https://arxiv.org/abs/2210.03629)** (2022): A project out of Princeton and Google to test and improve the reasoning and planning abilities of LLMs. ([blog post](https://ai.googleblog.com/2022/11/react-synergizing-reasoning-and-acting.html))

* **[Generative agents: Interactive simulacra of human behavior](https://arxiv.org/abs/2304.03442)** (2023): Researchers at Stanford and Google used LLMs to power agents, in a setting akin to “The Sims,” whose interactions are emergent rather than programmed.

* **[Reflexion: an autonomous agent with dynamic memory and self-reflection](https://arxiv.org/abs/2303.11366)** (2023): Work from researchers at Northeastern University and MIT on teaching LLMs to solve problems more reliably by learning from their mistakes and past experiences.

* **[Toolformer: Language models can teach themselves to use tools](https://arxiv.org/abs/2302.04761)** (2023): This project from Meta trained LLMs to use external tools (APIs, in this case, pointing to things like search engines and calculators) in order to improve accuracy without increasing model size. 

* **[Auto-GPT: An autonomous GPT-4 experiment](https://github.com/Significant-Gravitas/Auto-GPT)**: An open source experiment to expand on the capabilities of GPT-4 by giving it a collection of tools (internet access, file storage, etc.) and choosing which ones to use in order to solve a specific task.

* **[BabyAGI](https://github.com/yoheinakajima/babyagi)**: This Python script utilizes GPT-4 and vector databases (to store context) in order to plan and executes a series of tasks that solve a broader objective.

### **Other data modalities**

#### Code generation


* **[Evaluating large language models trained on code](https://arxiv.org/abs/2107.03374)** (2021): This is OpenAI’s research paper for Codex, the code-generation model behind the GitHub Copilot product. ([blog post](https://openai.com/blog/openai-codex))

* **[Competition-level code generation with AlphaCode](https://www.science.org/stoken/author-tokens/ST-905/full)** (2021): This research from DeepMind demonstrates a model capable of writing better code than human programmers. ([blog post](https://www.deepmind.com/blog/competitive-programming-with-alphacode))

* **[CodeGen: An open large language model for code with multi-turn program synthesis](https://arxiv.org/abs/2203.13474)** (2022): CodeGen comes out of the AI research arm at Salesforce, and currently underpins the Replit Ghostwriter product for code generation. ([blog post](https://blog.salesforceairesearch.com/codegen/))

#### Video generation


* **[Make-A-Video: Text-to-video generation without text-video data](https://arxiv.org/abs/2209.14792)** (2022): A model from Meta that creates short videos from text prompts, but also adds motion to static photo inputs or creates variations of existing videos. ([blog post](https://makeavideo.studio/))

* **[Imagen Video: High definition video generation with diffusion models](https://arxiv.org/abs/2210.02303)** (2022): Just what it sounds like: a version of Google’s image-based Imagen model optimized for producing short videos from text prompts. ([website](https://imagen.research.google/video/))

#### Human biology and medical data


* **[Strategies for pre-training graph neural networks](https://arxiv.org/pdf/1905.12265.pdf)** (2020): This publication laid the groundwork for effective pre-training methods useful for applications across drug discovery, such as molecular property prediction and protein function prediction. ([blog post](https://snap.stanford.edu/gnn-pretrain/))

* **[Improved protein structure prediction using potentials from deep learning](https://www.nature.com/articles/s41586-019-1923-7)** (2020): DeepMind’s protein-centric transformer model, AlphaFold, made it possible to predict protein structure from sequence—a true breakthrough which has already had far-reaching implications for understanding biological processes and developing new treatments for diseases. ([blog post](https://www.deepmind.com/blog/alphafold-a-solution-to-a-50-year-old-grand-challenge-in-biology)) ([explainer](https://www.blopig.com/blog/2021/07/alphafold-2-is-here-whats-behind-the-structure-prediction-miracle/))

* **[Large language models encode clinical knowledge](https://arxiv.org/abs/2212.13138)** (2022): Med-PaLM is a LLM capable of correctly answering US Medical License Exam style questions. The team has since published results on the performance of Med-PaLM2, which achieved a score on par with “expert” test takers. Other teams have performed similar experiments with [ChatGPT](https://www.medrxiv.org/content/10.1101/2022.12.19.22283643v2) and [GPT-4](https://arxiv.org/abs/2303.13375). ([video](https://www.youtube.com/watch?v=saWEFDRuNJc))

#### Audio generation


* **[Jukebox: A generative model for music](https://arxiv.org/abs/2005.00341)** (2020): OpenAI’s foray into music generation using transformers, capable of producing music, vocals, and lyrics with minimal training. ([blog post](https://openai.com/research/jukebox))

* **[AudioLM: a language modeling approach to audio generation](https://arxiv.org/pdf/2209.03143.pdf)** (2022): AudioLM is a Google project for generating multiple types of audio, including speech and instrumentation. ([blog post](https://ai.googleblog.com/2022/10/audiolm-language-modeling-approach-to.html))

* **[MusicLM: Generating nusic from text](https://arxiv.org/abs/2301.11325)** (2023): Current state of the art in AI-based music generation, showing higher quality and coherence than previous attempts. ([blog post](https://google-research.github.io/seanet/musiclm/examples/))

#### Multi-dimensional image generation


* **[NeRF: Representing scenes as neural radiance fields for view synthesis](https://arxiv.org/abs/2003.08934) (2020): Research from a UC-Berkeley-led team on “synthesizing novel views of complex scenes” using 5D coordinates. ([website](https://www.matthewtancik.com/nerf))

* **[DreamFusion: Text-to-3D using 2D diffusion](https://arxiv.org/pdf/2209.14988.pdf)** (2022): Work from researchers at Google and UC-Berkeley that builds on NeRF to generate 3D images from 2D inputs. ([website](https://dreamfusion3d.github.io/))


.
