---
title: AIML - AI Prompt engineering
date: 2023-04-24 11:11:11 -0400
description:
categories: [51AIML]
# img: /assets/img/sample/rabbit.png
tags: [AIML]
---

# AIML - Prompt engineering

- [AIML - Prompt engineering](#aiml---prompt-engineering)
  - [prompt engineering](#prompt-engineering)
    - [Prompt](#prompt)
    - [History of prompt engineering](#history-of-prompt-engineering)
    - [why is prompt engineering important](#why-is-prompt-engineering-important)
  - [benefits of prompt engineering](#benefits-of-prompt-engineering)
  - [prompt engineering use cases](#prompt-engineering-use-cases)
  - [Standard Definitions](#standard-definitions)
  - [Prompt elements](#prompt-elements)
  - [Prompt Engineering Use Cases](#prompt-engineering-use-cases-1)
  - [Prompt Engineering techniques](#prompt-engineering-techniques)
    - [Zero-shot Prompting:](#zero-shot-prompting)
    - [N-shot Prompting:](#n-shot-prompting)
    - [Chain-of-thought CoT prompting](#chain-of-thought-cot-prompting)
    - [Tree-of-thought prompting](#tree-of-thought-prompting)
    - [Maieutic 启发性的  prompting](#maieutic-启发性的--prompting)
    - [Complexity-based prompting](#complexity-based-prompting)
    - [Generated knowledge prompting](#generated-knowledge-prompting)
    - [Least-to-most prompting](#least-to-most-prompting)
    - [Self-refine prompting](#self-refine-prompting)
    - [Directional-stimulus prompting](#directional-stimulus-prompting)
  - [prompt design Best practices](#prompt-design-best-practices)
    - [Begin with the basics](#begin-with-the-basics)
    - [Crafting effective prompts (instructions)](#crafting-effective-prompts-instructions)
    - [Unambiguous prompts](#unambiguous-prompts)
    - [Adequate context within the prompt](#adequate-context-within-the-prompt)
    - [Balance between targeted information and desired output](#balance-between-targeted-information-and-desired-output)
    - [Experiment and refine the prompt](#experiment-and-refine-the-prompt)


ref:
- https://aws.amazon.com/what-is/prompt-engineering/#:~:text=Prompt%20engineering%20is%20the%20process,high%2Dquality%20and%20relevant%20output.
- https://www.leewayhertz.com/prompt-engineering/
-

---

## prompt engineering

Prompt engineering
- the process to guide generative artificial intelligence (generative AI) solutions to generate desired outputs.
- Even though generative AI attempts to mimic humans, it requires detailed instructions to create high-quality and relevant output.
- In prompt engineering, you choose the most appropriate formats, phrases, words, and symbols that guide the AI to interact with the users more meaningfully.
- Prompt engineers use creativity plus trial and error to create a collection of input texts, so an application's generative AI works as expected.

---

### Prompt

prompt
- `prompt`: a natural language text that requests the generative AI to perform a specific task.

- `Generative AI`: an artificial intelligence solution that creates new content like stories, conversations, videos, images, and music. It's powered by very large machine learning (ML) models that use deep neural networks that have been pretrained on vast amounts of data.

- `large language models (LLMs)`: very flexible and can perform various tasks.
  - However, because they're so open-ended, the users can interact with generative AI solutions through countless input data combinations.
  - not every type of input generates helpful output.
  - Generative AI systems `require context and detailed information to produce accurate and relevant responses`.

---


### History of prompt engineering

- Pre-transformer era (Before 2017)

  - Prompt engineering was less common before the `development of transformer-based models like OpenAI’s generative pre-trained transformer (GPT)`.
  - **Contextual knowledge and adaptability** are lacking in earlier language models like `recurrent neural networks (RNNs)` and `convolutional neural networks (CNNs)`, which restricts the potential for prompt engineering.

- Pre-training and the emergence of transformers (2017)

  - The introduction of transformers, specifically with the “Attention Is All You Need” paper by Vaswani in 2017, revolutionized the field of NLP.
  - Transformers made it possible to `pre-train language models on a broad scale` and `teach them how to represent words and sentences in context`.
  - However, throughout this time, prompt engineering was still a relatively unexplored technique.

- Fine-tuning and the rise of GPT (2018)

  - A major turning point for rapid engineering occurred with the introduction of OpenAI’s GPT models. GPT models demonstrated the effectiveness of pre-training and fine-tuning on particular downstream tasks. For a variety of purposes, researchers and practitioners have started using quick engineering techniques to direct the behavior and output of GPT models.

- Advancements in prompt engineering techniques (2018–present)

  - As the understanding of prompt engineering grew, researchers began experimenting with different approaches and strategies. This included designing context-rich prompts, using rule-based templates, incorporating system or user instructions, and exploring techniques like prefix tuning. The goal was to enhance control, mitigate biases and improve the overall performance of language models.





---

### why is prompt engineering important

Prompt engineering plays a crucial role in fine-tuning language models for specific applications, improving their accuracy, and ensuring more reliable results.
- Language models, such as GPT-3, have shown impressive capabilities in generating human-like text.
- However, without proper guidance, these models may produce responses that are either irrelevant, biased, or lack coherence.
- Prompt engineering allows us to steer these models towards desired behaviors and produce outputs that align with our intentions.

- In prompt engineering, you continuously refine prompts until get the desired outcomes from the AI system.





Prompt engineering jobs have increased significantly since the launch of generative AI.
- `bridge the gap` between the end users and the large language model.
- `identify scripts and templates` that the users can customize and complete to get the best result from the language models.
- experiment with different types of inputs to `build a prompt library` that application developers can reuse in different scenarios.

- makes AI applications more efficient and effective. Application developers typically encapsulate open-ended user input inside a prompt before passing it to the AI model.

For example
- AI chatbots.
- A user may enter an incomplete problem statement like, "Where to purchase a shirt."
- Internally, the application's code uses an engineered prompt that says, "You are a sales assistant for a clothing company. A user, based in Alabama, United States, is asking you where to purchase a shirt.
- Respond with the three nearest store locations that currently stock a shirt."
- The chatbot then generates more relevant and accurate information.


## benefits of prompt engineering

- **Greater developer control**

  - Prompt engineering gives developers more control over users' interactions with the AI. Effective prompts provide intent and establish context to the large language models. They help the AI refine the output and present it concisely in the required format.

  - They also prevent the users from misusing the AI or requesting something the AI does not know or cannot handle accurately. For instance, you may want to limit the users from generating inappropriate content in a business AI application.

- **Improved user experience**

  - Users avoid trial and error and still receive coherent, accurate, and relevant responses from AI tools. Prompt engineering makes it easy for users to obtain relevant results in the first prompt. It helps mitigate bias that may be present from existing human bias in the large language models’ training data.

  - enhances the user-AI interaction so the AI understands the user's intention even with minimal input. For example, requests to summarize a legal document and a news article get different results adjusted for style and tone. This is true even if both users just tell the application, "Summarize this document."

- **Increased flexibility**

  - Higher levels of abstraction improve AI models and allow organizations to create more flexible tools at scale. A prompt engineer can create prompts with domain-neutral instructions highlighting logical links and broad patterns. Organizations can rapidly reuse the prompts across the enterprise to expand their AI investments.

  - For example, to find opportunities for process optimization, the prompt engineer can create different prompts that train the AI model to find inefficiencies using broad signals rather than context-specific data. The prompts can then be used for diverse processes and business units.



---


## prompt engineering use cases

used in sophisticated AI systems to improve user experience with the learning language model.

- Subject matter expertise

  - in applications that `require the AI to respond with subject matter expertise`.
  - A prompt engineer with experience in the field can guide the AI to reference the correct sources and frame the answer appropriately based on the question asked.

  - For example
    - medical field, a physician could use a prompt-engineered language model to generate differential diagnoses for a complex case. The medical professional only needs to enter the symptoms and patient details. The application uses engineered prompts to guide the AI first to list possible diseases associated with the entered symptoms. Then it narrows down the list based on additional patient information.

- Critical thinking

  - Critical thinking applications `require the language model to solve complex problems`. To do so, the model analyzes information from different angles, evaluates its credibility, and makes reasoned decisions. Prompt engineering enhances a model's data analysis capabilities.

  - For instance
    - decision-making scenarios, prompt a model to list all possible options, evaluate each option, and recommend the best solution.

- Creativity

  - Creativity involves `generating new ideas, concepts, or solutions`. Prompt engineering can be used to enhance a model's creative abilities in various scenarios.

  - For instance,
    - writing scenarios, a writer could use a prompt-engineered model to help generate ideas for a story. The writer may prompt the model to list possible characters, settings, and plot points then develop a story with those elements. Or a graphic designer could prompt the model to generate a list of color palettes that evoke a certain emotion then create a design using that palette.


---



## Standard Definitions


`Label`:
- The specific category or task we want the language model to focus on, such as sentiment analysis, summarization, or question-answering.

`Logic`:
- The underlying rules, constraints, or instructions that guide the language model’s behavior within the given prompt.

`Model Parameters (LLM Parameters)`:
- Refers to the specific settings or configurations of the language model, including `temperature, top-k, and top-p sampling`, that influence the generation process.


**Temperature**
- controls the randomness of the model’s output.
- This is useful for tasks requiring precise and factual answers, like a fact-based question-answer system.
  - Lower values make the model’s output more deterministic, favoring the most probable next token.
  - high value induces more randomness in the model’s responses, allowing for more creative and diverse results.
- This is beneficial for creative tasks like poem generation.

**Top_p**
- used in a sampling technique known as `nucleus sampling`
- influences the determinism of the model’s response.
  - A lower value results in more exact and factual answers
  - a higher value increases the diversity of the responses.

One key recommendation is to adjust either ‘Temperature’ or ‘Top_p,’ but not both simultaneously, to prevent overcomplicating the system and to better control the effect of these settings.


---

## Prompt elements

When designing prompts, it’s essential to understand the basic structures and formatting techniques.

- Prompts often consist of instructions and placeholders that guide the model’s response.

- By providing clear and specific instructions, we can guide the model’s focus and produce more accurate results

- For example
  - in sentiment analysis, a prompt might include a placeholder for the text to be analyzed along with instructions such as “Analyze the sentiment of the following text: .”

- `Instruction`:
  - This is the directive given to the model that details what is expected in terms of the task to be performed.
  - This could range from “translate the following text into French” to “generate a list of ideas for a science fiction story”.
  - The instruction is usually the first part of the prompt and sets the overall task for the model.

- `Context`:
  - This element provides additional information that can guide the model’s response.
  - The context can help the model understand the style, tone, and specifics of the information needed.
  - Providing relevant background or context to ensure the model understands the task or query.
  - For instance,
    - in a translation task, you might provide some background on the text to be translated (like it’s a dialogue from a film or a passage from a scientific paper).

- `Task Specification`:
  - Clearly defining the task or objective the model should focus on, such as generating a summary or answering a specific question.

- `Constraints`:
  - Including any limitations or constraints to guide the model’s behavior, such as word count restrictions or specific content requirements.

- `Input data`:
  - This refers to the actual data that the model will be working with.
  - For instance,
    - In a translation task, this would be the text to be translated.
    - In a question-answering task, this would be the question being asked.

- `Output indicator`:
  - This part of the prompt signals to the model the format in which the output should be generated.
  - This can help narrow down the model’s output and guide it towards more useful responses.
  - For instance
    - specify that you want the model’s response in the form of a list, a paragraph, a single sentence, or any other specific structure.


---

## Prompt Engineering Use Cases

Prompt engineering can be applied to various NLP tasks. Let’s explore some common use cases:

- Information Extraction
- Text Summarization
- Question Answering
- Code Generation
- Text Classification






---

## Prompt Engineering techniques

> Prompt engineering is a dynamic and evolving field. It requires both linguistic skills and creative expression to fine-tune prompts and obtain the desired response from the generative AI tools.

Here are some examples of techniques that prompt engineers use to improve their AI models' `natural language processing (NLP)` tasks.

---

### Zero-shot Prompting:

- In zero-shot prompting, models are trained to perform tasks they haven’t been explicitly trained on. Instead, the prompt provides a clear task specification without any labeled examples.

---

### N-shot Prompting:

- N-shot prompting involves fine-tuning models with limited or no labeled data for a specific task.

- By providing a small number of labeled examples, language models can learn to generalize and perform the task accurately.

- N-shot prompting encompasses zero-shot and few-shot prompting approaches.

---


### Chain-of-thought CoT prompting

- a technique that breaks down a complex question into smaller, logical parts that mimic a train of thought.

- involves breaking down complex tasks into a sequence of simpler questions or steps.

- it guide the model through a coherent chain of prompts, we can ensure context-aware responses and improve the overall quality of the generated text.

- it helps the model solve problems in a series of intermediate steps rather than directly answering the question. This enhances its reasoning ability.

- perform several chain-of-though rollouts for complex tasks and choose the most commonly reached conclusion. If the rollouts disagree significantly, a person can be consulted to correct the chain of thought.

For example
- question: "What is the capital of France?"
  - the model might perform several rollouts leading to answers like
    - "Paris,"
    - "The capital of France is Paris,"
    - and "Paris is the capital of France."
  - Since all rollouts lead to the same conclusion, "Paris" would be selected as the final answer.

```bash
# Prompt:
"Identify the main theme of the given text."
"Provide three supporting arguments that highlight this theme."
"Summarize the text in a single sentence."

# Example Text:
"The advancement of technology has revolutionized various industries, leading to increased efficiency and productivity. It has transformed the way we communicate, works, and access information."

# Output:
Main Theme: "The advancement of technology and its impact on industries."
Supporting Arguments:
Increased efficiency and productivity
Transformation of communication, work, and information access
Revolutionizing various industries
Summary: "Technology's advancements have revolutionized industries, enhancing efficiency and transforming communication, work, and information access."
```


---

### Tree-of-thought prompting

The tree-of-thought technique
- generalizes chain-of-thought prompting.
- It prompts the model to generate one or more possible next steps. Then it runs the model on each possible next step using a tree search method.

For example
- question: "What are the effects of climate change?"
  - the model might first generate possible next steps like
    - "List the environmental effects"
    - and "List the social effects."
    - It would then elaborate on each of these in subsequent steps.


---


### Maieutic 启发性的  prompting

Maieutic prompting
- similar to tree-of-thought prompting.
- The model is prompted to answer a question with an explanation.
- The model is then prompted to explain parts of the explanation,.
- Inconsistent explanation trees are pruned or discarded.
- This improves performance on complex commonsense reasoning.

For example
- question: "Why is the sky blue?"
- the model might first answer,
  - "The sky appears blue to the human eye because the short waves of blue light are scattered in all directions by the gases and particles in the Earth's atmosphere."
  - It might then expand on parts of this explanation, such as why blue light is scattered more than other colors and what the Earth's atmosphere is composed of.

---

### Complexity-based prompting

- involves performing several chain-of-thought rollouts.
- It chooses the rollouts with the longest chains of thought then chooses the most commonly reached conclusion.

For example
- question: a complex math problem,
- the model might perform several rollouts, each involving multiple steps of calculations. It would consider the rollouts with the longest chain of thought, which for this example would be the most steps of calculations. The rollouts that reach a common conclusion with other rollouts would be selected as the final answer.

---

### Generated knowledge prompting

- involves leveraging external knowledge bases or generated content to enhance the model’s responses.

- By incorporating relevant information into prompts, models can provide detailed and accurate answers or generate content based on acquired knowledge.

- prompting the model to first generate relevant facts needed to complete the prompt. Then it proceeds to complete the prompt.

- This often results in higher completion quality as the model is conditioned on relevant facts.

For example
- question: prompts the model to write an essay on the effects of deforestation.
  - The model might first generate facts like
    - "deforestation contributes to climate change"
    - and "deforestation leads to loss of biodiversity."
  - Then it would elaborate on the points in the essay.


```bash
# Prompt:
"Based on the understanding of historical events, provide a brief explanation of the causes of World War II."

# Generated Knowledge:
"The main causes of World War II include territorial disputes, economic instability, the rise of totalitarian regimes, and the failure of international diplomacy."

# Output:
"The causes of World War II were influenced by territorial disputes, economic instability, the rise of totalitarian regimes, and the failure of international diplomacy."
```


---

### Least-to-most prompting

- the model is prompted first to list the subproblems of a problem, and then solve them in sequence. This approach ensures that later subproblems can be solved with the help of answers to previous subproblems.

For example
- question: prompts the model with a math problem like "Solve for x in equation 2x + 3 = 11." The model might first list the subproblems as "Subtract 3 from both sides" and "Divide by 2". It would then solve them in sequence to get the final answer.

--


### Self-refine prompting

- the model is prompted to solve the problem, critique its solution, and then resolve the problem considering the problem, solution, and critique.

-  focus on maintaining consistency and coherence in language model responses.

-  By comparing generated outputs and ensuring they align with previously generated content or instructions, we can improve the overall quality and coherence of model responses.

- The problem-solving process repeats until a it reaches a predetermined reason to stop.

- For example, it could run out of tokens or time, or the model could output a stop token.

For example
- question: prompts a model, "Write a short essay on literature."
  - The model might draft an essay, critique it for lack of specific examples, and rewrite the essay to include specific examples.
  - This process would repeat until the essay is deemed satisfactory or a stop criterion is met.


```bash
# Prompt:
"Generate a story beginning with the following sentence:"
"Continue the story from the previous prompt, ensuring consistency and coherence."
"Conclude the story in a meaningful and satisfying way."
# Example:
Prompt: "Generate a story beginning with the following sentence: 'Once upon a time in a small village…'"
Output: "Once upon a time in a small village, there lived a young girl named Emma who possessed a magical power."

Prompt: "Continue the story from the previous prompt, ensuring consistency and coherence."
Output: "Emma's magical power allowed her to communicate with animals, and she used this gift to help her community and protect the village from harm."

Prompt: "Conclude the story in a meaningful and satisfying way."
Output: "As the years went by, Emma's reputation as a guardian of the village grew, and her selflessness and bravery became legendary."

```


---

### Directional-stimulus prompting

- includes a hint or cue, such as desired keywords, to guide the language model toward the desired output.

For example
- question: the prompt is to write a poem about love,
  - the prompt engineer may craft prompts that include "heart," "passion," and "eternal."
  - The model might be prompted, "Write a poem about love that includes the words 'heart,' 'passion,' and 'eternal'."
  - This would guide the model to craft a poem with these keywords.


---

## prompt design Best practices


### Begin with the basics

- While embarking on the journey of designing prompts you need to remember that it’s a step-by-step process that demands persistent tweaking and testing to achieve excellence. Platforms like OpenAI or Cohere provide a user-friendly environment for this venture.

- Kick off with basic prompts, gradually enriching them with more components and context as you strive for enhanced outcomes.

- Maintaining different versions of the prompts is crucial in this progression. Through this guide, you will discover that clarity, simplicity, and precision often lead to superior results.

- For complex tasks involving numerous subtasks, consider deconstructing them into simpler components, progressively developing as you achieve promising results.

- This approach prevents an overwhelming start to the prompt design process.



### Crafting effective prompts (instructions)

- one of the most potent tools is the instruction you give to the language model.

- Instructions such as “Write,” “Classify,” “Summarize,” “Translate,” “Order,” etc., guide the model to execute a variety of tasks.

- Remember, crafting an effective instruction often involves a considerable amount of experimentation. To optimize the instruction for the specific use case, test different instruction patterns with varying keywords, contexts, and data types. The rule of thumb here is to ensure the context is as specific and relevant to the task as possible.

- Here is a practical tip:
  - most prompt designers suggest placing the instruction at the start of the prompt.
  - A clear separator, like “###”, could be used to distinguish the instruction from the context.

- For example:
  - `“### Instruction ### Translate the following text to French: Text: “Good morning!”`



### Unambiguous prompts

- Be Specific

- Clearly define the desired output

- provide precise instructions to guide the model’s response.

- Clearly define the desired response in the prompt to avoid misinterpretation by the AI.

- This helps the AI to focus only on the request and provide a response that aligns with the objective.

- Be Contextually Aware: Incorporate relevant context into the prompt to ensure the model understands the desired task or query.

- For instance
  - asking for a novel summary, clearly state that you are looking for a summary, not a detailed analysis.


### Adequate context within the prompt

- Provide adequate context within the prompt and include output requirements in the prompt input, confining it to a specific format.

- For instance
  - want a list of the most popular movies of the 1990s in a table. you should explicitly state how many movies you want to be listed and ask for table formatting.


### Balance between targeted information and desired output

- Keep it Concise

- Avoid overly long prompts that may confuse the model. Focus on essential instructions and information.

- Balance simplicity and complexity in the prompt to avoid vague, unrelated, or unexpected answers.
  - too simple may lack context
  - too complex may confuse the AI.
- This is especially important for complex topics or domain-specific language, which may be less familiar to the AI. Instead, use simple language and reduce the prompt size to make the question more understandable.


### Experiment and refine the prompt

- Prompt engineering is an iterative process. It's essential to experiment with different ideas and test the AI prompts to see the results. You may need multiple tries to optimize for accuracy and relevance. Continuous testing and iteration reduce the prompt size and help the model generate better output. There are no fixed rules for how the AI outputs information, so flexibility and adaptability are essential.

- Test and Iterate

- Experiment with different prompt designs and evaluate the model’s responses to refine and improve the prompt over time.






.
