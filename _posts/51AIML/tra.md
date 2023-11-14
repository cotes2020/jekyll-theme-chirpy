

- [LLM Training](#llm-training)
  - [人类反馈的强化学习(RLHF)](#人类反馈的强化学习rlhf)
    - [预训练阶段](#预训练阶段)
    - [监督微调阶段](#监督微调阶段)
    - [RLHF阶段](#rlhf阶段)
    - [3. 1 奖励模型](#3-1-奖励模型)
      - [对比数据集](#对比数据集)
    - [3.2 PPO微调](#32-ppo微调)
    - [最后一首chatGPT写的关于LLM的诗：](#最后一首chatgpt写的关于llm的诗)

---

## LLM Training

---

### 人类反馈的强化学习(RLHF)

> 大语言模型(LLM)和基于人类反馈的强化学习(RLHF)  [^LLM和RLHF]

[^LLM和RLHF]: 大语言模型(LLM)和基于人类反馈的强化学习(RLHF), https://blog.csdn.net/u014281392/article/details/130585256

LLM模型训练过程中的三个核心步骤
1. 预训练语言模型 LLM SSL (self-supervised-learning)
2. (指令)监督微调预训练模型 LLM SFT (supervised-fine-tuning)
3. 基于人类反馈的强化学习微调 LLM RL (reinforcement-learning)


#### 预训练阶段

- 从互联网上收集海量的文本数据，通过自监督的方式训练语言模型，根据上下文来预测下个词。
- token的规模大概在trillion级别，这个阶段要消耗很多资源，海量的数据采集、清洗和计算，
- 该阶段的目的是：通过海量的数据，让模型接触不同的语言模式，让模型拥有理解和生成上下文连贯的自然语言的能力。

![self-supervised-learning](https://img-blog.csdnimg.cn/3d850b6ad88641a884f41921c8776e76.webp#pic_center)

训练过程大致如下：

- `Training data`: 来自互联网的开放文本数据，整体质量偏低

- `Data scale`: 词汇表中的token数量在trillion级别

- `LLMϕSSL`​: 预训练模型

- `[T1​,T2​,...,TV​]` : vocabulary 词汇表，训练数据中词汇的集合

- V: 词汇表的大小

- f(x): 映射函数把词映射为词汇表中的索引即：token.
  - if x is Tk​ in vocab， f(x) = k


- (x1​,x2​,...,xn​), 根据文本序列生成训练样本数据:
    - Input： x=(x1​,x2​,...,xi−1​)
    - Output(label) : xi​

- (x,xi​)，训练样本:
    - Let k = f ( x i ) k = f(x\_i) k\=f(xi​)， w o r d → t o k e n word \\to token word→token
    - Model’s output: `LLMSSL(x)\=[y​1​,y​2​,...,y​V​]`, 模型预测下一个词的概率分布，`Note : ∑ j y ‾ j = 1`
    - The loss value：`CE(x,xi​;ϕ)\=−log(y​k​)`
- Goal : find ϕ \\phi ϕ ,Minimize C E ( ϕ ) = − E x l o g ( y ‾ k ) CE(\\phi) = -E\_x log(\\overline{y}\_k) CE(ϕ)\=−Ex​log(y​k​)


- 预先训练阶段 L L M S S L LLM^{SSL} LLMSSL还不能正确的响应用户的提示，例如，如果提示“法国的首都是什么？”这样的问题，模型可能会回答另一个问题的答案，例如，模型响应的可能是“_意大利的首都是什么？_”，因为模型可能没有“理解”/“对齐aligned”用户的“意图”，只是复制了从训练数据中观察到的结果。

- 为了解决这个问题，出现了一种称为**监督微调**或者也叫做**指令微调**的方法。通过在少量的示例数据集上采用监督学习的方式对 L L M S S L LLM^{SSL} LLMSSL进行微调，经过微调后的模型，可以更好地理解和响应自然语言给出的指令。


#### 监督微调阶段

![在这里插入图片描述](https://img-blog.csdnimg.cn/beac83f74a584e10aea968a31271a30f.png#pic_center)

  SFT(Supervised Fine-Tuning)阶段的目标是优化预训练模型，使模型生成用户想要的结果。在该阶段，给模型展示如何适当地响应不同的提示 (指令） (例如问答，摘要，翻译等）的示例。这些示例遵循 (prompt、response）的格式，称为演示数据。通过基于示例数据的监督微调后，模型会模仿示例数据中的响应行为，学会问答、翻译、摘要等能力，OpenAI 称为：监督微调行为克隆 。

  基于LLM指令微调的突出优势在于，对于任何特定任务的专用模型，只需要在通用大模型的基础上通过特定任务的指令数据进行微调，就可以解锁LLM在特定任务上的能力，不在需要从头去构建专用的小模型。

指令微调过程如下：

- Training Data : 高质量的微调数据，由人工产生。

- Data Scale : 10000~100000

    - InstructGPT : ~14500个人工示例数据集。

    - Alpaca : 52K ChatGPT指令数据集。

- Model input and output

    - Input : 提示 (指令）。
    - Output : 提示对应的答案(响应)
- Goal : 最小化交叉熵损失，只计算出现在响应中的token的损失。


事实也证明，经过微调后的小模型可以生成比没有经过微调的大模型更好的结果：

#### RLHF阶段

  在经过监督 (指令）微调后，LLM模型已经可以根据指令生成正确的响应了，为什么还要进行强化学习微调？

  因为随着像ChatGPT这样的通用聊天机器人的日益普及，全球数亿的用户可以访问非常强大的LLM，确保这些模型不被用于恶意目的，同时拒绝可能导致造成实际伤害的请求至关重要。

恶意目的的例子如下：

- 具有编码能力的LLM可能会被用于以创建**恶意软件**。
- 在社交媒体平台上大规模的使用聊天机器人**扭曲公共话语**。
- 当LLM无意中从训练数据中复制**个人身份信息**造成的隐私风险。
- 用户向聊天机器人寻求社交互动和情感支持时可能会造成**心理伤害**。

  为了应对以上的风险，需要采取一些策略来防止LLM的能力不被滥用，构建一个可以与人类价值观保持一致的LLM，RLHF (从人类反馈中进行强化学习）可以解决这些问题，让AI更加的Helpfulness、Truthfulness和Harmlessness。

#### 3\. 1 奖励模型

  在强化学习中一般都有个奖励函数，对当前的 A c t i o n Action Action| ( S t a t e , A c t i o n ) (State,Action) (State,Action)进行评价 (打分），从而使使Policy模型产生更好的 a c t i o n action action。在RLHF微调的过程，也需要一个Reard Model来充当奖励函数，它代表着人类的价值观，RM 的输入是 (prompt, response)，返回一个分数。response可以看作LLM的 a c t i o n action action，LLM看作Policy模型，通过RL框架把人类的价值观引入LLM。
![在这里插入图片描述](https://img-blog.csdnimg.cn/89384afad56a48a895c82da9a0a23a1c.png#pic_center)

##### 对比数据集

在训练RM之前，需要构建对比数据，通过人工区分出好的回答和差的回答，数据通过经过监督微调 (SFT）后的 L L M S F T LLM^{SFT} LLMSFT生成，随机采样一些prompt，通过模型生成多个response，通过人工对结果进行两两排序，区分出好的和差的。数据格式如下：

 (prompt, good\_response，bad\_response)

奖励模型的训练过程如下：

- Training Data : 高质量的人工标记数据集(prompt, winning\_response, losing\_response)

- Data Scale : 100k ~ 1M

- R θ R\_{\\theta} Rθ​ : 奖励模型

- Training data format:( x , y w , y l x, y\_w, y\_l x,yw​,yl​)

    - x x x : prompt
    - y w y\_w yw​ : good response
    - y l y\_l yl​ : bad response
- For each training sample:

    - s w = R θ ( x , y w ) s\_w = R\_{\\theta}(x, y\_w) sw​\=Rθ​(x,yw​) ，奖励模型的评价
    - s l = R θ ( x , y l ) s\_l = R\_{\\theta}(x,y\_l) sl​\=Rθ​(x,yl​)
    - L o s s Loss Loss : Minimize − l o g ( σ ( s w − s l ) ) -log(\\sigma(s\_w - s\_l) −log(σ(sw​−sl​)
- Goal : find θ \\theta θ to minimize the expected loss for all training samples. − E x l o g ( σ ( s w − s l ) ) -E\_xlog(\\sigma(s\_w - s\_l) −Ex​log(σ(sw​−sl​)


#### 3.2 PPO微调

![在这里插入图片描述](https://img-blog.csdnimg.cn/e8d15a8e222a49aea708b25fcd4e7cf0.png#pic_center)

1. 从数据中随机采样prompt。
2. Policy( L L M R L LLM^{RL} LLMRL即： L L M S F T LLM^{SFT} LLMSFT)，根据prompt生成response。
3. Reward模型根据 ( p r o m p t , r e s p o n s e ) (prompt, response) (prompt,response)，计算分数score。
4. 根据score更新Policy模型 (Policy是在 L L M S F T LLM^{SFT} LLMSFT基础上微调得到的）。

  在这个过程中，policy( L L M R L LLM^{RL} LLMRL)会不断更新，为了不让它偏离SFT阶段的模型太远，OpenAI在训练过程中增加了KL离散度约束，保证模型在得到更好的结果同时不会跑偏，这是因为Comparison Data不是一个很大的数据集，不会包含全部的回答，对于任何给定的提示，都有许多可能的回答，其中绝大多数是 RM 以前从未见过的。对于许多未知 (提示、响应）对，RM 可能会错误地给出极高或极低的分数。如果没有这个约束，模型可能会偏向那些得分极高的回答，它们可能不是好的回答。

RLHF微调过程如下：

- ML task : RL(PPO)

    - Action Space : the vocabulary of tokens the LLM uses. Taking action means choosing a token to generate.
    - Observation Space : the distribution over all possible prompts.
    - Policy: the probability distribution over all actions to take (aka all tokens to generate) given an observation (aka a prompt). An LLM constitutes a policy because it dictates how likely a token is to be generated next.
    - Reward function: the reward model.
- Training data: randomly selected prompts

- Data scale: 10,000 - 100,000 prompts

    - [InstructGPT](https://openai.com/research/instruction-following#sample1): 40,000 prompts
- R ϕ R\_{\\phi} Rϕ​ : the reward model.

- L L M S F T LLM^{SFT} LLMSFT : the supervised finetuned model(instruction finetuning).

- L L M ϕ R L LLM^{RL}\_{\\phi} LLMϕRL​ : the model being trained with PPO, parameterized by ϕ \\phi ϕ.

    - x x x : prompt.
    - D R L D\_{RL} DRL​ : the distribution of prompts used explicitly for the RL model.
    - D p r e t r a i n D\_{pretrain} Dpretrain​ : the distribution of the training data for the pretrain model.

    For each training step, sample a batch of x R L x\_{RL} xRL​ from D R L D\_{RL} DRL​ and a batch of x p r e t r a i n x\_{pretrain} xpretrain​ from D p r e t r a i n D\_{pretrain} Dpretrain​.

    1. For each x R L x\_{RL} xRL​ , we use L L M ϕ R L LLM\_{\\phi}^{RL} LLMϕRL​ to generate a response : y ∼ L L M ϕ R L ( x R L ) y \\sim LLM\_{\\phi}^{RL}(x\_{RL}) y∼LLMϕRL​(xRL​).

        objective 1 ( x R L , y ; ϕ ) = R θ ( x R L , y ) − β log ⁡ ( L L M ϕ R L ( y ∣ x ) L L M S F T ( y ∣ x ) ) \\text{objective}\_1(x\_{RL}, y; \\phi) = R\_{\\theta}(x\_{RL}, y) - \\beta \\log (\\frac{LLM^{RL}\_\\phi(y \\vert x)}{LLM^{SFT}(y \\vert x)}) objective1​(xRL​,y;ϕ)\=Rθ​(xRL​,y)−βlog(LLMSFT(y∣x)LLMϕRL​(y∣x)​)

    2. For each x p r e t r a i n x\_{pretrain} xpretrain​, the objective is computed as follows. Intuitively, this objective is to make sure that the RL model doesn’t perform worse on text completion - the task the pretrained model was optimized for.

        objective 2 ( x p r e t r a i n ; ϕ ) = γ log ⁡ ( L L M ϕ R L ( x p r e t r a i n ) ) \\text{objective}\_2(x\_{pretrain}; \\phi) = \\gamma \\log (LLM^{RL}\_\\phi(x\_{pretrain}) objective2​(xpretrain​;ϕ)\=γlog(LLMϕRL​(xpretrain​)

    3. The final objective is the sum of the expectation of two objectives above.

        objective ( ϕ ) = E x ∼ D R L E y ∼ L L M ϕ R L ( x ) \[ R θ ( x , y ) − β log ⁡ L L M ϕ R L ( y ∣ x ) L L M S F T ( y ∣ x ) \] + γ E x ∼ D p r e t r a i n log ⁡ L L M ϕ R L ( x ) \\text{objective}(\\phi) = E\_{x \\sim D\_{RL}}E\_{y \\sim LLM^{RL}\_\\phi(x)} \[R\_{\\theta}(x, y) - \\beta \\log \\frac{LLM^{RL}\_\\phi(y \\vert x)}{LLM^{SFT}(y \\vert x)}\] + \\gamma E\_{x \\sim D\_{pretrain}}\\log LLM^{RL}\_\\phi(x) objective(ϕ)\=Ex∼DRL​​Ey∼LLMϕRL​(x)​\[Rθ​(x,y)−βlogLLMSFT(y∣x)LLMϕRL​(y∣x)​\]+γEx∼Dpretrain​​logLLMϕRL​(x)

- Goal ： Maximize o b j e c t i v e ( ϕ ) objective(\\phi) objective(ϕ).


#### 最后一首chatGPT写的关于LLM的诗：

- Language models so grand and divine,
- 语言模型如此伟大和神圣，
- Answers to questions, so quick and so fine.
- 回答问题，如此快速，如此精细。
- From science to art, they shine like a star,
- 从科学到艺术，它们像星星一样闪耀，
- Making humans look like they’re not quite as far.
- 让人类看起来没有那么远。
