
# NER

## overall

NER
- 命名实体识别
- Named Entity Recognition
- 就是从一段文本中抽取到找到任何你想要的东西, 可能是某个字, 某个词, 或者某个短语。通常是用序列标注(Sequence Tagging)的方式来做, 老 NLP task 了
- 是指识别文本中具有特定意义的实体，主要包括人名、地名、机构名、专有名词等。
- NER是信息提取、问答系统、句法分析、机器翻译、面向Semantic Web的元数据标注等应用领域的重要基础工具，在自然语言处理技术走向实用化的过程中占有重要的地位。
- 在搜索场景下，NER是深度查询理解（Deep Query Understanding，简称 DQU）的底层基础信号，主要应用于搜索召回、用户意图识别、实体链接等环节，NER信号的质量，直接影响到用户的搜索体验。


实现NER的方法
- 序列标注 Sequence Tagging
  - NER ≠ 序列标注
  - 序列标注只是实现NER的其中一种方法, 且并不一定是最好的那种方法。还有一些其他方法
- 指针标注(标记实体start与end的position)、
- 片段排列标注(枚举所有n-gram判断是否是实体)等。


传统序列标注方式有很多难搞的场景, 举两个最主要的：

1. `多实体`有`交叉重叠`：
   1. 例如“马亲王发布新书长安十二时辰”, 其中“长安”和“长安十二时辰”可能都是待抽取实体, 一个地名一个书名, 两个我都要, 怎么搞
2. 实体名`非连续`：
   1. 例如“我这有iphone11和12”, 包含了两个实体, “iphone11”很好抽, “iphone12”可太难了

新潮的 NER 以及信息抽取方面的研究, 都会考虑到这些问题, 然后使用基于`指针标注`或`片段排列标注`的新方法来取代传统序列标注方法。
- 在模型方面, 现在的 NER 已经不是 LSTM-CRF & BERT-CRF 大一统的时代了。

---


## Sequence Tagging 传统序列标注方式

---

### 命名实体识别实践与探索


为什么说流水的NLP铁打的NER？[^流水的NLP铁打的NER]
- NLP四大任务嘛,  **分类、生成、序列标注、句子对标注** 。
- **分类任务**
  - 面太广了, 万物皆可分类, 各种方法层出不穷；
  - 句子对标注, 经常是体现人工智能对人类语言理解能力的标准秤, 孪生网络、DSSM、ESIM 各种模型一年年也是秀的飞起；
- **生成任务**
  - 目前人工智障 NLP 能力的天花板, 虽然经常会处在说不出来人话的状态, 但也不断吸引 CopyNet、VAE、GAN 各类选手前来挑战；
- 唯有**序列标注**
  - 数年如一日, 不忘初心, 原地踏步
  - 到现在一提到 NER, 还是会一下子只想到 LSTM-CRF, 铁打不动的模型, 没得挑也不用挑, 用就完事了, 不用就是不给面子

[^流水的NLP铁打的NER]: 流水的NLP铁打的NER：命名实体识别实践与探索, https://zhuanlan.zhihu.com/p/166496466

- 命名实体识别虽然是一个历史悠久的老任务了, 但是自从2015年有人使用了 LSTM-CRF 模型之后, 这个模型和这个任务简直是郎才女貌, 天造地设, 轮不到任何妖怪来反对。直到后来出现了BERT。

两个问题：
1. 2015-2019年, BERT出现之前4年的时间, 命名实体识别就只有 **LSTM-CRF** 了吗？
2. 2019年BERT出现之后, 命名实体识别就只有 **BERT-CRF** (或者 `BERT-LSTM-CRF`)了吗？

现在的NER还在做的事情, 主要分几个方面:

1. **多特征**:
   1. 实体识别不是一个特别复杂的任务, 不需要太深入的模型, 那么就是加特征, 特征越多效果越好,
   2. 所以字特征、词特征、词性特征、句法特征、KG表征等等的就一个个加吧, 甚至有些中文 NER 任务里还加入了拼音特征、笔画特征。。
   3. 心有多大, 特征就有多多

2. **多任务**:
   1. 很多时候做 NER 的目的并不仅是为了 NER, 而是服务于一个更大的目标或系统, 比如信息抽取、问答系统等等。
   2. 如果把整个大任务做一个端到端的模型, 就需要做成一个多任务模型, 把 NER 作为其中一个子任务；
   3. 另外, 单纯的 NER 也可以做成多任务, 比如实体类型过多时, 仅用一个序列标注任务来同时抽取实体与判断实体类型, 会有些力不从心, 就可以拆成两个子任务来做

3. **时令大杂烩**:
   1. 把当下比较流行的深度学习话题或方法跟 NER 结合一下, 比如
      1. 结合强化学习的 NER、
      2. 结合 few-shot learning 的 NER、
      3. 结合多模态信息的 NER、
      4. 结合跨语种学习的 NER 等等的, 具体就不提了

所以沿着上述思路, 就在一个中文NER任务上做一些实践, 写一些模型。都列在下面了,
- 首先是 **LSTM-CRF** 和 **BERT-CRF** ,
- 然后就是几个多任务模型,
- Cascade 开头的(因为实体类型比较多, 把NER拆成两个任务, 一个用来识别实体, 另一个用来判断实体类型),
- WLF Word Level Feature(即在原本字级别的序列标注任务上加入词级别的表征)
- WOL Weight of Loss(即在loss函数方面通过设置权重来权衡Precision与Recall, 以达到提高F1的目的)

![](https://pic2.zhimg.com/v2-3062da7d38adce1213af496239f04bd9_b.jpg)

* 代码：上述所有模型的代码都在这里, 带 BERT 的可以自己去下载 [BERT_CHINESE](https://link.zhihu.com/?target=https%3A//storage.googleapis.com/bert_models/2018_11_03/chinese_L-12_H-768_A-12.zip) 预训练的 ckpt 模型, 然后解压到 bert_model 目录下

* 环境：Python3, Tensorflow1.12

* 数据：一个电商场景下商品标题中的实体识别, 因为是工作中的数据, 并且通过远程监督弱标注的质量也一般, 完整数据就不放了。但是我 sample 了一些数据留在 git 里了, 直接 git clone 完, 代码原地就能跑

用纯 HMM 或者 CRF 做 NER 的话就不讲了, 比较古老了。

从 LSTM-CRF 开始讲起

#### 1. BI-LSTM-CRF (Bi-directional LSTM)

**LSTM-CRF**
- 应该是2015年被提出的模型 [^Bidirectional_LSTM-CRF]
- 模型架构在今天来看非常简单

- BI-LSTM 即 Bi-directional LSTM
  - 就是有两个 LSTM cell,
  - 一个从左往右跑得到第一层表征向量 _l_
  - 一个从右往左跑得到第二层向量 _r_
  - 然后两层向量加一起得到第三层向量 _c_

- 如果不使用CRF的话, 这里就可以直接接一层全连接与`softmax`, 输出结果了；
- 如果用CRF的话, 需要把 _c_ 输入到 CRF 层中, 经过 CRF 一通专业缜密的计算, 它来决定最终的结果

[^Bidirectional_LSTM-CRF]: Bidirectional LSTM-CRF Models for Sequence Tagging, https://arxiv.org/pdf/1508.01991.pdf

![](https://pic3.zhimg.com/v2-16458a338f695c6cbe82532af3b84cc6_b.jpg)


BIO 标记法
- 用于表示序列标注结果的 BIO 标记法。
- 序列标注里标记法有很多, 最主要的还是 BIO 与 BIOES 这两种。
  - B 就是标记某个实体词的开始,
  - I 表示某个实体词的中间,
  - E 表示某个实体词的结束,
  - S 表示这个实体词仅包含当前这一个字。
- 区别很简单, 看图就懂。
- 一般实验效果上差别不大, 有些时候用 BIOES 可能会有一内内的优势

![](https://pic1.zhimg.com/v2-d95ef52e02af82bed6740a003c141db8_b.jpg)

- 如果在某些场景下不考虑实体类别(比如问答系统), 那就直接完事了
- 但是很多场景下需要同时考虑实体类别(比如事件抽取中需要抽取主体客体地点机构等等), 那么就需要扩展 BIO 的 tag 列表, 给每个“实体类型”都分配一个 B 与 I 的标签, 例如用“B-brand”来代表“实体词的开始, 且实体类型为品牌”。
- 当实体类别过多时, BIOES 的标签列表规模可能就爆炸了

基于 Tensorflow 来实现 **LSTM-CRF** 代码也很简单, 直接上

```py
self.inputs_seq = tf.placeholder(tf.int32, [None, None], name="inputs_seq") # B * S
self.inputs_seq_len = tf.placeholder(tf.int32, [None], name="inputs_seq_len") # B
self.outputs_seq = tf.placeholder(tf.int32, [None, None], name='outputs_seq') # B * S

with tf.variable_scope('embedding_layer'):
    embedding_matrix = tf.get_variable("embedding_matrix", [vocab_size_char, embedding_dim], dtype=tf.float32)
    embedded = tf.nn.embedding_lookup(embedding_matrix, self.inputs_seq) # B * S * D

with tf.variable_scope('encoder'):
    cell_fw = tf.nn.rnn_cell.LSTMCell(hidden_dim)
    cell_bw = tf.nn.rnn_cell.LSTMCell(hidden_dim)
    ((rnn_fw_outputs, rnn_bw_outputs), (rnn_fw_final_state, rnn_bw_final_state)) = tf.nn.bidirectional_dynamic_rnn(
        cell_fw=cell_fw,
        cell_bw=cell_bw,
        inputs=embedded,
        sequence_length=self.inputs_seq_len,
        dtype=tf.float32
    )
    rnn_outputs = tf.add(rnn_fw_outputs, rnn_bw_outputs) # B * S * D

with tf.variable_scope('projection'):
    logits_seq = tf.layers.dense(rnn_outputs, vocab_size_bio) # B * S * V
    probs_seq = tf.nn.softmax(logits_seq) # B * S * V
    if not use_crf:
        preds_seq = tf.argmax(probs_seq, axis=-1, name="preds_seq") # B * S
    else:
        log_likelihood, transition_matrix = tf.contrib.crf.crf_log_likelihood(logits_seq, self.outputs_seq, self.inputs_seq_len)
        preds_seq, crf_scores = tf.contrib.crf.crf_decode(logits_seq, transition_matrix, self.inputs_seq_len)

with tf.variable_scope('loss'):
    if not use_crf:
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits_seq, labels=self.outputs_seq) # B * S
        masks = tf.sequence_mask(self.inputs_seq_len, dtype=tf.float32) # B * S
        loss = tf.reduce_sum(loss * masks, axis=-1) / tf.cast(self.inputs_seq_len, tf.float32) # B
    else:
        loss = -log_likelihood / tf.cast(self.inputs_seq_len, tf.float32) # B
```

Tensorflow 里调用 **CRF** 非常方便
- 主要就 `crf_log_likelihood` 和 `crf_decode` 这两个函数, 结果和 loss 就都给你算出来了。
- 它要学习的参数也很简单, 就是这个 `transition_matrix`, 形状为 $V*V$, $V$ 是输出端 BIO 的词表大小。

但是有一个小小的缺点
- 就是官方实现的 `crf_log_likelihood` 里某个未知的角落有个 stack 操作, 会悄悄地吃掉很多的内存。
- 如果 $V$ 较大, 内存占用量会极高, 训练时间极长。比如有 500 个实体类别, 也就是 $V=500*2+1=1001$, 训练 1epoch 的时间从 30min 暴增到 400min
- 好消息是, Tensorflow2.0 里, 这个问题不再有了
- 坏消息是, Tensorflow2.0 直接把 tf.contrib.crf 移除了, 目前还没有官方实现的 CRF 接口

```py
/usr/local/lib/python3.6/dist-packages/tensorflow/python/ops/gradients_impl.py:112: UserWarning: Converting sparse IndexedSlices to a dense Tensor of unknown shape. This may consume a large amount of memory.
  "Converting sparse IndexedSlices to a dense Tensor of unknown shape. "
```

![](https://pic3.zhimg.com/v2-40caa1b4733671f5ccabae99f3d6f216_b.png)


再说一下为什么要加 CRF。
- 从开头的 Leaderboard 里可以看到, BiLSTM 的 F1 Score 在72%, 而 BiLSTM-CRF 达到 80%, 提升明显

![](https://pic2.zhimg.com/v2-61255f9651ab75b02b7200cdc3d1bec5_b.jpg)


那么为什么提升这么大呢？
- CRF 的原理, 网上随便搜就一大把, 就不讲了(因为的确很难, 我也没太懂)
- 从实验的角度可以简单说说, 就是 LSTM 只能通过输入判断输出, 但是 CRF 可以通过学习`转移矩阵`, `看前后的输出来判断当前的输出`。
- 这样就能学到一些规律(比如“O 后面不能直接接 I” “B-brand 后面不可能接 I-color”), 这些规律在有时会起到至关重要的作用

- 例子:
  - A 是没加 CRF 的输出结果, B 是加了 CRF 的输出结果, 一看就懂不细说了
  - ![](https://pic1.zhimg.com/v2-694e0210c9672c3565558104fbc7bcc8_b.jpg)


#### 2. BERT-CRF & BERT-LSTM-CRF

- 用 BERT 来做, 结构上跟上面是一样的, 只是把 LSTM 换成 BERT 就 ok 了, 直接上代码

- 首先把 BERT 这部分模型搭好, 直接用 BERT 的官方代码。
- 这里我把序列长度都标成了“S+2”是为了提醒自己每条数据前后都加了“[CLS]”和“[SEP]”, 出结果时需要处理掉

```py
from bert import modeling as bert_modeling

self.inputs_seq = tf.placeholder(
    shape=[None, None], dtype=tf.int32, name="inputs_seq")
    # B * (S+2)
self.inputs_mask = tf.placeholder(
    shape=[None, None], dtype=tf.int32, name="inputs_mask")
    # B * (S+2)
self.inputs_segment = tf.placeholder(
    shape=[None, None], dtype=tf.int32, name="inputs_segment")
    # B * (S+2)
self.outputs_seq = tf.placeholder(
    shape=[None, None], dtype=tf.int32, name='outputs_seq')
    # B * (S+2)

bert_config = bert_modeling.BertConfig.from_json_file("./bert_model/bert_config.json")

bert_model = bert_modeling.BertModel(
    config=bert_config,
    is_training=True,
    input_ids=self.inputs_seq,
    input_mask=self.inputs_mask,
    token_type_ids=self.inputs_segment,
    use_one_hot_embeddings=False
)

bert_outputs = bert_model.get_sequence_output() # B * (S+2) * D
```

然后在后面接东西就可以了, 可以接 LSTM, 可以接 CRF

```py
if not use_lstm:
    hiddens = bert_outputs
else:
    with tf.variable_scope('bilstm'):
        cell_fw = tf.nn.rnn_cell.LSTMCell(300)
        cell_bw = tf.nn.rnn_cell.LSTMCell(300)
        ((rnn_fw_outputs, rnn_bw_outputs), (rnn_fw_final_state, rnn_bw_final_state)) = tf.nn.bidirectional_dynamic_rnn(
            cell_fw=cell_fw,
            cell_bw=cell_bw,
            inputs=bert_outputs,
            sequence_length=inputs_seq_len,
            dtype=tf.float32
        )
        rnn_outputs = tf.add(rnn_fw_outputs, rnn_bw_outputs) # B * (S+2) * D
    hiddens = rnn_outputs

with tf.variable_scope('projection'):
    logits_seq = tf.layers.dense(hiddens, vocab_size_bio) # B * (S+2) * V
    probs_seq = tf.nn.softmax(logits_seq)

    if not use_crf:
        preds_seq = tf.argmax(probs_seq, axis=-1, name="preds_seq") # B * (S+2)
    else:
        log_likelihood, transition_matrix = tf.contrib.crf.crf_log_likelihood(logits_seq, self.outputs_seq, inputs_seq_len)
        preds_seq, crf_scores = tf.contrib.crf.crf_decode(logits_seq, transition_matrix, inputs_seq_len)
```

- BERT确实强,
- 相比较单纯使用 BERT, 增加了 CRF 后效果有所提高但区别不大, 再增加 BiLSTM 后区别很小, 甚至降低了那么一内内

![](https://pic3.zhimg.com/v2-60c96e63016a6aaa69bda23d66580ae6_b.jpg)

另外, BERT 还有一个 **至关重要** 的训练技巧, 就是调整学习率。
- BERT内的参数在 fine-tuning 时, 学习率一定要调小, 特别后面还接了别的东西时, 一定要按两个学习率走, 甚至需要尝试多次反复调, 要不然 BERT 很容易就步子迈大了掉沟里爬不上来

参数优化时分两个学习率, 实现起来就是这样

```py
with tf.variable_scope('opt'):
    params_of_bert = []
    params_of_other = []
    for var in tf.trainable_variables():
        vname = var.name
        if vname.startswith("bert"):
            params_of_bert.append(var)
        else:
            params_of_other.append(var)
    opt1 = tf.train.AdamOptimizer(1e-4)
    opt2 = tf.train.AdamOptimizer(1e-3)
    gradients_bert = tf.gradients(loss, params_of_bert)
    gradients_other = tf.gradients(loss, params_of_other)
    gradients_bert_clipped, norm_bert = tf.clip_by_global_norm(gradients_bert, 5.0)
    gradients_other_clipped, norm_other = tf.clip_by_global_norm(gradients_other, 5.0)
    train_op_bert = opt1.apply_gradients(zip(gradients_bert_clipped, params_of_bert))
    train_op_other = opt2.apply_gradients(zip(gradients_other_clipped, params_of_other))
```

#### 3. Cascade

- 如果需要考虑实体类别, 那么就需要扩展 BIO 的 tag 列表, 给每个“实体类型”都分配一个 B 与 I 的标签, 但是当类别数较多时, 标签词表规模很大, 相当于在每个字上都要做一次类别数巨多的分类任务, 不科学, 也会影响效果

- 从这个点出发, 就尝试把 NER 改成一个多任务学习的框架, 两个任务, 一个任务用来`单纯抽取实体`, 一个任务用来`判断实体类型`

- 直接上图看区别

![](https://pic2.zhimg.com/v2-8b05c6ee1d3107e3aab149679c96416d_b.jpg)

- 这个是参考 ACL2020 的一篇论文[^Novel_Cascade]的思路改的

[^Novel_Cascade]: A Novel Cascade Binary Tagging Framework for Relational Triple Extraction, https://www.aclweb.org/anthology/2020.acl-main.136.pdf

- “Cascade”这个词是这个论文里提出来的
  - 翻译过来就是“级联”, 直观来讲就是“锁定对应关系”。
  - 结合模型来说, 在第一步得到实体识别的结果之后, 返回去到 LSTM 输出那一层, 找各个实体词的表征向量, 然后再把实体的表征向量输入一层全连接做分类, 判断实体类型

- 如何得到实体整体的表征向量
  - 论文里是把各个实体词的向量做平均
  - 看了源码, 好像只把每个实体最开头和最末尾的两个词做了平均。
  - 更省事, 只取了每个实体最末尾的一个词

- 具体实现上这样写：
  - 在训练时, 每个词, 无论是不是实体词, 都过一遍全连接, 做实体类型分类计算 loss, 然后把非实体词对应的 loss 给 mask 掉；
  - 在预测时, 就取实体最后一个词对应的分类结果, 作为实体类型。上图解释

![](https://pic4.zhimg.com/v2-3880b0ef4c560dcdf570c7be3cf22417_b.jpg)

- 效果:
  - 将单任务 NER 改成多任务 NER 之后,
  - 基于 LSTM 的模型效果降低了 0.4%,
  - 基于 BERT 的模型提高了 1.7%, 整体还是提高更明显。
  - 另外, 由于 BIO 词表得到了缩减, CRF 运行时间以及消耗内存迅速减少, 训练速度得到提高

![](https://pic1.zhimg.com/v2-02a9124f11ce54129326700cbe6a9cb4_b.jpg)

- NER 中的实体类型标签较多的问题
  - 一篇文章 [^Scaling_Up_Open_Tagging], 这篇论文主要就是为了解决实体类型标签过多的问题(成千上万的数量级)。
  - 文中的方法是：把标签作为输入, 也就是把所有可能的实体类型标签都一个个试一遍, 根据输入的标签不同, 模型会有不同的实体抽取结果。
  - "复现了一下, 效果并不好, 具体表现就是无论输入什么标签, 模型都倾向于把所有的实体都抽出来, 不管这个实体是不是对应这个实体类型标签。"

[^Scaling_Up_Open_Tagging]: Scaling Up Open Tagging from Tens to Thousands: Comprehension Empowered Attribute Value Extraction from Product Title. ACL 2019, https://link.zhihu.com/?target=https%3A//www.aclweb.org/anthology/P19-1514.pdf

#### 4. Word-Level Feature

- 中文 NER 和英文 NER 有个比较明显的区别, 就是英文 NER 是从单词级别(`word level`)来做, 而中文 NER 一般是字级别(`character level`)来做。
- 不仅是 NER, 很多 NLP 任务也是这样, BERT 也是这样

- 因为中文没法天然分词, 只能靠分词工具, 分出来的不一定对,
  - 比如“黑 **啤酒** 精酿”, 如果被错误分词为“黑 **啤** 、 **酒** 精、酿”, 那么“ **啤酒** ”这个实体就抽取不到了。
  - 类似情况有很多

- 但是无论字级别、词级别, 都是非常贴近文本原始内容的特征, 蕴含了很重要的信息。
  - 比如对于英文来说, 给个单词“Geilivable”你基本看不懂啥意思, 但是看到它以“-able”结尾, 就知道可能不是名词；
  - 对于中文来说, 给个句子“小龙女说我也想过过过儿过过的生活”就一时很难找到实体在哪, 但是如果分好词给你, 一眼就能找到了。
  - 就这个理解力来说, 模型跟人是一样的

- 在英文 NLP 任务中, 想要把`字级别特征`加入到`词级别特征`上去, 一般是这样：
  - 单独用一个BiLSTM 作为 character-level 的编码器, 把单词的各个字拆开, 送进 LSTM 得到向量 $v^c$；
  - 然后和原本 word-level 的(经过 embedding matrix 得到的)的向量 $v^w$ 加在一起, 就能得到融合两种特征的表征向量。
  - 如图所示
  - ![](https://pic2.zhimg.com/v2-8f04e8dd2026fc833f0b5f14452aae2d_b.jpg)


- 对于中文 NER 任务, 我的输入是字级别的, 怎么把词级别的表征结果加入进来呢？
  - ACL2018 有个文章[^Chinese_NER]是做这个的, 提出了一种 `Lattice-LST`M 的结构, 但是涉及比较底层的改动, 不好实现。
  - 后来在 ACL2020 论文里看到一篇文章[^Simplify_Lexicon_in_Chinese], 简单明了。
  - 再简化一下, 直接把字和词分别通过 embedding matrix 做表征, 按照对应关系, 拼在一起就完事了,
  - 看图就懂
  - ![](https://pic4.zhimg.com/v2-033f184f1f19ef97d92cf1469375712b_b.jpg)
  - 从结果上看, 增加了词级别特征后, 提升很明显
  - ![](https://pic4.zhimg.com/v2-8c7b4444aa266827053f87726b9146c3_b.png)

[^Chinese_NER]: Chinese NER Using Lattice LSTM, https://arxiv.org/pdf/1805.02023.pdf

[^Simplify_Lexicon_in_Chinese]: Simplify the Usage of Lexicon in Chinese NER, https://www.aclweb.org/anthology/2020.acl-main.528.pdf

  - 很可惜, 我还没有找到把词级别特征结合到 BERT 中的方法。
  - 因为 BERT 是字级别预训练好的模型, 如果单纯从 embedding 层这么拼接, 那后面那些 Transformer 层的参数就都失效了

- 上面的论文里也提到了和 BERT 结合的问题, 论文里还是用 LSTM 来做, 只是把句子通过 BERT 得到的编码结果作为一个“额外特征”拼接过来。但是我觉得这不算“结合”, 至少不应该。但是也非常容易理解为什么论文里要这么做, BERT 当道的年代, 不讲道理, 打不过就只能加入, 方法不同也得强融, 么得办法


#### 5. Weight of Loss

- 大多数 NLP task 的评价指标有这三个：`Precision / Recall / F1Score`
  - Precision 是找出来的有多少是正确的
  - Recall 是正确的有多少被找出来了
  - F1Score 是二者的一个均衡分

- 这里有三点常识
  1. 方法固定的条件下, 一般来说, 提高了 Precision 就会降低 Recall, 提高了 Recall 就会降低 Precision, 结合指标定义很好理解
  2. 通常来说, F1Score 是最重要的指标, 为了让 F1Score 最大化, 通常需要调整权衡 Precision 与 Recall 的大小, 让两者达到近似, 此时 F1Score 是最大的
  3. 但是 F1Score 大, 不代表模型就好。因为结合工程实际来说, 不同场景不同需求下, 对 P/R 会有不同的要求。有些场景就是要求准, 不允许出错, 所以对 Precision 要求比较高, 而有些则相反, 不希望有漏网之鱼, 所以对 Recall 要求高

- 对于一个分类任务, 是很容易通过设置一个可调的“阈值”来达到控制 P/R 的目的的。
  - 例子, 判断一张图是不是 H 图,
  - 做一个二分类模型, 假设模型认为图片是 H 图的概率是 p, 人为设定一个阈值 a, 假如 p>a 则认为该图片是 H 图。默认情况 p=0.5, 此时如果降低 p, 就能达到提高 Recall 降低 Precision 的目的

- 但是 NER 任务怎么整呢, 他的结果是一个完整的序列, 你又不能给每个位置都卡一个阈值, 没有意义
  - 一个办法, 通过控制模型学习时的 Loss 来控制 P/R：
  - 如果模型没有识别到一个本应该识别到的实体, 就增大对应的 Loss, 加重对模型的惩罚；
  - 如果模型识别到了一个不应该识别到的实体, 就减小对应的 Loss, 当然是选择原谅他
  - 实现上也是通过 mask 来实现
  - ![](https://pic1.zhimg.com/v2-255491e3dbd35ba55f873fc5f56287e0_b.jpg)

实现也非常简单, 对应的代码:


```py
# logits_bio 是预测结果，形状为 B*S*V，softmax 之后就是每个字在BIO词表上的分布概率，不过不用写softmax，因为下面的函数会帮你做
# self.outputs_seq_bio 是期望输出，形状为 B*S
# 这是原本计算出来的 loss
loss_bio = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits_bio, labels=self.outputs_seq_bio) # B * S
# 这是根据期望的输出，获得 mask 向量，向量里出现1的位置代表对应的字是一个实体词，而 O_tag_index 就是 O 在 BIO 词表中的位置
masks_of_entity = tf.cast(tf.not_equal(self.outputs_seq_bio, O_tag_index), tf.float32) # B * S
# 这是基于 mask 计算 weights
weights_of_loss = masks_of_entity + 0.5 # B  *S
# 这是加权后的 loss
loss_bio = loss_bio * weights_of_loss # B * S
```

- 但是很可惜, 我还不知道怎么把这个方法和 CRF 结合起来。因为在代码里, CRF 通过函数crf_log_likelihood 直接计算得到整个句子级别的 loss, 而不是像上面一样, 用交叉熵在每个字上计算 loss, 所以这种基于 mask 的方法就没法用了

- 但是从实验效果来看, 虽然去掉了 CRF, 但是加入 WOL 之后的方法的 F1Score 还是要大一些。原本 Precision 远大于 Recall, 通过权衡, 把两个分数拉到同个水平, 可以提升最终的 F1Score

![](https://pic1.zhimg.com/v2-89beab0511aeca0822f8386d0edf04c8_b.jpg)

除此之外, 在所有深度学习任务上, 都可以通过调整 Loss 来达到各种特殊的效果, 还是挺有意思的, 放飞想象, 突破自我

## NER 应用

### NER 在搜索召回中的应用

实体识别在搜索召回中的应用 [^美团搜索中NER技术的探索与实践]
- 在O2O搜索中，对商家POI的描述是商家名称、地址、品类等多个互相之间相关性并不高的文本域。
- 如果对O2O搜索引擎也采用全部文本域命中求交的方式，就可能会产生大量的误召回。
- 我们的解决方法如下图1所示，让特定的查询只在特定的文本域做倒排检索，我们称之为“结构化召回”，可保证召回商家的强相关性。
- 举例来说，对于“海底捞”这样的请求，有些商家地址会描述为“海底捞附近几百米”，若采用全文本域检索这些商家就会被召回，显然这并不是用户想要的。
- 而结构化召回基于`NER将“海底捞”识别为商家`，然后只在商家名相关文本域检索，从而只召回海底捞品牌商家，精准地满足了用户需求。

[^美团搜索中NER技术的探索与实践]: https://tech.meituan.com/2020/07/23/ner-in-meituan-nlp.html

![图1 实体识别与召回策略](https://p0.meituan.net/travelcube/eb3b70f7a58883469170264b8bc3cebc181390.png@1120w_390h_80q)


有别于其他应用场景，美团搜索的NER任务具有以下特点：

* **新增实体数量庞大且增速较快**：本地生活服务领域发展迅速，新店、新商品、新服务品类层出不穷；用户Query往往夹杂很多非标准化表达、简称和热词（如“牵肠挂肚”、“吸猫”等），这对实现高准确率、高覆盖率的NER造成了很大挑战。

* **领域相关性强**：搜索中的实体识别与业务供给高度相关，除通用语义外需加入业务相关知识辅助判断，比如“剪了个头发”，通用理解是泛化描述实体，在搜索中却是个商家实体。

* **性能要求高**：从用户发起搜索到最终结果呈现给用户时间很短，NER作为DQU的基础模块，需要在毫秒级的时间内完成。近期，很多基于深度网络的研究与实践显著提高了NER的效果，但这些模型往往计算量较大、预测耗时长，如何优化模型性能，使之能满足NER对计算时间的要求，也是NER实践中的一大挑战。


#### 技术选型

针对O2O领域NER 任务的特点，我们整体的技术选型是“实体词典匹配+模型预测”的框架

![图2 实体识别整体架构](https://p0.meituan.net/travelcube/781201d27843e279a81ace4336dd5d0b156164.png@1256w_1166h_80q)

实体词典匹配和模型预测两者解决的问题各有侧重，在当前阶段缺一不可。下面通过对三个问题的解答来说明我们为什么这么选。

**为什么需要实体词典匹配？**

- 主要有以下四个原因：

  - 搜索中用户查询的头部流量通常较短、表达形式简单，且集中在商户、品类、地址等三类实体搜索，实体词典匹配虽简单但处理这类查询准确率也可达到90%以上。

  - NER下游使用方中有些对响应时间要求极高，词典匹配速度快，基本不存在性能问题。

  - NER领域相关，通过挖掘业务数据资源获取业务实体词典，经过在线词典匹配后可保证识别结果是领域适配的。

  - 新业务接入更加灵活，只需提供业务相关的实体词表就可完成新业务场景下的实体识别。


**有了实体词典匹配为什么还要模型预测？**

- 有以下两方面的原因：

  - 随着搜索体量的不断增大，中长尾搜索流量表述复杂，越来越多OOV（Out Of Vocabulary）问题开始出现，实体词典已经无法满足日益多样化的用户需求，模型预测具备泛化能力，可作为词典匹配的有效补充。

  - 实体词典匹配无法解决歧义问题，比如“黄鹤楼美食”，“黄鹤楼”在实体词典中同时是武汉的景点、北京的商家、香烟产品，词典匹配不具备消歧能力，这三种类型都会输出，而模型预测则可结合上下文，不会输出“黄鹤楼”是香烟产品。

**实体词典匹配、模型预测两路结果是怎么合并输出的？**

- 目前我们采用训练好的`CRF权重网络`作为打分器，来对实体词典匹配、模型预测两路输出的NER路径进行打分。在词典匹配无结果或是其路径打分值明显低于模型预测时，采用模型识别的结果，其他情况仍然采用词典匹配结果。

---

#### 实体词典匹配

传统的NER技术仅能处理通用领域既定、既有的实体，但无法应对垂直领域所特有的实体类型。
- 在美团搜索场景下，通过对POI结构化信息、商户评论数据、搜索日志等独有数据进行离线挖掘，可以很好地解决领域实体识别问题。
- 经过离线实体库不断的丰富完善累积后，在线使用轻量级的词库匹配实体识别方式简单、高效、可控，且可以很好地覆盖头部和腰部流量。
- 目前，基于实体库的在线NER识别率可以达到92%。

##### 离线挖掘

- 美团具有丰富多样的结构化数据，通过对领域内结构化数据的加工处理可以获得`高精度的初始实体库`。
  - 例如：
  - 从商户基础信息中，可以获取商户名、类目、地址、售卖商品或服务等类型实体。
  - 从猫眼文娱数据中，可以获取电影、电视剧、艺人等类型实体。
  - 然而，用户搜索的实体名往往夹杂很多非标准化表达，与业务定义的标准实体名之间存在差异，如何从非标准表达中挖掘领域实体变得尤为重要。

- 现有的**新词挖掘技术**主要分为无监督学习、有监督学习和远程监督学习
  - **无监督学习**
    - 通过频繁序列产生候选集，并通过计算紧密度和自由度指标进行筛选，这种方法虽然可以产生充分的候选集合，但仅通过特征阈值过滤无法有效地平衡精确率与召回率，现实应用中通常挑选较高的阈值保证精度而牺牲召回。
  - 先进的新词挖掘算法大多为**有监督学习**
    - 这类算法通常涉及复杂的语法分析模型或深度网络模型，且依赖领域专家设计繁多规则或大量的人工标记数据。
  - **远程监督学习**
    - 通过开源知识库生成少量的标记数据，虽然一定程度上缓解了人力标注成本高的问题。然而小样本量的标记数据仅能学习简单的统计模型，无法训练具有高泛化能力的复杂模型。

- 我们的离线实体挖掘是多源多方法的，涉及到的数据源包括结构化的商家信息库、百科词条，半结构化的搜索日志，以及非结构化的用户评论（UGC）等。使用的挖掘方法也包含多种，包括规则、传统机器学习模型、深度学习模型等。
- UGC作为一种非结构化文本，蕴含了大量非标准表达实体名。下面我们将详细介绍一种针对UGC的垂直领域新词自动挖掘方法，该方法主要包含三个步骤，如下图3所示：

![图3 一种适用于垂直领域的新词自动挖掘方法](https://p0.meituan.net/travelcube/1b31b67fe3b8a413811170d7e5b26ac6259130.png@2502w_578h_80q)


- **Step1：候选序列挖掘**。频繁连续出现的词序列，是潜在新型词汇的有效候选，我们采用频繁序列产生充足候选集合。

- **Step2：基于远程监督的大规模有标记语料生成**。
  - 频繁序列随着给定语料的变化而改变，因此人工标记成本极高。
  - 利用领域已有累积的实体词典作为远程监督词库，将Step1中候选序列与实体词典的交集作为训练正例样本。
  - 同时，通过对候选序列分析发现，在上百万的频繁Ngram中仅约10%左右的候选是真正的高质新型词汇。
  - 因此，对于负例样本，采用负采样方式生产训练负例集[1]。
  - 针对海量UGC语料，我们设计并定义了四个维度的统计特征来衡量候选短语可用性：

    - **频率**：有意义的新词在语料中应当满足一定的频率，该指标由Step1计算得到。

    - **紧密度**：主要用于评估新短语中连续元素的共现强度，包括T分布检验、皮尔森卡方检验、逐点互信息、似然比等指标。

    - **信息度**：新发现词汇应具有真实意义，指代某个新的实体或概念，该特征主要考虑了词组在语料中的逆文档频率、词性分布以及停用词分布。

    - **完整性**：新发现词汇应当在给定的上下文环境中作为整体解释存在，因此应同时考虑词组的子集短语以及超集短语的紧密度，从而衡量词组的完整性。

  - 在经过小样本标记数据构建和多维度统计特征提取后，训练二元分类器来计算候选短语预估质量。由于训练数据负例样本采用了负采样的方式，这部分数据中混合了少量高质量的短语，为了减少负例噪声对短语预估质量分的影响，可以通过集成多个弱分类器的方式减少误差。对候选序列集合进行模型预测后，将得分超过一定阈值的集合作为正例池，较低分数的集合作为负例池。

**Step3: 基于深度语义网络的短语质量评估**。在有大量标记数据的情况下，深度网络模型可以自动有效地学习语料特征，并产出具有泛化能力的高效模型。BERT通过海量自然语言文本和深度模型学习文本语义表征，并经过简单微调在多个自然语言理解任务上刷新了记录，因此我们基于BERT训练短语质量打分器。为了更好地提升训练数据的质量，我们利用搜索日志数据对Step2中生成的大规模正负例池数据进行远程指导，将有大量搜索记录的词条作为有意义的关键词。我们将正例池与搜索日志重合的部分作为模型正样本，而将负例池减去搜索日志集合的部分作为模型负样本，进而提升训练数据的可靠性和多样性。此外，我们采用Bootstrapping方式，在初次得到短语质量分后，重新根据已有短语质量分以及远程语料搜索日志更新训练样本，迭代训练提升短语质量打分器效果，有效减少了伪正例和伪负例。

在UGC语料中抽取出大量新词或短语后，参考AutoNER[2]对新挖掘词语进行类型预测，从而扩充离线的实体库。

### 3.2 在线匹配

原始的在线NER词典匹配方法直接针对Query做双向最大匹配，从而获得成分识别候选集合，再基于词频（这里指实体搜索量）筛选输出最终结果。这种策略比较简陋，对词库准确度和覆盖度要求极高，所以存在以下几个问题：

* 当Query包含词库未覆盖实体时，基于字符的最大匹配算法易引起切分错误。例如，搜索词“海坨山谷”，词库仅能匹配到“海坨山”，因此出现“海坨山/谷”的错误切分。

* 粒度不可控。例如，搜索词“星巴克咖啡”的切分结果，取决于词库对“星巴克”、“咖啡”以及“星巴克咖啡”的覆盖。

* 节点权重定义不合理。例如，直接基于实体搜索量作为实体节点权重，当用户搜索“信阳菜馆”时，“信阳菜/馆”的得分大于“信阳/菜馆”。


为了解决以上问题，在进行实体字典匹配前引入了CRF分词模型，针对垂直领域美团搜索制定分词准则，人工标注训练语料并训练CRF分词模型。同时，针对模型分词错误问题，设计两阶段修复方式：

1.  结合模型分词Term和基于领域字典匹配Term，根据动态规划求解Term序列权重和的最优解。
2.  基于Pattern正则表达式的强修复规则。最后，输出基于实体库匹配的成分识别结果。

![图4 实体在线匹配](https://p0.meituan.net/travelcube/7bad42de690d7127b80364ee0bc356f5118403.png@1234w_702h_80q)

图4 实体在线匹配

#### 模型在线预测
--

对于长尾、未登录查询，我们使用模型进行在线识别。 NER模型的演进经历了如下图5所示的几个阶段，目前线上使用的主模型是BERT[3]以及BERT+LR级联模型，另外还有一些在探索中模型的离线效果也证实有效，后续我们会综合考虑性能和收益逐步进行上线。搜索中NER线上模型的构建主要面临三个问题：

1.  性能要求高：NER作为基础模块，模型预测需要在毫秒级时间内完成，而目前基于深度学习的模型都有计算量大、预测时间较长的问题。
2.  领域强相关：搜索中的实体类型与业务供给高度相关，只考虑通用语义很难保证模型识别的准确性。
3.  标注数据缺乏： NER标注任务相对较难，需给出实体边界切分、实体类型信息，标注过程费时费力，大规模标注数据难以获取。

针对性能要求高的问题，我们的线上模型在升级为BERT时进行了一系列的性能调优；针对NER领域相关问题，我们提出了融合搜索日志特征、实体词典信息的知识增强NER方法；针对训练数据难以获取的问题，我们提出一种弱监督的NER方法。下面我们详细介绍下这些技术点。

![图5 NER模型演进](https://p1.meituan.net/travelcube/e6792448af09baa57bac84a44521bf5996645.png@1564w_452h_80q)

图5 NER模型演进

### 4.1 BERT模型

BERT是谷歌于2018年10月公开的一种自然语言处理方法。该方法一经发布，就引起了学术界以及工业界的广泛关注。在效果方面，BERT刷新了11个NLP任务的当前最优效果，该方法也被评为2018年NLP的重大进展以及NAACL 2019的best paper[4,5]。BERT和早前OpenAI发布的GPT方法技术路线基本一致，只是在技术细节上存在略微差异。两个工作的主要贡献在于使用预训练+微调的思路来解决自然语言处理问题。以BERT为例，模型应用包括2个环节：

* 预训练（Pre-training），该环节在大量通用语料上学习网络参数，通用语料包括Wikipedia、Book Corpus，这些语料包含了大量的文本，能够提供丰富的语言相关现象。

* 微调（Fine-tuning），该环节使用“任务相关”的标注数据对网络参数进行微调，不需要再为目标任务设计Task-specific网络从头训练。


将BERT应用于实体识别线上预测时面临一个挑战，即预测速度慢。我们从模型蒸馏、预测加速两个方面进行了探索，分阶段上线了BERT蒸馏模型、BERT+Softmax、BERT+CRF模型。

**4.1.1 模型蒸馏**

我们尝试了对BERT模型进行剪裁和蒸馏两种方式，结果证明，剪裁对于NER这种复杂NLP任务精度损失严重，而模型蒸馏是可行的。模型蒸馏是用简单模型来逼近复杂模型的输出，目的是降低预测所需的计算量，同时保证预测效果。Hinton在2015年的论文中阐述了核心思想[6]，复杂模型一般称作Teacher Model，蒸馏后的简单模型一般称作Student Model。Hinton的蒸馏方法使用伪标注数据的概率分布来训练Student Model，而没有使用伪标注数据的标签来训练。作者的观点是概率分布相比标签能够提供更多信息以及更强约束，能够更好地保证Student Model与Teacher Model的预测效果达到一致。在2018年NeurIPS的Workshop上，[7]提出一种新的网络结构BlendCNN来逼近GPT的预测效果，本质上也是模型蒸馏。BlendCNN预测速度相对原始GPT提升了300倍，另外在特定任务上，预测准确率还略有提升。关于模型蒸馏，基本可以得到以下结论：

* **模型蒸馏本质是函数逼近**。针对具体任务，笔者认为只要Student Model的复杂度能够满足问题的复杂度，那么Student Model可以与Teacher Model完全不同，选择Student Model的示例如下图6所示。举个例子，假设问题中的样本（x，y）从多项式函数中抽样得到，最高指数次数d=2；可用的Teacher Model使用了更高指数次数（比如d=5），此时，要选择一个Student Model来进行预测，Student Model的模型复杂度不能低于问题本身的复杂度，即对应的指数次数至少达到d=2。

* **根据无标注数据的规模，蒸馏使用的约束可以不同**。如图7所示，如果无标注数据规模小，可以采用值（logits）近似进行学习，施加强约束；如果无标注数据规模中等，可以采用分布近似；如果无标注数据规模很大，可以采用标签近似进行学习，即只使用Teacher Model的预测标签来指导模型学习。


![](https://p0.meituan.net/travelcube/da814261ba3c4c12d2799162a5cef32355236.png@884w_385h_80q)

有了上面的结论，我们如何在搜索NER任务中应用模型蒸馏呢？ 首先先分析一下该任务。与文献中的相关任务相比，搜索NER存在有一个显著不同：作为线上应用，搜索有大量无标注数据。用户查询可以达到千万/天的量级，数据规模上远超一些离线测评能够提供的数据。据此，我们对蒸馏过程进行简化：不限制Student Model的形式，选择主流的推断速度快的神经网络模型对BERT进行近似；训练不使用值近似、分布近似作为学习目标，直接使用标签近似作为目标来指导Student Model的学习。

我们使用IDCNN-CRF来近似BERT实体识别模型，IDCNN（Iterated Dilated CNN）是一种多层CNN网络，其中低层卷积使用普通卷积操作，通过滑动窗口圈定的位置进行加权求和得到卷积结果，此时滑动窗口圈定的各个位置的距离间隔等于1。高层卷积使用膨胀卷积（Atrous Convolution）操作，滑动窗口圈定的各个位置的距离间隔等于d（d>1）。通过在高层使用膨胀卷积可以减少卷积计算量，同时在序列依赖计算上也不会有损失。在文本挖掘中，IDCNN常用于对LSTM进行替换。实验结果表明，相较于原始BERT模型，在没有明显精度损失的前提下，蒸馏模型的在线预测速度有数十倍的提升。

**4.1.2 预测加速**

BERT中大量小算子以及Attention计算量的问题，使得其在实际线上应用时，预测时长较高。我们主要使用以下三种方法加速模型预测，同时对于搜索日志中的高频Query，我们将预测结果以词典方式上传到缓存，进一步减少模型在线预测的QPS压力。下面介绍下模型预测加速的三种方法：

**算子融合**：通过降低Kernel Launch次数和提高小算子访存效率来减少BERT中小算子的耗时开销。我们这里调研了Faster Transformer的实现。平均时延上，有1.4x~2x左右加速比；TP999上，有2.1x~3x左右的加速比。该方法适合标准的BERT模型。开源版本的Faster Transformer工程质量较低，易用性和稳定性上存在较多问题，无法直接应用，我们基于NV开源的Faster Transformer进行了二次开发，主要在稳定性和易用性进行了改进：

* 易用性：支持自动转换，支持Dynamic Batch，支持Auto Tuning。
* 稳定性：修复内存泄漏和线程安全问题。

**Batching**：Batching的原理主要是将多次请求合并到一个Batch进行推理，降低Kernel Launch次数、充分利用多个GPU SM，从而提高整体吞吐。在max\_batch\_size设置为4的情况下，原生BERT模型，可以在将平均Latency控制在6ms以内，最高吞吐可达1300 QPS。该方法十分适合美团搜索场景下的BERT模型优化，原因是搜索有明显的高低峰期，可提升高峰期模型的吞吐量。

**混合精度**：混合精度指的是FP32和FP16混合的方式，使用混合精度可以加速BERT训练和预测过程并且减少显存开销，同时兼顾FP32的稳定性和FP16的速度。在模型计算过程中使用FP16加速计算过程，模型训练过程中权重会存储成FP32格式，参数更新时采用FP32类型。利用FP32 Master-weights在FP32数据类型下进行参数更新，可有效避免溢出。混合精度在基本不影响效果的基础上，模型训练和预测速度都有一定的提升。

### 4.2 知识增强的NER

如何将特定领域的外部知识作为辅助信息嵌入到语言模型中，一直是近些年的研究热点。K-BERT[8]、ERNIE[9]等模型探索了知识图谱与BERT的结合方法，为我们提供了很好的借鉴。美团搜索中的NER是领域相关的，实体类型的判定与业务供给高度相关。因此，我们也探索了如何将供给POI信息、用户点击、领域实体词库等外部知识融入到NER模型中。

**4.2.1 融合搜索日志特征的Lattice-LSTM**

在O2O垂直搜索领域，大量的实体由商家自定义（如商家名、团单名等），实体信息隐藏在供给POI的属性中，单使用传统的语义方式识别效果差。Lattice-LSTM[10]针对中文实体识别，通过增加词向量的输入，丰富语义信息。我们借鉴这个思路，结合搜索用户行为，挖掘Query 中潜在短语，这些短语蕴含了POI属性信息，然后将这些隐藏的信息嵌入到模型中，在一定程度上解决领域新词发现问题。与原始Lattice-LSTM方法对比，识别准确率千分位提升5个点。

![图8  融合搜索日志特征的Lattice-LSTM构建流程](https://p1.meituan.net/travelcube/f3a75d844a828bb2a2f7971d6b6757a2484477.png@2162w_962h_80q)

图8 融合搜索日志特征的Lattice-LSTM构建流程

**1) 短语挖掘及特征计算**

该过程主要包括两步：匹配位置计算、短语生成，下面详细展开介绍。

![图 9 短语挖掘及特征计算](https://p1.meituan.net/travelcube/2be31e701abd81d0f31a38190ae9cf04143597.png@1836w_402h_80q)

图 9 短语挖掘及特征计算

**Step1：匹配位置计算**。对搜索日志进行处理，重点计算查询与文档字段的详细匹配情况以及计算文档权重（比如点击率）。如图9所示，用户输入查询是“手工编织”，对于文档d1（搜索中就是POI），“手工”出现在字段“团单”，“编织”出现在字段“地址”。对于文档2，“手工编织”同时出现在“商家名”和“团单”。匹配开始位置、匹配结束位置分别对应有匹配的查询子串的开始位置以及结束位置。

**Step2：短语生成**。以Step1的结果作为输入，使用模型推断候选短语。可以使用多个模型，从而生成满足多个假设的结果。我们将候选短语生成建模为整数线性规划（Integer Linear Programmingm，ILP）问题，并且定义了一个优化框架，模型中的超参数可以根据业务需求进行定制计算，从而获得满足不用假设的结果。对于一个具体查询Q，每种切分结果都可以使用整数变量xij来表示：xij=1表示查询i到j的位置构成短语，即Qij是一个短语，xij=0表示查询i到j的位置不构成短语。优化目标可以形式化为：在给定不同切分xij的情况下，使收集到的匹配得分最大化。优化目标及约束函数如图10所示，其中p：文档，f：字段，w：文档p的权重，wf：字段f的权重。xijpf：查询子串Qij是否出现在文档p的f字段，且最终切分方案会考虑该观测证据，Score(xijpf)：最终切分方案考虑的观测得分，w(xij)：切分Qij对应的权重，yijpf : 观测到的匹配，查询子串Qij出现在文档p的f字段中。χmax：查询包含的最大短语数。这里，χmax、wp、wf 、w(xij)是超参数，在求解ILP问题前需要完成设置，这些变量可以根据不同假设进行设置：可以根据经验人工设置，另外也可以基于其他信号来设置，设置可参考图10给出的方法。最终短语的特征向量表征为在POI各属性字段的点击分布。

![图10 短语生成问题抽象以及参数设置方法](https://p0.meituan.net/travelcube/f702315775be32922da1ae5677c9b80774362.png@694w_338h_80q)

图10 短语生成问题抽象以及参数设置方法

**2) 模型结构**

![图11 融合搜索日志特征的Lattice-LSTM模型结构](https://p1.meituan.net/travelcube/2af525566623c686a919234ee6181f7853413.png@902w_508h_80q)

图11 融合搜索日志特征的Lattice-LSTM模型结构

模型结构如图11所示，蓝色部分表示一层标准的LSTM网络（可以单独训练，也可以与其他模型组合），输入为字向量，橙色部分表示当前查询中所有词向量，红色部分表示当前查询中的通过Step1计算得到的所有短语向量。对于LSTM的隐状态输入，主要由两个层面的特征组成：当前文本语义特征，包括当前字向量输入和前一时刻字向量隐层输出；潜在的实体知识特征，包括当前字的短语特征和词特征。下面介绍当前时刻潜在知识特征的计算以及特征组合的方法：（下列公式中，σ表示sigmoid函数，⊙表示矩阵乘法）

![](https://p0.meituan.net/travelcube/155850c8cffee9bf9e4183c8ee432459208959.png@1158w_674h_80q)

![](https://p1.meituan.net/travelcube/34277ea44d0d708147712872757b0662146873.png@1184w_324h_80q)

**4.2.2 融合实体词典的两阶段NER**

我们考虑将领域词典知识融合到模型中，提出了两阶段的NER识别方法。该方法是将NER任务拆分成实体边界识别和实体标签识别两个子任务。相较于传统的端到端的NER方法，这种方法的优势是实体切分可以跨领域复用。另外，在实体标签识别阶段可以充分使用已积累的实体数据和实体链接等技术提高标签识别准确率，缺点是会存在错误传播的问题。

在第一阶段，让BERT模型专注于实体边界的确定，而第二阶段将实体词典带来的信息增益融入到实体分类模型中。第二阶段的实体分类可以单独对每个实体进行预测，但这种做法会丢失实体上下文信息，我们的处理方法是：将实体词典用作训练数据训练一个IDCNN分类模型，该模型对第一阶段输出的切分结果进行编码，并将编码信息加入到第二阶段的标签识别模型中，联合上下文词汇完成解码。基于Benchmark标注数据进行评估，该模型相比于BERT-NER在Query粒度的准确率上获得了1%的提升。这里我们使用IDCNN主要是考虑到模型性能问题，大家可视使用场景替换成BERT或其他分类模型。

![图12 融合实体词典的两阶段NER](https://p0.meituan.net/travelcube/d5787da81fb5ef60b70eb03944d52066270011.png@2492w_1688h_80q)

图12 融合实体词典的两阶段NER

### 4.3 弱监督NER

![13 弱监督标注数据生成流程](https://p0.meituan.net/travelcube/a77f77a5f1f141eba3b3909bb7aa89c6185047.png@1968w_785h_80q)

13 弱监督标注数据生成流程

针对标注数据难获取问题，我们提出了一种弱监督方案，该方案包含两个流程，分别是弱监督标注数据生成、模型训练。下面详细描述下这两个流程。

**Step1：弱监督标注样本生成**

1) 初版模型：利用已标注的小批量数据集训练实体识别模型，这里使用的是最新的BERT模型，得到初版模型ModelA。

2) 词典数据预测：实体识别模块目前沉淀下百万量级的高质量实体数据作为词典，数据格式为实体文本、实体类型、属性信息。用上一步得到的ModelA预测改词典数据输出实体识别结果。

3) 预测结果校正：实体词典中实体精度较高，理论上来讲模型预测的结果给出的实体类型至少有一个应该是实体词典中给出的该实体类型，否则说明模型对于这类输入的识别效果并不好，需要针对性地补充样本，我们对这类输入的模型结果进行校正后得到标注文本。校正方法我们尝试了两种，分别是整体校正和部分校正，整体校正是指整个输入校正为词典实体类型，部分校正是指对模型切分出的单个Term 进行类型校正。举个例子来说明，“兄弟烧烤个性diy”词典中给出的实体类型为商家，模型预测结果为修饰词+菜品+品类，没有Term属于商家类型，模型预测结果和词典有差异，这时候我们需要对模型输出标签进行校正。校正候选就是三种，分别是“商家+菜品+品类”、“修饰词+商家+品类”、“修饰词+菜品+商家”。我们选择最接近于模型预测的一种，这样选择的理论意义在于模型已经收敛到预测分布最接近于真实分布，我们只需要在预测分布上进行微调，而不是大幅度改变这个分布。那从校正候选中如何选出最接近于模型预测的一种呢？我们使用的方法是计算校正候选在该模型下的概率得分，然后与模型当前预测结果（当前模型认为的最优结果）计算概率比，概率比计算公式如公式2所示，概率比最大的那个就是最终得到的校正候选，也就是最终得到的弱监督标注样本。在“兄弟烧烤个性diy”这个例子中，“商家+菜品+品类”这个校正候选与模型输出的“修饰词+菜品+品类”概率比最大，将得到“兄弟/商家 烧烤/菜品 个性diy/品类”标注数据。

![图 14 标签校正](https://p0.meituan.net/travelcube/bf48535cfa1b1fef9c90edb458c1f14c100117.png@1722w_328h_80q)

图 14 标签校正

![公式 2 概率比计算](https://p0.meituan.net/travelcube/78d86cc3a875de85786ae4cacc7dba6029223.png@1296w_136h_80q)

公式 2 概率比计算

**Step2：弱监督模型训练**

弱监督模型训练方法包括两种：一是将生成的弱监督样本和标注样本进行混合不区分重新进行模型训练；二是在标注样本训练生成的ModelA基础上，用弱监督样本进行Fine-tuning训练。这两种方式我们都进行了尝试。从实验结果来看，Fine-tuning效果更好。

#### 总结和展望
-

本文介绍了O2O搜索场景下NER任务的特点及技术选型，详述了在实体词典匹配和模型构建方面的探索与实践。

实体词典匹配针对线上头腰部流量，离线对POI结构化信息、商户评论数据、搜索日志等独有数据进行挖掘，可以很好的解决领域实体识别问题，在这一部分我们介绍了一种适用于垂直领域的新词自动挖掘方法。除此之外，我们也积累了其他可处理多源数据的挖掘技术，如有需要可以进行约线下进行技术交流。

模型方面，我们围绕搜索中NER模型的构建的三个核心问题（性能要求高、领域强相关、标注数据缺乏）进行了探索。针对性能要求高采用了模型蒸馏，预测加速的方法， 使得NER 线上主模型顺利升级为效果更好的BERT。在解决领域相关问题上，分别提出了融合搜索日志、实体词典领域知识的方法，实验结果表明这两种方法可一定程度提升预测准确率。针对标注数据难获取问题，我们提出了一种弱监督方案，一定程度缓解了标注数据少模型预测效果差的问题。

未来，我们会在解决NER未登录识别、歧义多义、领域相关问题上继续深入研究，欢迎业界同行一起交流。

#### 参考资料


[1] Automated Phrase Mining from Massive Text Corpora. 2018.

[2] Learning Named Entity Tagger using Domain-Specific Dictionary. 2018.

[3] Bidirectional Encoder Representations from Transformers. 2018

[4] [https://www.jiqizhixin.com/articles/2018-12-30](https://www.jiqizhixin.com/articles/2018-12-30)

[5] [https://naacl2019.org/blog/best-papers/](https://naacl2019.org/blog/best-papers/)

[6] Hinton et al. Distilling the Knowledge in a Neural Network. 2015.

[7] Yew Ken Chia et al.Transformer to CNN: Label-scarce distillation for efficient text classification. 2018.

[8] K-BERT: Enabling Language Representation with Knowledge Graph. 2019.

[9] Enhanced Language Representation with Informative Entities. 2019.

[10] Chinese NER Using Lattice LSTM. 2018.

#### 作者简介


丽红，星池，燕华，马璐，廖群，志安，刘亮，李超，张弓，云森，永超等，均来自美团搜索与NLP部。

招聘信息
----

美团搜索部，长期招聘搜索、推荐、NLP算法工程师，坐标北京。欢迎感兴趣的同学发送简历至：tech@meituan.com（邮件标题注明：搜索与NLP部）

[算法](/tags/%E7%AE%97%E6%B3%95.html), [AI平台](/tags/ai%E5%B9%B3%E5%8F%B0.html), [NER](/tags/ner.html), [BERT](/tags/bert.html), [NLP](/tags/nlp.html), [AI](/tags/ai.html)

#看看其他

[前一篇: 智能搜索模型预估框架Augur的建设与实践](https://tech.meituan.com/2020/07/16/augur-in-meituan-nlp.html "智能搜索模型预估框架Augur的建设与实践") [后一篇: 新一代垃圾回收器ZGC的探索与实践](https://tech.meituan.com/2020/08/06/new-zgc-practice-in-meituan.html "新一代垃圾回收器ZGC的探索与实践")

#一起聊聊

如发现文章有错误、对内容有疑问，都可以关注美团技术团队微信公众号（meituantech），在后台给我们留言。

![美团技术团队微信二维码](https://p1.meituan.net/travelcube/b0364d579285ab22aa6235bd100d7c22178175.png)

分享一线技术实践，沉淀成长学习经验

$CONFIG['data']['footerLink']=[{"name":"网站首页","link":"/"},{"name":"文章存档","link":"/archives"},{"name":"关于我们","link":"/about"}];

一行代码，亿万生活。

* [网站首页](/)
* [文章存档](/archives)
* [关于我们](/about)

![](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABgAAAAYCAYAAADgdz34AAAABGdBTUEAALGPC/xhBQAAACBjSFJNAAB6JgAAgIQAAPoAAACA6AAAdTAAAOpgAAA6mAAAF3CculE8AAAABmJLR0QAAAAAAAD5Q7t/AAAACXBIWXMAAAsSAAALEgHS3X78AAAAjklEQVRIx81V2xHAIAjTnruyhmuwbb96Z6kYOJWaTx6SKGBm5poMIKJXnDXvsgTNoGgMrUxR3nYFnwLMXHusNTvyx73BA+1ONTvyb1eQvXNgjQ9T4EbbLaizQhQsKTBSsVyBvLbi7YoWaDZS6gzaDPNeUfM2lXOgHSjz4ncRYub92c7Zpgjnb1ON8e9vcAOBkF++GF/4vQAAAABJRU5ErkJggg==)

![](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAFAAAABQCAYAAACOEfKtAAAABGdBTUEAALGPC/xhBQAAACBjSFJNAAB6JgAAgIQAAPoAAACA6AAAdTAAAOpgAAA6mAAAF3CculE8AAAABmJLR0QAAAAAAAD5Q7t/AAAACXBIWXMAAAsSAAALEgHS3X78AAAB/ElEQVR42u3bX0rcUBiG8cc/o+5CwW5CL0rpJirY+2zDbZzbti6klXYhCm7B0Ssv7AfqJAMz7zk5n/A+VzMhTL78IOSQITullCvc1u32HuCjZ0AxA4oZUMyAYgYUM6CYAcUMKGZAMQOKGVDMgGIGFDOgmAHFDChmQDEDihlQzIBiBhQzoJgBxQwoZkAxA4rt9x5gTQ/Ab2AP+Awc9h5orKyAD8DPYRjuAUopt8AlcNR7sPdlvITf4AEMw3AHXAPL3sO9LxvgCl6UFTET4ApeKeVTKeUkvmdEzAI4igdcAJeZETMALpnG2wcWJEbsDbgEfozgfePtCiEtYk/AdXiLkf1TIvYC3BQvWof42ONEegBuixdNIf6iA+LcgCpelAZxTsAxvFM2x4tSIM4FOIV3wXZ4USAex4a5EecAbIUXLYDvvRBbA7bGi7ohtgScCy/qgtgKcG68aAqx2TqxBWAvvGgM8ZZGiLUBe+NFY3fnJog1AbPgRQdMIz7VOkhNwD+J8KIpxJtaB2hyE0mCF60g1myn4iv/j8BfXp7jnZED73VPwL//n895gZWr+bfmIfB1ZpRNOgC+1P7R3k+kP3wGFDOgmAHFDChmQDEDihlQzIBiBhQzoJgBxQwoZkAxA4oZUMyAYgYUM6CYAcUMKGZAMQOKGVDMgGIGFDOg2DOEU/uBJ0Ro/gAAAABJRU5ErkJggg==)

扫码关注技术博客

![](https://p1.meituan.net/travelcube/7d0f734bcd029f452d415ce7d521a0d9632811.gif)

try{window.dataLayer=window.dataLayer||[];function gtag(){dataLayer.push(arguments);} gtag('js',new Date());gtag('config','UA-55279261-1');}catch(e){}try{var \_hmt=\_hmt||[];var hm=document.createElement("script");hm.src="https://hm.baidu.com/hm.js?7158c55a533ed0cf57dede022b1e6aed";var s=document.getElementsByTagName("script")[0];s.parentNode.insertBefore(hm,s);}catch(e){}






.
