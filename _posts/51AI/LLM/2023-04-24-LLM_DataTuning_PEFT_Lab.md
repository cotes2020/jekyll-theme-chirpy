---
title: LLM - Data Tuning - PEFT Lab
date: 2023-04-24 11:11:11 -0400
description:
categories: [51AI, LLM]
# img: /assets/img/sample/rabbit.png
tags: [AI, ML]
---


# PEFT Lab

- [PEFT Lab](#peft-lab)
  - [ChatGLM-6B 微调实践](#chatglm-6b-微调实践)
  - [ChatGLM-6B + P-Tuning v2](#chatglm-6b--p-tuning-v2)
    - [模型下载](#模型下载)
    - [试用原始模型](#试用原始模型)
    - [量化细节](#量化细节)
    - [模型推理](#模型推理)
    - [灾难性遗忘问题](#灾难性遗忘问题)
  - [使用 ChatGLM2-6B 复用 ChatGLM-6B 进行 P-Tuning v2 流程需要注意的点。](#使用-chatglm2-6b-复用-chatglm-6b-进行-p-tuning-v2-流程需要注意的点)
    - [训练启动方式](#训练启动方式)
    - [模型推理](#模型推理-1)
  - [ChatGLM-6B + LoRA](#chatglm-6b--lora)
    - [LoRA 配置参数](#lora-配置参数)
    - [训练启动方式](#训练启动方式-1)
  - [ChatGLM2-6B + LoRA](#chatglm2-6b--lora)
    - [训练启动方式](#训练启动方式-2)
  - [ChatGLM-6B + LoRA + Accelerate + Deepspeed](#chatglm-6b--lora--accelerate--deepspeed)
    - [Docker 容器构建](#docker-容器构建)
    - [Python 环境构建](#python-环境构建)
    - [训练启动方式](#训练启动方式-3)

---

## ChatGLM-6B 微调实践

**实验环境**: 2 张 A30 卡(单卡显存 24G)，CentOS7。

**显存占用**:

| 模型方案                   | 训练方案       | 显存占用      |
| -------------------------- | -------------- | ------------- |
| ChatGLM-6B+P-Tuning v2     | 单卡训练       | 8G 左右       |
| ChatGLM2-6B+P-Tuning v2    | 单卡训练       | 8G 左右       |
| ChatGLM-6B+LoRA            | 两卡 DDP       | 单卡 13G 左右 |
| ChatGLM2-6B+LoRA           | 两卡 DDP       | 单卡 13G 左右 |
| ChatGLM-6B+LoRA+int8 量化  | 两卡流水线并行 | 两卡 13G 左右 |
| ChatGLM2-6B+LoRA+int8 量化 | 两卡流水线并行 | 两卡 27G 左右 |
| ChatGLM-6B+LoRA            | 两卡 Deepspeed | 单卡 11G 左右 |

---

## ChatGLM-6B + P-Tuning v2
- 官方任务实践: [【官方教程】ChatGLM-6B 微调](https://www.bilibili.com/video/BV1fd4y1Z7Y5/?spm_id_from=333.999.0.0&vd_source=25d0b87065d3da39fe110c6e0b4906e1)

### 模型下载

下载[ChatGLM-6B](https://www.huggingface.co/THUDM/chatglm-6b/tree/main)模型的方法很多，这里介绍官方给出的最快下载方式。

- **下载模型实现**: 由于下载整体模型较慢，所以我们先下载模型实现，再手动下载模型参数文件。

- **下载模型实现**
  - 需先[安装 Git LFS](https://docs.github.com/zh/repositories/working-with-files/managing-large-files/installing-git-large-file-storage?platform=mac)
  - `GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/THUDM/chatglm-6b`
  - 安装好之后再下载模型实现。

- **手动下载模型参数文件**:

  - **脚本方式(推荐)**:

    ```bash
    git clone git@github.com:chenyifanthu/THU-Cloud-Downloader.git
    cd THU-Cloud-Downloader
    pip install argparse requests tqdm
    python main.py --link https://cloud.tsinghua.edu.cn/d/fb9f16d6dc8f482596c2/ --save .chatglm-6b
    ```

  - **直接下载**: 从[ChatGLM-6B](https://cloud.tsinghua.edu.cn/d/fb9f16d6dc8f482596c2/)中将所有文件下载下来，替换模型实现步骤下载的文件夹`./chatglm-6b`中的文件。

  - **百度网盘下载**: 为了防止官方微调模型，导致模型与训练代码不适配，在百度网盘保存了一份模型参数文件，优先级较低，大家按需提取。链接: [ChatGLM-6B](https://pan.baidu.com/s/1A5zVKtQYfML0omsMYPnWfg)，提取码: 0314。

- **下载训练代码**:
  - [ChatGLM-6B](https://github.com/THUDM/ChatGLM-6B)。

  - `git clone git@github.com:THUDM/ChatGLM-6B.git`

  - 同上文模型下载一致，官网代码存在更新的可能，若想顺利运行本项目，可从百度网盘下载代码。链接: [ChatGLM-6B](https://pan.baidu.com/s/1bZWPdaayh2-FotCJdigqQw)， 提取码: 0314。

### 试用原始模型

- **安装包**:

  ```bash
  pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

  # 具体安装包
  protobuf
  transformers==4.27.1
  cpm_kernels
  torch>=1.10
  gradio
  mdtex2html
  sentencepiece
  accelerate
  ```

- **模型试用**:
  - 进行简单试用的启动命令，不使用量化，单卡显存 13G 左右，使用 8bit 量化，单卡显存 8G 左右。
  - `CUDA_VISIBLE_DEVICES=1 python cli_demo.py`

- **注意**:
  - **模型路径**: 因为前文中，我们已经下载了 chatglm-6B 模型，因此使用原始模型进行试用时，需要修改模型下载路径，即将`cli_demo.py`和`web_demo.py`中的`tokenizer`和`model`加载路径，`THUDM/chatglm-6b`修改为本地路径。后面包括训练在内的所有过程，都要注意这一点，就不重复赘述。![pic](https://img-blog.csdnimg.cn/cc620f27024341b8bd1690eb5dda2fdd.png#pic_center)


### 量化细节

- 量化的处理方式也进行了标记。
- 量化操作一般用于推理，加快推理速度，训练过程一般不采用此操作。
- 同时，量化操作是作用于部分参数，将这部分参数转换为 8 位整数表示，同时将`requires_grad`属性置为`False`。

- **训练前安装包**: `pip install rouge_chinese nltk jieba datasets`

- **数据集下载**:
  - [Tsinghua Cloud](https://cloud.tsinghua.edu.cn/f/b3f119a008264b1cabd1/?dl=1)。下载至目录`./ptuning`，ADGEN 数据集任务为根据输入(content)生成一段广告词(summary)。

```json
{
  "content": "类型#上衣*版型#宽松*版型#显瘦*图案#线条*衣样式#衬衫*衣袖型#泡泡袖*衣款式#抽绳",
  "summary": "这件衬衫的款式非常的宽松，利落的线条可以很好的隐藏身材上的小缺点，穿在身上有着很好的显瘦效果。领口装饰了一个可爱的抽绳，漂亮的绳结展现出了十足的个性，配合时尚的泡泡袖型，尽显女性甜美可爱的气息。"
}
```
- **启动训练**:

  ```bash
  cd ./ptuning
  sh train.sh
  ```

- **注意**: 训练过程中可能会出现错误[init_process_group error](https://github.com/THUDM/ChatGLM-6B/issues/1169)，可按照[fix pturning init_process_group error](https://github.com/THUDM/ChatGLM-6B/pull/1173/files)进行解决。


### 模型推理

```py
#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
################################################################################
#
# Copyright (c) 2023 Baidu.com, Inc. All Rights Reserved
#
################################################################################
"""
File    :   predict.py
brief   :   brief
Date    :   2023/07/03 08:00:52
Author  :   zhangce06
Contact :   zhangce06@baidu.com
"""

from transformers import AutoConfig, AutoModel, AutoTokenizer
import torch
import os
import platform
import signal
import readline

# pre_seq_len = 128

# 载入Tokenizer
tokenizer = AutoTokenizer.from_pretrained("../../chatglm-6b-model", trust_remote_code=True)
config = AutoConfig.from_pretrained("../../chatglm-6b-model", trust_remote_code=True, pre_seq_len=128)
# config.pre_seq_len = pre_seq_len
model = AutoModel.from_pretrained("../../chatglm-6b-model", config=config, trust_remote_code=True)

CHECKPOINT_PATH = "output/adgen-chatglm-6b-pt-128-2e-2/checkpoint-3000"
prefix_state_dict = torch.load(os.path.join(CHECKPOINT_PATH, "pytorch_model.bin"))
new_prefix_state_dict = {}
for k, v in prefix_state_dict.items():
    if k.startswith("transformer.prefix_encoder."):
        new_prefix_state_dict[k[len("transformer.prefix_encoder."):]] = v
model.transformer.prefix_encoder.load_state_dict(new_prefix_state_dict)

# 之后根据需求可以进行量化
# Comment out the following line if you don't use quantization
model = model.quantize(4)
model = model.half().cuda()
model.transformer.prefix_encoder.float()
model = model.eval()

os_name = platform.system()
clear_command = 'cls' if os_name == 'Windows' else 'clear'
stop_stream = False

def build_prompt(history):
    prompt = "欢迎使用 ChatGLM-6B 模型，输入内容即可进行对话，clear 清空对话历史，stop 终止程序"
    for query, response in history:
        prompt += f"\n\n用户: {query}"
        prompt += f"\n\nChatGLM-6B: {response}"
    return prompt

def signal_handler(signal, frame):
    global stop_stream
    stop_stream = True

def main():
    history = []
    global stop_stream
    print("欢迎使用 ChatGLM-6B 模型，输入内容即可进行对话，clear 清空对话历史，stop 终止程序")
    while True:
        query = input("\n用户: ")
        if query.strip() == "stop":
            break
        if query.strip() == "clear":
            history = []
            os.system(clear_command)
            print("欢迎使用 ChatGLM-6B 模型，输入内容即可进行对话，clear 清空对话历史，stop 终止程序")
            continue
        count = 0
        for response, history in model.stream_chat(tokenizer, query, history=history):
            if stop_stream:
                stop_stream = False
                break
            else:
                count += 1
                if count % 8 == 0:
                    os.system(clear_command)
                    print(build_prompt(history), flush=True)
                    signal.signal(signal.SIGINT, signal_handler)
        os.system(clear_command)
        print(build_prompt(history), flush=True)

if __name__ == "__main__":
    main()
```

### 灾难性遗忘问题

- 在该数据集上进行微调后，会出现灾难性遗忘的情况，在数据集有限的情况下，目前通过实践总结出下面三种做法，可在一定程度上缓解灾难性遗忘

- **学习率调整**: 通过调整学习率进行解决的[灾难性遗忘问题](https://github.com/THUDM/ChatGLM-6B/issues/1148)；
- **采用 LoRA 方法**: 参见「**ChatGLM-6B + LoRA ⇒ 真实任务实践**」；
- **采用 ChatGLM2-6B**: ChatGLM2-6B 确实比 ChatGLM-6B 强。使用相同的超参数进行微调训练，ChatGLM2-6B 在上述的广告数据集上微调后，确实没有出现灾难性遗忘的问题。不过仍然存在其他问题，大家自行体验。

---

## 使用 ChatGLM2-6B 复用 ChatGLM-6B 进行 P-Tuning v2 流程需要注意的点。

- **模型下载**:
  - 模型下载方式同 ChatGLM-6B 相同
  - 先下载模型实现[ChatGLM2-6B](https://huggingface.co/THUDM/chatglm2-6b/tree/main)，
  - 再下载模型参数文件[ChatGLM2-6B](https://cloud.tsinghua.edu.cn/d/674208019e314311ab5c/?p=/chatglm2-6b&mode=list)
  - 注意这里博主是直接手动下载的，脚本下载方式没有尝试成功，大家可以试一试。

  - **百度网盘下载**: 同样在百度网盘保存了一份模型参数文件，优先级较低，大家按需提取。链接: [ChatGLM2-6B](https://pan.baidu.com/s/1VsVY1di492WSRt1GsY8uGg)，提取码: 0625。

- **下载训练代码**:
  - ChatGLM2-6B 官方没有微调代码，因此微调代码博主还是采用的 ChatGLM-6B 的代码[ChatGLM-6B](https://github.com/THUDM/ChatGLM-6B)，下载方式不变。
  - 如果只是试用 ChatGLM2-6B，则可以下载 ChatGLM2-6B 的官方代码[ChatGLM2-6B](https://github.com/THUDM/ChatGLM2-6B)(百度网盘下载方式，链接: [ChatGLM2-6B](https://pan.baidu.com/s/1OemV9rXON92HybmMWm_AeA)，提取码: 0625)，试用方式也同 ChatGLM-6B 一致。不论是微调还是试用，记得更换模型文件路径。

- **试用细节**: ChatGLM-6B 试用时，可以使用半精度 FP16 加载模型，命令是`model.half()`，ChatGLM2-6B 则不用，因为其本身就是半精度状态。可通过如下命令查看模型参数的精度构成，可以发现，未使用 FP16 加载模型前，ChatGLM-6B 的模型参数精度是 FP16 和 FP32 混合的，ChatGLM2-6B 则只有 FP16 精度的参数。

```py
      model = AutoModel.from_pretrained("../../chatglm-6b-model", trust_remote_code=True)
      for name, param in model.named_parameters():
        if param.requires_grad == True:
            print(f"{name},------------,{param.dtype}")
```

- **安装包**:
  - ChatGLM2-6B 需要适配更高版本的 transformers 和 pytorch，才能发挥推理性能的优势。
  - 因此，试用 ChatGLM2-6B 时，安装包如下:
    ```bash
    # 具体安装包
    protobuf
    transformers==4.30.2
    cpm_kernels
    torch>=2.0
    gradio
    mdtex2html
    sentencepiece
    accelerate
    ```

  - 如果需要微调 ChatGLM2-6B，则同 ChatGLM-6B 一致，安装如下 python 包:
    ```
    pip install rouge_chinese nltk jieba datasets
    ```

  - **数据集下载**: 无变化，同 ChatGLM-6B 一致。

### 训练启动方式

基本无变化，大体流程同 ChatGLM-6B 一致。有两个地方需要注意，一个是脚本`./ptuning/train.sh`中的各种文件路径按需调整；另一个是`./ptuning/main.py`文件`line 220`左右进行如下修改:

```py
    # 适配ChatGLM1
    # context_length = input_ids.index(tokenizer.bos_token_id)
    # mask_position = context_length - 1
    # labels = [-100] * context_length + input_ids[mask_position+1:]

    # 适配ChatGLM2
    context_length = len(input_ids) - len(b_ids)
    mask_position = context_length
    labels = [-100] * context_length + input_ids[mask_position:]```
```

### 模型推理

基本无变化，同样注意修改模型文件路径。

---

## ChatGLM-6B + LoRA

官方任务实践
- 参考代码[ChatGLM_Tuning](https://github.com/zejunwang1/chatglm_tuning/blob/main/README.md)，实现了 ChatGLM-6B 基于 LoRA 的微调流程。
- 具体代码见[LLM 微调实践](https://github.com/DankoZhang/LLM/blob/main/README.md)。
- 模型文件同样可根据前文的方法进行获取，其中官方的模型可能存在更新，如果想顺利复现训练过程，建议从网盘进行下载。

### LoRA 配置参数

```yaml
r: lora矩阵的秩，矩阵A和矩阵B相连接的宽度，r<<d，以 int 表示。较低的秩会导致较小的更新矩阵和较少的可训练参数
target_modules: 模型中使用LoRA更新矩阵的模块，模型中常见的是，更新注意力模块
lora_alpha : LoRA缩放因子
bias : 指定是否应训练bias 参数。"none": 均不可；"all": 均可；"lora_only": 只有lora部分的bias可训练
lora_dropout: lora层的dropout比率
task_type: 模型任务类型，例如CAUSAL_LM任务
```

- **注意**:
  - **参数更新**: 模型经过 LoRA 配置加载后，可更新模型参数只有 LoRA 部分，且参数精度被重置为 FP32；
  - **量化方式**: `load_in_8bit=True`和`quantize(8)`区别，LoRA 微调时只能用前者，由 bitsandbytes 库提供；P-Tuning v2 可以采用后者，参考[量化方式区别](https://github.com/hiyouga/ChatGLM-Efficient-Tuning/issues/69)。

### 训练启动方式

- **数据并行**:

```bash
# 切换路径
cd chatglm-ft-lora/

# 启动训练
CUDA_VISIBLE_DEVICES=1,2 torchrun --nproc_per_node=2 train.py \
  --train_args_file ./conf/chatglm2_6b_lora.json \
  --model_name_or_path ../../chatglm2-6b-model/ \
  --data_path ./data/AdvertiseGen/train.jsonl \
  --max_input_length 128 \
  --max_output_length 256
```

- **模型(流水线)并行**:
```bash
# 切换路径
cd ./chatglm-ft-lora/

# 启动训练
CUDA_VISIBLE_DEVICES=1,2 python train.py \
  --train_args_file ./conf/chatglm_6b_lora.json \
  --model_name_or_path ../../chatglm-6b-model/ \
  --data_path ./data/AdvertiseGen/train.jsonl \
  --max_input_length 128 \
  --max_output_length 256 \
  --int8
```

- **注意**: 进行模型并行训练时，需要注意一个问题，即安装包问题。
  - **安装包问题**: 采用模型并行时，还需安装`accelerate` `bitsandbytes` `scipy` `tensorboardX`四个安装包。


---

## ChatGLM2-6B + LoRA

官方任务实践

- 实现了 ChatGLM2-6B 基于 LoRA 的微调流程。
- 具体代码见[LLM 微调实践](https://github.com/DankoZhang/LLM/blob/main/README.md)。模型文件同样可根据前文的方法进行获取，其中官方的模型可能存在更新，如果想顺利复现训练过程，建议从网盘进行下载。

- **LoRA 配置参数**: 同 ChatGLM-6B；


### 训练启动方式

- **数据并行**:

```bash
# 切换路径
cd ./chatglm2-ft-lora/

# 启动训练
CUDA_VISIBLE_DEVICES=1,2 torchrun --nproc_per_node=2 train.py --train_args_file ./conf/chatglm2_6b_lora.json --model_name_or_path ../../chatglm2-6b-model/ --data_path ./data/AdvertiseGen/train.jsonl --max_input_length 128 --max_output_length 256
```

- **注意**: 使用 ChatGLM2-6B 进行数据并行训练时，需要注意一个问题，即并行问题。

  - **并行问题**: 实际运行时，如果报错如下，说明显存不够了，我当时因为另一张卡并非完全空余，就修改了并行策略，只采用了单卡训练。

    ```bash
    # 错误内容
    RuntimeError: CUDA error: CUBLAS_STATUS_NOT_INITIALIZED when calling cublasCreate(handle)

    # 单卡训练
    CUDA_VISIBLE_DEVICES=1 torchrun --nproc_per_node=1 train.py --train_args_file ./conf/chatglm2_6b_lora.json --model_name_or_path ../../chatglm2-6b-model/ --data_path ./data/AdvertiseGen/train.jsonl --max_input_length 128 --max_output_length 256
    ```

- **模型(流水线)并行**:

```bash
# 切换路径
cd chatglm2-ft-lora/

# 启动训练
CUDA_VISIBLE_DEVICES=1,2 python train.py --train_args_file ./conf/chatglm2_6b_lora.json --model_name_or_path ../../chatglm2-6b-model/ --data_path ./data/AdvertiseGen/train.jsonl --max_input_length 128 --max_output_length 256 --int8
```

- **注意**: 进行模型并行训练时，需要注意两个问题，即安装包问题 模型源码修改问题。

  - **安装包问题**: 采用模型并行时，还需安装`accelerate` `bitsandbytes` `scipy` `tensorboardX`四个安装包；
  - **模型源码修改问题**: 采用模型并行训练时，如果报错如下`found at least two devices, cuda:1 and cuda:0!`，是模型源码问题。如果采用官方模型，可能这个 bug 已经被修复，但是如果采用的是百度网盘下载的模型，这个问题可能会出现，因此需要解决掉。解决办法可参考[bug 修复](https://github.com/yuanzhoulvpi2017/zero_nlp/issues/139)。具体来说，对`modeling_chatglm.py`文件的`955`行代码附近做如下修改(只修改一行，其余不变):

```py
    # 原代码
    loss = None
    if labels is not None:
        lm_logits = lm_logits.to(torch.float32)

        # Shift so that tokens < n predict n
        shift_logits = lm_logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous() #<<<------------------看这里
        # Flatten the tokens
        loss_fct = CrossEntropyLoss(ignore_index=-100)
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        lm_logits = lm_logits.to(hidden_states.dtype)
        loss = loss.to(hidden_states.dtype)

    if not return_dict:
        output = (lm_logits,) + transformer_outputs[1:]
        return ((loss,) + output) if loss is not None else output

    return CausalLMOutputWithPast(
        loss=loss,
        logits=lm_logits,
        past_key_values=transformer_outputs.past_key_values,
        hidden_states=transformer_outputs.hidden_states,
        attentions=transformer_outputs.attentions,
    )

    # 修改为
    loss = None
    if labels is not None:
        lm_logits = lm_logits.to(torch.float32)

        # Shift so that tokens < n predict n
        shift_logits = lm_logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous().to(shift_logits.device) #<<<--------------------看这里
        # Flatten the tokens
        loss_fct = CrossEntropyLoss(ignore_index=-100)
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        lm_logits = lm_logits.to(hidden_states.dtype)
        loss = loss.to(hidden_states.dtype)

    if not return_dict:
        output = (lm_logits,) + transformer_outputs[1:]
        return ((loss,) + output) if loss is not None else output

    return CausalLMOutputWithPast(
        loss=loss,
        logits=lm_logits,
        past_key_values=transformer_outputs.past_key_values,
        hidden_states=transformer_outputs.hidden_states,
        attentions=transformer_outputs.attentions,
    )
```


## ChatGLM-6B + LoRA + Accelerate + Deepspeed

官方任务实践
- 参考了代码[LLM-tuning](https://github.com/jiangxinyang227/LLM-tuning/blob/master/README.md)，实现了该流程，具体代码见[LLM 微调实践](https://github.com/DankoZhang/LLM/blob/main/README.md)。
- ChatGLM2-6B 可参考前文代码，对 tokensize 改写，进行适配训练即可。
- 由于 Deepspeed 框架对环境依赖性很高，因此我们采用 docker 技术，构建**cuda11.7**+**torch2.0.0**+**python3.10**虚拟环境。
- Docker 构建的具体方法参考[Docker 基础知识](https://blog.csdn.net/qq_39439006/article/details/131906881?csdn_share_tail=%7B%22type%22:%22blog%22,%22rType%22:%22article%22,%22rId%22:%22131906881%22,%22source%22:%22qq_39439006%22%7D)，此处简要介绍整体流程。

### Docker 容器构建

```dockerfile
# 运行容器
docker run -itd -v 宿主机路径:容器路径 --shm-size=8gb --rm --runtime=nvidia --gpus all --network host --name GPU-Docker nvidia/cuda:11.7.1-devel-ubi8 /bin/bash

# 进入容器
docker exec -it GPU-Docker /bin/bash

# 注
--shm-size=8gb必须加上，不然运行代码会报存储错误
```

### Python 环境构建

- **Python 安装**: 自行下载 Python3.10 版本的[Miniconda](https://docs.conda.io/en/latest/miniconda.html#installing) ;

- **注**: 记得在容器内设定 Python 环境变量
  ```bash
  vi ~/.bashrc
  export PATH=/home/LLM/ChatGLM-FT/miniconda3/bin:$PATH
  source ~/.bashrc
  ```

- **虚拟环境构建**: 参考[Python 基础知识](https://blog.csdn.net/qq_39439006/article/details/131925283?csdn_share_tail=%7B%22type%22:%22blog%22,%22rType%22:%22article%22,%22rId%22:%22131925283%22,%22source%22:%22qq_39439006%22%7D)；

- **依赖包安装**: 以下所有安装包的版本都是推荐，可按实际情况自行调整。

```bash
# torch安装
pip install torch==2.0.0+cu117 torchvision==0.15.1+cu117 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cu117

# 其他模块安装
pip install transformers==4.31.0
pip install datasets==2.14.0
pip install peft==0.4.0
pip install accelerate==0.21.0
pip install deepspeed==0.10.0
pip install sentencepiece==0.1.99
```

---

### 训练启动方式

```bash
# 切换路径
cd ./chatglm-ft-lora-dp/

# 启动训练
accelerate launch --config_file ./conf/accelerate_config.yaml
```

- **模型加载说明**:

  - `empty_init=False`: 目前如果使用 Deepspeed 进行训练，在加载 ChatGLM 模型时，参数`empty_init`必须置为 False(参考[empty_init 问题](https://github.com/THUDM/ChatGLM-6B/issues/530))，后续官方可能会更新源码，修复该问题；
  - `trust_remote_code=True`: 加载模型代码时，加上此参数，防止报错；
  - `torch_dtype=torch.float16`，FP16 加载模型；
  - `args.base_model`: 模型文件路径，最后一定是以`/`结尾，如`./chatglm-6b-model/`，`./chatglm-6b-model`会报错。
    ```py
    model = AutoModel.from_pretrained(
                args.base_model,
                empty_init=False,
                torch_dtype=torch.float16,
                trust_remote_code=True
            )
    ```

- **注意**: 模型训练过程中，如果出现如下错误: `ValueError: max() arg is an empty sequence`，需要对 deepspeed 源码进行修改。
  ```py
  # 源码路径
  ./miniconda3/envs/zhangce-dp/lib/python3.10/site-packages/deepspeed/runtime/zero/stage3.py

  # 原代码
  largest_partitioned_param_numel = max([
      max([max(tensor.numel(), tensor.ds_numel) for tensor in fp16_partitioned_group])
      for fp16_partitioned_group in self.fp16_partitioned_groups
  ])

  # 修改后代码
  largest_partitioned_param_numel = max([
      max([max(tensor.numel(), tensor.ds_numel) for tensor in fp16_partitioned_group])
      for fp16_partitioned_group in self.fp16_partitioned_groups if len (fp16_partitioned_group) > 0
  ])
  ```
