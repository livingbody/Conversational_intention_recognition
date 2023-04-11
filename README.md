# 一、基于PaddleNLP的对话意图识别

## 1.比赛介绍
意图识别是指分析用户的核心需求，输出与查询输入最相关的信息，例如在搜索中要找电影、查快递、市政办公等需求，这些需求在底层的检索策略会有很大的不同，错误的识别几乎可以确定找不到能满足用户需求的内容，导致产生非常差的用户体验；在对话过程中要准确理解对方所想表达的意思，这是具有很大挑战性的任务。

意图识别的准确性能在很大程度上影响着搜索的准确性和对话系统的智能性，在本赛题中我们需要选手对中文对话进行意图识别。

## 2.数据集介绍
- 训练数据：大约1.2万条中文对话
- 测试数据：3000条无标注对话

## 3.提交样式
评分使用准确率进行评分，准确率值越大越好。

- 实操方案不允许使用外部数据集，允许使用公开的外部预训练模型。
- 实操方案需要在指定平台进行评分，提交csv格式。

提交样例：
```
ID,Target
1,TVProgram-Play
2,HomeAppliance-Control
3,Audio-Play
4,Alarm-Update
5,HomeAppliance-Control
6,FilmTele-Play
7,FilmTele-Play
8,Music-Play
9,Calendar-Query
10,Video-Play
11,Alarm-Update
12,Music-Play
13,Travel-Query
14,TVProgram-Play
```

## 4.基本思路
![](https://ai-studio-static-online.cdn.bcebos.com/247187fe7fab49c2b602d464409e57b2ea2679a7c0584f34bdb4b53503d88797)



# 二、环境准备


```python
%cd ~
!git clone https://gitee.com/paddlepaddle/PaddleNLP/
```

    /home/aistudio
    正克隆到 'PaddleNLP'...
    remote: Enumerating objects: 47494, done.[K
    remote: Counting objects: 100% (34730/34730), done.[K
    remote: Compressing objects: 100% (17072/17072), done.[K
    remote: Total 47494 (delta 23983), reused 27016 (delta 16711), pack-reused 12764[K
    接收对象中: 100% (47494/47494), 87.84 MiB | 4.86 MiB/s, 完成.
    处理 delta 中: 100% (32328/32328), 完成.
    检查连接... 完成。



```python
!pip install -U paddlenlp
```

# 三、数据处理


```python
import pandas as pd
df=pd.read_csv('data/data208091/train.csv',sep='\t',header=None)
df.head(10)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>还有双鸭山到淮阴的汽车票吗13号的</td>
      <td>Travel-Query</td>
    </tr>
    <tr>
      <th>1</th>
      <td>从这里怎么回家</td>
      <td>Travel-Query</td>
    </tr>
    <tr>
      <th>2</th>
      <td>随便播放一首专辑阁楼里的佛里的歌</td>
      <td>Music-Play</td>
    </tr>
    <tr>
      <th>3</th>
      <td>给看一下墓王之王嘛</td>
      <td>FilmTele-Play</td>
    </tr>
    <tr>
      <th>4</th>
      <td>我想看挑战两把s686打突变团竞的游戏视频</td>
      <td>Video-Play</td>
    </tr>
    <tr>
      <th>5</th>
      <td>我想看和平精英上战神必备技巧的游戏视频</td>
      <td>Video-Play</td>
    </tr>
    <tr>
      <th>6</th>
      <td>2019年古装爱情电视剧小女花不弃的花絮播放一下</td>
      <td>Video-Play</td>
    </tr>
    <tr>
      <th>7</th>
      <td>找一个2004年的推理剧给我看一会呢</td>
      <td>FilmTele-Play</td>
    </tr>
    <tr>
      <th>8</th>
      <td>自驾游去深圳都经过那些地方啊</td>
      <td>Travel-Query</td>
    </tr>
    <tr>
      <th>9</th>
      <td>给我转播今天的女子双打乒乓球比赛现场</td>
      <td>Video-Play</td>
    </tr>
  </tbody>
</table>
</div>



## 1.生成label文件


```python
labels=df[1].unique()
# 打开文件并写入列表中的元素  
with open('label.txt', 'w') as f:  
    for item in labels:  
        f.write(str(item) + '\n')
```


```python
!cat label.txt
```

    Travel-Query
    Music-Play
    FilmTele-Play
    Video-Play
    Radio-Listen
    HomeAppliance-Control
    Weather-Query
    Alarm-Update
    Calendar-Query
    TVProgram-Play
    Audio-Play
    Other



```python
%cd ~/PaddleNLP/applications/text_classification/multi_class
!mkdir data
```

    /home/aistudio/PaddleNLP/applications/text_classification/multi_class


## 2.划分数据集
- train_test_split直接按照 8:2 划分训练集、测试集


```python
import os
from sklearn.model_selection import train_test_split
# 划分训练及测试集
train_data, dev_data= train_test_split( df, test_size=0.2)
root='data'
train_filename = os.path.join(root, 'train.txt')
dev_filename = os.path.join(root, 'dev.txt')
train_data.to_csv(train_filename, index=False, sep="\t", header=None)
dev_data.to_csv(dev_filename, index=False, sep="\t", header=None)
```

## 3.数据整理

训练需要准备指定格式的本地数据集,如果没有已标注的数据集，可以参考[文本分类任务doccano数据标注使用指南](https://gitee.com/paddlepaddle/PaddleNLP/blob/develop/applications/text_classification/doccano.md)进行文本分类数据标注。指定格式本地数据集目录结构：

### 3.1目录结构

```bash
data/
├── train.txt # 训练数据集文件
├── dev.txt # 开发数据集文件
└── label.txt # 分类标签文件
```

### 3.2数据集格式
训练、开发、测试数据集 文件中文本与标签类别名用tab符'\t'分隔开，文本中避免出现tab符'\t'。

train.txt/dev.txt/test.txt 文件格式：

```bash
<文本>'\t'<标签>
<文本>'\t'<标签>
...
```
### 3.3分类标签格式
label.txt(分类标签文件)记录数据集中所有标签集合，每一行为一个标签名。

- label.txt 文件格式：

```bash
<标签>
<标签>

```


```python
!cp ~/data/data208091/test.csv data/test.txt
!cp ~/label.txt data/label.txt
```


```python
!tree data
```

    data
    ├── bad_case.txt
    ├── dev.txt
    ├── label.txt
    ├── test.txt
    └── train.txt
    
    0 directories, 5 files


# 四、模型训练

**使用使用 Trainer API 对模型进行微调**

只需输入模型、数据集等就可以使用 Trainer API 高效快速地进行预训练、微调和模型压缩等任务，可以一键启动多卡训练、混合精度训练、梯度累积、断点重启、日志显示等功能，Trainer API 还针对训练过程的通用训练配置做了封装，比如：优化器、学习率调度等。



## 1.训练参数

主要的配置的参数为：

- `do_train`: 是否进行训练。
- `do_eval`: 是否进行评估。
- `debug`: 与`do_eval`配合使用，是否开启debug模型，对每一个类别进行评估。
- `do_export`: 训练结束后是否导出静态图。
- `do_compress`: 训练结束后是否进行模型裁剪。
- `model_name_or_path`: 内置模型名，或者模型参数配置目录路径。默认为`ernie-3.0-tiny-medium-v2-zh`。
- `output_dir`: 模型参数、训练日志和静态图导出的保存目录。
- `device`: 使用的设备，默认为`gpu`。
- `num_train_epochs`: 训练轮次，使用早停法时可以选择100。
- `early_stopping`: 是否使用早停法，也即一定轮次后评估指标不再增长则停止训练。
- `early_stopping_patience`: 在设定的早停训练轮次内，模型在开发集上表现不再上升，训练终止；默认为4。
- `learning_rate`: 预训练语言模型参数基础学习率大小，将与learning rate scheduler产生的值相乘作为当前学习率。
- `max_length`: 最大句子长度，超过该长度的文本将被截断，不足的以Pad补全。提示文本不会被截断。
- `per_device_train_batch_size`: 每次训练每张卡上的样本数量。可根据实际GPU显存适当调小/调大此配置。
- `per_device_eval_batch_size`: 每次评估每张卡上的样本数量。可根据实际GPU显存适当调小/调大此配置。
- `max_length`: 最大句子长度，超过该长度的文本将被截断，不足的以Pad补全。提示文本不会被截断。
- `train_path`: 训练集路径，默认为"./data/train.txt"。
- `dev_path`: 开发集集路径，默认为"./data/dev.txt"。
- `test_path`: 测试集路径，默认为"./data/dev.txt"。
- `label_path`: 标签路径，默认为"./data/label.txt"。
- `bad_case_path`: 错误样本保存路径，默认为"./data/bad\_case.txt"。
- `width_mult_list`：裁剪宽度（multi head）保留的比例列表，表示对self\_attention中的 `q`、`k`、`v` 以及 `ffn` 权重宽度的保留比例，保留比例乘以宽度（multi haed数量）应为整数；默认是None。 训练脚本支持所有`TraingArguments`的参数，更多参数介绍可参考[TrainingArguments 参数介绍](https://gitee.com/link?target=https%3A%2F%2Fpaddlenlp.readthedocs.io%2Fzh%2Flatest%2Ftrainer.html%23trainingarguments)。

## 2.开始训练


```python
!python train.py \
    --do_train \
    --do_eval \
    --do_export \
    --model_name_or_path ernie-3.0-tiny-medium-v2-zh \
    --output_dir checkpoint \
    --device gpu \
    --num_train_epochs 100 \
    --early_stopping True \
    --early_stopping_patience 5 \
    --learning_rate 3e-5 \
    --max_length 128 \
    --per_device_eval_batch_size 32 \
    --per_device_train_batch_size 32 \
    --metric_for_best_model accuracy \
    --load_best_model_at_end \
    --logging_steps 5 \
    --evaluation_strategy epoch \
    --save_strategy epoch \
    --save_total_limit 1    
```

## 3.训练日志

```bash
[2023-04-11 17:30:31,229] [    INFO] -   Num examples = 2420
[2023-04-11 17:30:31,229] [    INFO] -   Total prediction steps = 76
[2023-04-11 17:30:31,229] [    INFO] -   Pre device batch size = 32
[2023-04-11 17:30:31,229] [    INFO] -   Total Batch size = 32

  0%|                                                    | 0/76 [00:00<?, ?it/s]
  9%|████                                        | 7/76 [00:00<00:01, 58.78it/s]
 17%|███████▎                                   | 13/76 [00:00<00:01, 53.60it/s]
 25%|██████████▊                                | 19/76 [00:00<00:01, 53.72it/s]
 33%|██████████████▏                            | 25/76 [00:00<00:00, 54.40it/s]
 41%|█████████████████▌                         | 31/76 [00:00<00:00, 53.83it/s]
 49%|████████████████████▉                      | 37/76 [00:00<00:00, 53.93it/s]
 57%|████████████████████████▎                  | 43/76 [00:00<00:00, 54.07it/s]
 64%|███████████████████████████▋               | 49/76 [00:00<00:00, 53.70it/s]
 72%|███████████████████████████████            | 55/76 [00:01<00:00, 54.41it/s]
 80%|██████████████████████████████████▌        | 61/76 [00:01<00:00, 53.62it/s]
 88%|█████████████████████████████████████▉     | 67/76 [00:01<00:00, 53.80it/s]
                                                                                
eval_loss: 0.38935500383377075, eval_accuracy: 0.9347107438016529, eval_micro_precision: 0.9347107438016529, eval_micro_recall: 0.9347107438016529, eval_micro_f1: 0.9347107438016529, eval_macro_precision: 0.8868087630817776, eval_macro_recall: 0.883506204765109, eval_macro_f1: 0.8840559834605317, eval_runtime: 1.4272, eval_samples_per_second: 1695.617, eval_steps_per_second: 53.251, epoch: 12.0
 12%|████▌                                 | 3636/30300 [02:54<15:32, 28.61it/s]
100%|███████████████████████████████████████████| 76/76 [00:01<00:00, 58.29it/s]
                                                                                [2023-04-11 17:30:32,658] [    INFO] - Saving model checkpoint to checkpoint/checkpoint-3636
[2023-04-11 17:30:32,659] [    INFO] - Configuration saved in checkpoint/checkpoint-3636/config.json
[2023-04-11 17:30:33,291] [    INFO] - tokenizer config file saved in checkpoint/checkpoint-3636/tokenizer_config.json
[2023-04-11 17:30:33,291] [    INFO] - Special tokens file saved in checkpoint/checkpoint-3636/special_tokens_map.json
[2023-04-11 17:30:34,681] [    INFO] - Deleting older checkpoint [checkpoint/checkpoint-3333] due to args.save_total_limit
[2023-04-11 17:30:34,814] [    INFO] - 
Training completed. 

[2023-04-11 17:30:34,814] [    INFO] - Loading best model from checkpoint/checkpoint-2121 (score: 0.9400826446280992).
train_runtime: 177.1813, train_samples_per_second: 5463.331, train_steps_per_second: 171.011, train_loss: 0.3273027794348789, epoch: 12.0
 12%|████▌                                 | 3636/30300 [02:57<21:39, 20.52it/s]
[2023-04-11 17:30:35,236] [    INFO] - Saving model checkpoint to checkpoint
[2023-04-11 17:30:35,238] [    INFO] - Configuration saved in checkpoint/config.json
[2023-04-11 17:30:35,875] [    INFO] - tokenizer config file saved in checkpoint/tokenizer_config.json
[2023-04-11 17:30:35,875] [    INFO] - Special tokens file saved in checkpoint/special_tokens_map.json
[2023-04-11 17:30:35,876] [    INFO] - ***** train metrics *****
[2023-04-11 17:30:35,876] [    INFO] -   epoch                    =       12.0
[2023-04-11 17:30:35,876] [    INFO] -   train_loss               =     0.3273
[2023-04-11 17:30:35,876] [    INFO] -   train_runtime            = 0:02:57.18
[2023-04-11 17:30:35,876] [    INFO] -   train_samples_per_second =   5463.331
[2023-04-11 17:30:35,876] [    INFO] -   train_steps_per_second   =    171.011
[2023-04-11 17:30:36,113] [    INFO] - ***** Running Evaluation *****
[2023-04-11 17:30:36,113] [    INFO] -   Num examples = 2420
[2023-04-11 17:30:36,113] [    INFO] -   Total prediction steps = 76
[2023-04-11 17:30:36,113] [    INFO] -   Pre device batch size = 32
[2023-04-11 17:30:36,113] [    INFO] -   Total Batch size = 32
100%|███████████████████████████████████████████| 76/76 [00:01<00:00, 55.75it/s]
[2023-04-11 17:30:37,541] [    INFO] - ***** eval metrics *****
[2023-04-11 17:30:37,541] [    INFO] -   epoch                   =       12.0
[2023-04-11 17:30:37,541] [    INFO] -   eval_accuracy           =     0.9401
[2023-04-11 17:30:37,541] [    INFO] -   eval_loss               =     0.2693
[2023-04-11 17:30:37,541] [    INFO] -   eval_macro_f1           =     0.8951
[2023-04-11 17:30:37,541] [    INFO] -   eval_macro_precision    =     0.8971
[2023-04-11 17:30:37,541] [    INFO] -   eval_macro_recall       =     0.8938
[2023-04-11 17:30:37,541] [    INFO] -   eval_micro_f1           =     0.9401
[2023-04-11 17:30:37,541] [    INFO] -   eval_micro_precision    =     0.9401
[2023-04-11 17:30:37,541] [    INFO] -   eval_micro_recall       =     0.9401
[2023-04-11 17:30:37,541] [    INFO] -   eval_runtime            = 0:00:01.42
[2023-04-11 17:30:37,541] [    INFO] -   eval_samples_per_second =   1695.007
[2023-04-11 17:30:37,541] [    INFO] -   eval_steps_per_second   =     53.232
[2023-04-11 17:30:37,543] [    INFO] - Exporting inference model to checkpoint/export/model
[2023-04-11 17:30:43,826] [    INFO] - Inference model exported.
[2023-04-11 17:30:43,827] [    INFO] - tokenizer config file saved in checkpoint/export/tokenizer_config.json
[2023-04-11 17:30:43,827] [    INFO] - Special tokens file saved in checkpoint/export/special_tokens_map.json
[2023-04-11 17:30:43,827] [    INFO] - id2label file saved in checkpoint/export/id2label.json
```

## 4.训练结果和可选模型

程序运行时将会自动进行训练，评估。同时训练过程中会自动保存开发集上最佳模型在指定的 `output_dir` 中，保存模型文件结构如下所示：

```bash
checkpoint/
├── export # 静态图模型
├── config.json # 模型配置文件
├── model_state.pdparams # 模型参数文件
├── tokenizer_config.json # 分词器配置文件
├── vocab.txt
└── special_tokens_map.json
```

- 中文训练任务（文本支持含部分英文）推荐使用"ernie-1.0-large-zh-cw"、"ernie-3.0-tiny-base-v2-zh"、"ernie-3.0-tiny-medium-v2-zh"、"ernie-3.0-tiny-micro-v2-zh"、"ernie-3.0-tiny-mini-v2-zh"、"ernie-3.0-tiny-nano-v2-zh"、"ernie-3.0-tiny-pico-v2-zh"。
- 英文训练任务推荐使用"ernie-3.0-tiny-mini-v2-en"、 "ernie-2.0-base-en"、"ernie-2.0-large-en"。
- 英文和中文以外语言的文本分类任务，推荐使用基于96种语言（涵盖法语、日语、韩语、德语、西班牙语等几乎所有常见语言）进行预训练的多语言预训练模型"ernie-m-base"、"ernie-m-large"，详情请参见[ERNIE-M论文](https://gitee.com/link?target=https%3A%2F%2Farxiv.org%2Fpdf%2F2012.15674.pdf)。

# 五、模型评估
训练后的模型我们可以开启debug模式，对每个类别分别进行评估，并打印错误预测样本保存在bad_case.txt。默认在GPU环境下使用，在CPU环境下修改参数配置为--device "cpu":

## 1.开始训练


```python
!python train.py \
    --do_eval \
    --debug True \
    --device gpu \
    --model_name_or_path checkpoint \
    --output_dir checkpoint \
    --per_device_eval_batch_size 32 \
    --max_length 128 \
    --test_path './data/dev.txt'
```

## 2.输出日志
```bash
[2023-04-11 17:38:48,156] [    INFO] - ----------------------------
[2023-04-11 17:38:48,156] [    INFO] - Class name: Calendar-Query
[2023-04-11 17:38:48,156] [    INFO] - Evaluation examples in dev dataset: 241(10.0%) | precision: 99.17 | recall: 99.17 | F1 score 99.17
[2023-04-11 17:38:48,156] [    INFO] - ----------------------------
[2023-04-11 17:38:48,156] [    INFO] - Class name: TVProgram-Play
[2023-04-11 17:38:48,156] [    INFO] - Evaluation examples in dev dataset: 47(1.9%) | precision: 71.43 | recall: 63.83 | F1 score 67.42
[2023-04-11 17:38:48,156] [    INFO] - ----------------------------
[2023-04-11 17:38:48,156] [    INFO] - Class name: Audio-Play
[2023-04-11 17:38:48,156] [    INFO] - Evaluation examples in dev dataset: 49(2.0%) | precision: 78.43 | recall: 81.63 | F1 score 80.00
[2023-04-11 17:38:48,156] [    INFO] - ----------------------------
[2023-04-11 17:38:48,156] [    INFO] - Class name: Other
[2023-04-11 17:38:48,156] [    INFO] - Evaluation examples in dev dataset: 40(1.7%) | precision: 65.85 | recall: 67.50 | F1 score 66.67
[2023-04-11 17:38:48,156] [    INFO] - ----------------------------
[2023-04-11 17:38:48,158] [    INFO] - Bad case in dev dataset saved in ./data/bad_case.txt
100%|███████████████████████████████████████████| 76/76 [00:01<00:00, 55.79it/s]
```

## 3.错误分析
预测错误的会进行题型，bad case。

文本分类预测过程中常会遇到诸如"模型为什么会预测出错误的结果"，"如何提升模型的表现"等问题。[Analysis模块](https://gitee.com/paddlepaddle/PaddleNLP/blob/develop/applications/text_classification/multi_class/analysis) 提供了**可解释性分析、数据优化**等功能，旨在帮助开发者更好地分析文本分类模型预测结果和对模型效果进行优化。

具体见 bad_case.txt


```python
!head -n10 data/bad_case.txt
```

    Text	Label	Prediction
    一禅小和尚第4集往后接着播放我要看呢	Video-Play	FilmTele-Play
    济南生活的交通进行时还在直播中吗我想看下	TVProgram-Play	Video-Play
    能否回放一下早上七点二十分的时事关提案吗我想看下	Video-Play	TVProgram-Play
    播放一下那个启航	FilmTele-Play	Music-Play
    电视只有声音而没有图像该打什么号码的电话	HomeAppliance-Control	Other
    最近有什么新电影，调到小楚和野营的节目爱电影了解一下	Radio-Listen	TVProgram-Play
    那射手座呢牧羊座呢牧羊座是白羊座吗	Other	Calendar-Query
    飞轮海负伤排舞谢歌迷为保持最佳状态进补图	Other	Video-Play
    吴彦祖还表示一旦老婆有了他就会停工一年当专业奶爸	Music-Play	FilmTele-Play


# 六、源码分析
## 1.train.py

```python


import functools
import json
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import paddle
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    precision_recall_fscore_support,
)
from utils import log_metrics_debug, preprocess_function, read_local_dataset

from paddlenlp.data import DataCollatorWithPadding
from paddlenlp.datasets import load_dataset
from paddlenlp.trainer import (
    CompressionArguments,
    EarlyStoppingCallback,
    PdArgumentParser,
    Trainer,
)
from paddlenlp.transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    export_model,
)
from paddlenlp.utils.log import logger


# 支持的模型列表
SUPPORTED_MODELS = [
    "ernie-1.0-large-zh-cw",
    "ernie-1.0-base-zh-cw",
    "ernie-3.0-xbase-zh",
    "ernie-3.0-base-zh",
    "ernie-3.0-medium-zh",
    "ernie-3.0-micro-zh",
    "ernie-3.0-mini-zh",
    "ernie-3.0-nano-zh",
    "ernie-3.0-tiny-base-v2-zh",
    "ernie-3.0-tiny-medium-v2-zh",
    "ernie-3.0-tiny-micro-v2-zh",
    "ernie-3.0-tiny-mini-v2-zh",
    "ernie-3.0-tiny-nano-v2-zh ",
    "ernie-3.0-tiny-pico-v2-zh",
    "ernie-2.0-large-en",
    "ernie-2.0-base-en",
    "ernie-3.0-tiny-mini-v2-en",
    "ernie-m-base",
    "ernie-m-large",
]


# 默认参数
# yapf: disable
@dataclass
class DataArguments:
    max_length: int = field(default=128, metadata={"help": "Maximum number of tokens for the model."})
    early_stopping: bool = field(default=False, metadata={"help": "Whether apply early stopping strategy."})
    early_stopping_patience: int = field(default=4, metadata={"help": "Stop training when the specified metric worsens for early_stopping_patience evaluation calls"})
    debug: bool = field(default=False, metadata={"help": "Whether choose debug mode."})
    train_path: str = field(default='./data/train.txt', metadata={"help": "Train dataset file path."})
    dev_path: str = field(default='./data/dev.txt', metadata={"help": "Dev dataset file path."})
    test_path: str = field(default='./data/dev.txt', metadata={"help": "Test dataset file path."})
    label_path: str = field(default='./data/label.txt', metadata={"help": "Label file path."})
    bad_case_path: str = field(default='./data/bad_case.txt', metadata={"help": "Bad case file path."})


@dataclass
class ModelArguments:
    model_name_or_path: str = field(default="ernie-3.0-tiny-medium-v2-zh", metadata={"help": "Build-in pretrained model name or the path to local model."})
    export_model_dir: Optional[str] = field(default=None, metadata={"help": "Path to directory to store the exported inference model."})
# yapf: enable


def main():
    """
    Training a binary or multi classification model
    """

    parser = PdArgumentParser((ModelArguments, DataArguments, CompressionArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    if training_args.do_compress:
        training_args.strategy = "dynabert"
    if training_args.do_train or training_args.do_compress:
        training_args.print_config(model_args, "Model")
        training_args.print_config(data_args, "Data")
    paddle.set_device(training_args.device)

    # Define id2label
    id2label = {}
    label2id = {}
    with open(data_args.label_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            l = line.strip()
            id2label[i] = l
            label2id[l] = i

    # Define model & tokenizer
    if os.path.isdir(model_args.model_name_or_path):
        model = AutoModelForSequenceClassification.from_pretrained(
            model_args.model_name_or_path, label2id=label2id, id2label=id2label
        )
    elif model_args.model_name_or_path in SUPPORTED_MODELS:
        model = AutoModelForSequenceClassification.from_pretrained(
            model_args.model_name_or_path, num_classes=len(label2id), label2id=label2id, id2label=id2label
        )
    else:
        raise ValueError(
            f"{model_args.model_name_or_path} is not a supported model type. Either use a local model path or select a model from {SUPPORTED_MODELS}"
        )
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)

    # load and preprocess dataset
    train_ds = load_dataset(read_local_dataset, path=data_args.train_path, label2id=label2id, lazy=False)
    dev_ds = load_dataset(read_local_dataset, path=data_args.dev_path, label2id=label2id, lazy=False)
    trans_func = functools.partial(preprocess_function, tokenizer=tokenizer, max_length=data_args.max_length)
    train_ds = train_ds.map(trans_func)
    dev_ds = dev_ds.map(trans_func)

    if data_args.debug:
        test_ds = load_dataset(read_local_dataset, path=data_args.test_path, label2id=label2id, lazy=False)
        test_ds = test_ds.map(trans_func)

    # Define the metric function.
    def compute_metrics(eval_preds):
        pred_ids = np.argmax(eval_preds.predictions, axis=-1)
        metrics = {}
        metrics["accuracy"] = accuracy_score(y_true=eval_preds.label_ids, y_pred=pred_ids)
        for average in ["micro", "macro"]:
            precision, recall, f1, _ = precision_recall_fscore_support(
                y_true=eval_preds.label_ids, y_pred=pred_ids, average=average
            )
            metrics[f"{average}_precision"] = precision
            metrics[f"{average}_recall"] = recall
            metrics[f"{average}_f1"] = f1
        return metrics

    def compute_metrics_debug(eval_preds):
        pred_ids = np.argmax(eval_preds.predictions, axis=-1)
        metrics = classification_report(eval_preds.label_ids, pred_ids, output_dict=True)
        return metrics

    # Define the early-stopping callback.
    if data_args.early_stopping:
        callbacks = [EarlyStoppingCallback(early_stopping_patience=data_args.early_stopping_patience)]
    else:
        callbacks = None

    # 定义 Trainer
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        criterion=paddle.nn.loss.CrossEntropyLoss(),
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        callbacks=callbacks,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics_debug if data_args.debug else compute_metrics,
    )

    # 训练
    if training_args.do_train:
        train_result = trainer.train()
        metrics = train_result.metrics
        trainer.save_model()
        trainer.log_metrics("train", metrics)
        for checkpoint_path in Path(training_args.output_dir).glob("checkpoint-*"):
            shutil.rmtree(checkpoint_path)

    # 测试、预测
    if training_args.do_eval:
        if data_args.debug:
            output = trainer.predict(test_ds)
            log_metrics_debug(output, id2label, test_ds, data_args.bad_case_path)
        else:
            eval_metrics = trainer.evaluate()
            trainer.log_metrics("eval", eval_metrics)

    # 模型导出
    if training_args.do_export:
        if model.init_config["init_class"] in ["ErnieMForSequenceClassification"]:
            input_spec = [paddle.static.InputSpec(shape=[None, None], dtype="int64", name="input_ids")]
        else:
            input_spec = [
                paddle.static.InputSpec(shape=[None, None], dtype="int64", name="input_ids"),
                paddle.static.InputSpec(shape=[None, None], dtype="int64", name="token_type_ids"),
            ]
        if model_args.export_model_dir is None:
            model_args.export_model_dir = os.path.join(training_args.output_dir, "export")
        export_model(model=trainer.model, input_spec=input_spec, path=model_args.export_model_dir)
        tokenizer.save_pretrained(model_args.export_model_dir)
        id2label_file = os.path.join(model_args.export_model_dir, "id2label.json")
        with open(id2label_file, "w", encoding="utf-8") as f:
            json.dump(id2label, f, ensure_ascii=False)
            logger.info(f"id2label file saved in {id2label_file}")

    # 模型压缩
    if training_args.do_compress:
        trainer.compress()
        for width_mult in training_args.width_mult_list:
            pruned_infer_model_dir = os.path.join(training_args.output_dir, "width_mult_" + str(round(width_mult, 2)))
            tokenizer.save_pretrained(pruned_infer_model_dir)
            id2label_file = os.path.join(pruned_infer_model_dir, "id2label.json")
            with open(id2label_file, "w", encoding="utf-8") as f:
                json.dump(id2label, f, ensure_ascii=False)
                logger.info(f"id2label file saved in {id2label_file}")

    for path in Path(training_args.output_dir).glob("runs"):
        shutil.rmtree(path)


if __name__ == "__main__":
    main()

```

## 2.utils.py

```python


import numpy as np

from paddlenlp.utils.log import logger

# 预处理
def preprocess_function(examples, tokenizer, max_length, is_test=False):
    """
    Builds model inputs from a sequence for sequence classification tasks
    by concatenating and adding special tokens.
    """
    result = tokenizer(examples["text"], max_length=max_length, truncation=True)
    if not is_test:
        result["labels"] = np.array([examples["label"]], dtype="int64")
    return result

# 读取数据集
def read_local_dataset(path, label2id=None, is_test=False):
    """
    Read dataset.
    """
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if is_test:
                sentence = line.strip()
                yield {"text": sentence}
            else:
                items = line.strip().split("\t")
                yield {"text": items[0], "label": label2id[items[1]]}


# 打印日志                
def log_metrics_debug(output, id2label, dev_ds, bad_case_path):
    """
    Log metrics in debug mode.
    """
    predictions, label_ids, metrics = output
    pred_ids = np.argmax(predictions, axis=-1)
    logger.info("-----Evaluate model-------")
    logger.info("Dev dataset size: {}".format(len(dev_ds)))
    logger.info("Accuracy in dev dataset: {:.2f}%".format(metrics["test_accuracy"] * 100))
    logger.info(
        "Macro average | precision: {:.2f} | recall: {:.2f} | F1 score {:.2f}".format(
            metrics["test_macro avg"]["precision"] * 100,
            metrics["test_macro avg"]["recall"] * 100,
            metrics["test_macro avg"]["f1-score"] * 100,
        )
    )
    for i in id2label:
        l = id2label[i]
        logger.info("Class name: {}".format(l))
        i = "test_" + str(i)
        if i in metrics:
            logger.info(
                "Evaluation examples in dev dataset: {}({:.1f}%) | precision: {:.2f} | recall: {:.2f} | F1 score {:.2f}".format(
                    metrics[i]["support"],
                    100 * metrics[i]["support"] / len(dev_ds),
                    metrics[i]["precision"] * 100,
                    metrics[i]["recall"] * 100,
                    metrics[i]["f1-score"] * 100,
                )
            )
        else:
            logger.info("Evaluation examples in dev dataset: 0 (0%)")
        logger.info("----------------------------")

    with open(bad_case_path, "w", encoding="utf-8") as f:
        f.write("Text\tLabel\tPrediction\n")
        for i, (p, l) in enumerate(zip(pred_ids, label_ids)):
            p, l = int(p), int(l)
            if p != l:
                f.write(dev_ds.data[i]["text"] + "\t" + id2label[l] + "\t" + id2label[p] + "\n")

    logger.info("Bad case in dev dataset saved in {}".format(bad_case_path))

```

# 七、模型预测
使用taskflow进行模型预测

- 加载模型
- 加载数据
- 进行预测

## 1.加载模型进行单个预测


```python
from paddlenlp import Taskflow

# 模型预测
cls = Taskflow("text_classification", task_path='checkpoint/export', is_static_model=True)
cls(["回放CCTV2的消费主张"])
```

    [2023-04-11 17:42:26,315] [    INFO] - We are using <class 'paddlenlp.transformers.ernie.tokenizer.ErnieTokenizer'> to load 'checkpoint/export'.
    W0411 17:42:26.472223   349 gpu_resources.cc:61] Please NOTE: device: 0, GPU Compute Capability: 7.0, Driver API Version: 11.2, Runtime API Version: 11.2
    W0411 17:42:26.475904   349 gpu_resources.cc:91] device: 0, cuDNN Version: 8.2.
    [2023-04-11 17:42:29,395] [    INFO] - Load id2label from checkpoint/export/id2label.json.





    [{'predictions': [{'label': 'TVProgram-Play', 'score': 0.9521104350237317}],
      'text': '回放CCTV2的消费主张'}]



## 2.读取待预测数据
读取待预测数据到列表


```python
with open('data/test.txt', 'r') as file:  
    mytests = file.readlines()

print(mytests[:3])
```

    ['回放CCTV2的消费主张\n', '给我打开玩具房的灯\n', '循环播放赵本山的小品相亲来听\n']


## 3.整体预测


```python
result = cls(mytests)
```


```python
print(result[:3])
```

    [{'predictions': [{'label': 'TVProgram-Play', 'score': 0.9521104350237317}], 'text': '回放CCTV2的消费主张\n'}, {'predictions': [{'label': 'HomeAppliance-Control', 'score': 0.9970951493859599}], 'text': '给我打开玩具房的灯\n'}, {'predictions': [{'label': 'Audio-Play', 'score': 0.9710607817649783}], 'text': '循环播放赵本山的小品相亲来听\n'}]


## 4.按格式保存


```python
f=open('/home/aistudio/result.txt', 'w')
f.write("ID,Target\n")
for i in range(len(result)):
    f.write(f"{i+1},{result[i]['predictions'][0]['label']}\n")
f.close()
```


```python
!head -n10 /home/aistudio/result.txt
```

    ID,Target
    1,TVProgram-Play
    2,HomeAppliance-Control
    3,Audio-Play
    4,Alarm-Update
    5,HomeAppliance-Control
    6,FilmTele-Play
    7,FilmTele-Play
    8,Music-Play
    9,Calendar-Query


# 八、提交结果

![](https://ai-studio-static-online.cdn.bcebos.com/6ac1d48783414efbae3fdae211d00f0e7b8a62a6ad4e4ebb84a9677c70d4d93d)

