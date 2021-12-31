<!--
 * @Author: zihe.zhu@qq.com
 * @Date: 2021-05-08 17:05:44
 * @LastEditTime: 2021-05-08 19:17:19
-->

<p align="center">
    <br>
    <img src="./appendix/sikubert.png" width="600"/>
    <br>
</p>
<p align="center">
<a href="https://github.com/SIKU-BERT/SikuBERT/blob/main/LICENSE"><img alt="GitHub license" src="https://img.shields.io/github/license/SIKU-BERT/SikuBERT"></a>
</p>


[TOC]

# 中文版

## 引言 

数字人文研究需要大规模语料库和高性能古文自然语言处理工具支持。预训练语言模型已经在英语和现代汉语文本上极大的提升了文本挖掘的精度，目前亟需专门面向古文自动处理领域的预训练模型。

我们以校验后的高质量《四库全书》全文语料作为训练集，基于BERT深度语言模型框架，构建了面向古文智能处理任务的SikuBERT和SikuRoBERTa预训练语言模型。

我们设计了面向《左传》语料的古文自动分词、断句标点、词性标注和命名实体识别4个下游任务，验证模型性能。


* `SikuBERT`和`SikuRoBERTa`基于`《四库全书》`语料训练，《四库全书》又称《钦定四库全书》，是清代乾隆时期编修的大型丛书。实验去除了原本中的注释部分，仅纳入正文部分，参与实验的训练集共纳入字数达`536,097,588`个，数据集内的汉字均为繁体中文。

* 基于领域适应训练（Domain-Adaptive Pretraining）的思想，`SikuBERT`和`SikuRoBERTa`在BERT结构的基础上结合大量古文语料，分别继续训练BERT和RoBERTa模型，以获取面向古文自动处理领域的预训练模型。

## 新闻 

- 2021/5/8 模型加入[Huggingface Transformers](https://github.com/huggingface/transformers)预训练模型“全家桶”。
- 2021/5/6  论文被第五届全国未来智慧图书馆发展论坛会议录用 
- 2021/8/20 论文于《图书馆论坛》[网络首发](https://kns.cnki.net/kcms/detail/detail.aspx?dbcode=CAPJ&dbname=CAPJLAST&filename=TSGL20210819003&v=hVxys9Ams5mx5uSYJEe%25mmd2BJZlUL9o0ByINTgCnWgJNacKD1c4igVV%25mmd2B%25mmd2F8yn71Wrz56s)
- 2021/9/13 更新sikuBERT和sikuRoberta,新发布的模型已具有包含《四库全书》原生词的新词表，新词表相比原先的bert-base的词表多了8000余字，在各项任务上的表现均超越前者。
- 2021/9/15 相关的python工具包sikufenci正式发布,可用于繁体古籍的自动分词，链接见https://github.com/SIKU-BERT/sikufenci
- 2021/11/6 本项目相关的单机版开源软件sikuaip正式发布，提供包括分词，断句，实体识别，文本分类等多种古文处理功能，可直接下载解压使用。链接见下文。
- 2021/12/10 基于本模型的第一个古汉语领域NLP工具评估比赛——**EvaHan 2022**发布，比赛详情见：https://circse.github.io/LT4HALA/2022/EvaHan

## 面向数字人文的古籍智能处理平台sikuaip1.0版本已正式发布
下载链接:https://pan.baidu.com/s/1--S-qyUedIvhBKwapQjPsA
提取码：m36d

平台的使用方法见“使用方法”文件夹，目前版本支持分词，断句，实体识别，文本分类，词性标注和自动标点六种功能，提供单文本处理和语料库处理两种文本处理模式，欢迎下载使用！




## 使用方法
### Huggingface Transformers
基于[Huggingface Transformers](https://github.com/huggingface/transformers)的`from_pretrained`方法可以直接在线获取SikuBERT和SikuRoBERTa模型。

- SikuBERT
```python
from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained("SIKU-BERT/sikubert")

model = AutoModel.from_pretrained("SIKU-BERT/sikubert")
```

- SikuRoBERTa
```python
from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained("SIKU-BERT/sikuroberta")

model = AutoModel.from_pretrained("SIKU-BERT/sikuroberta")
```

## 下载模型 
- 我们提供的模型是`PyTorch`版本。

### 调用
- 通过Huggingface官网直接下载，目前官网的模型已同步更新至最新版本:

    - **SikuBERT: https://huggingface.co/SIKU-BERT/sikubert**

    - **SikuRoBERTa: https://huggingface.co/SIKU-BERT/sikuroberta**

### 云盘 

旧版下载地址:
| 模型名称 | 网盘链接 |
| :-----  | :------ |
| sikubert | [链接](https://pan.baidu.com/s/1kGVzjyfDLcx4i52Vtzp_wA) 提取码: jn94 |
| sikuroberta | [链接](https://pan.baidu.com/s/1T8lQ0w6tkGDBO_F_DyrTVw) 提取码: ihgq |

拥有新词表的sikubert和sikuroberta下载方式已更新:
| 模型名称 | 网盘链接 |
| :-----  | :------ |
| sikubert_vocabtxt(推荐下载) | [链接](https://pan.baidu.com/s/1uuYgJMsST08usCl3902Gpw) 提取码: v68d |
| sikuroberta_vocabtxt(推荐下载) | [链接](https://pan.baidu.com/s/18Cbi8vA9nfOD0NA3qTOsdA) 提取码: 93cr |




## 验证与结果
- 我们SikuBERT和SikuRoBERTa应用在语源为繁体中文的《左传》语料上进行古文自动分词、词性标注、断句和实体识别实验。实验结果如下。

| 任务名   task type  | 预训练模型pretrained models | 精确率（P） | 召回率（R） | 调和平均值（F1） |
|:----------:|:-----------:|:------:|:-----:|:-----:|
| 分词 Participle   | BERT-base-chinese      | 86.99% | 88.15% | 87.56%    |
|               | RoBERTa                | 80.90% | 84.77% | 82.79%    |
|               | SikuBERT               | 88.62% | 89.08% | 88.84%    |
|               | SikuRoBERTa            | 88.48% | 89.03% | **88.88%**    |
| 词性标注 POS tag     | BERT-base-chinese      | 89.51% | 90.10% | 89.73%    |
|               | RoBERTa                | 86.70% | 88.45% | 87.50%    |
|               | SikuBERT               | 89.89% | 90.41% | **90.10%**    |
|               | SikuRoBERTa            | 89.74% | 90.49% | 90.06%    |
| 断句 Segmentation| BERT-base-chinese      | 78.77% | 78.63% | 78.70%    |
|               | RoBERTa                | 66.71% | 66.38% | 66.54%    |
|               | SikuBERT               | 87.38% | 87.68% | **87.53%**    |
|               | SikuRoBERTa            | 86.81% | 87.02% | 86.91%    |


| 任务名 task type    | 预训练模型pretrained models | 实体名 entity names | 精确率（P）                 | 召回率（R）  | 调和平均值（F1） |
|:----------:|:-----------:|:------:|:-----:|:-----:|:-----:|
| 实体识别  NER  | BERT-base-chinese      | nr(人名)  | 86.66%    | 87.35% | 87.00% |
|               |                        | ns(地名)  | 83.99%    | 87.00% | 85.47% |
|               |                        | t(时间)   | 96.96%    | 95.15% | 96.05% |
|               |                        | avg/prf | 86.99%    | 88.15% | 87.56% |
|               | RoBERTa                | nr(人名)  | 79.88%    | 83.69% | 81.74% |
|               |                        | ns(地名)  | 78.86%    | 84.08% | 81.39% |
|               |                        | t(时间)   | 91.45%    | 91.79% | 91.62% |
|               |                        | avg/prf | 80.90%    | 84.77% | 82.79% |
|               | SikuBERT               | nr(人名)  | 88.65%    | 88.23% | 88.44% |
|               |                        | ns(地名)  | 85.48%    | 88.20% | 86.81% |
|               |                        | t(时间)   | 97.34%    | 95.52% | 96.42% |
|               |                        | avg/prf | 88.62%    | 89.08% | 88.84% |
|               | SikuRoBERTa            | nr(人名)  | 87.74%    | 88.23% | **87.98%** |
|               |                        | ns(地名)  | 86.55%    | 88.73% | **87.62%** |
|               |                        | t(时间)   | 97.35%    | 95.90% | **96.62%** |
|               |                        | avg/prf | 88.48%    | 89.30% | **88.88%**|


## 引用
- 如果我们的内容有助您研究工作，欢迎在论文中引用。
- GB/T 7714-2015格式：[1]王东波,刘畅,朱子赫,刘江峰,胡昊天,沈思,李斌.SikuBERT与SikuRoBERTa：面向数字人文的《四库全书》预训练模型构建及应用研究[J/OL].图书馆论坛:1-14[2021-08-21].http://kns.cnki.net/kcms/detail/44.1306.G2.20210819.2052.008.html.


## 免责声明
- 报告中所呈现的实验结果仅表明在特定数据集和超参组合下的表现，并不能代表各个模型的本质。实验结果可能因随机数种子，计算设备而发生改变。**使用者可以在许可证范围内任意使用该模型，但我们不对因使用该项目内容造成的直接或间接损失负责。**


## 致谢
- SikuBERT是基于[中文BERT预训练模型](https://huggingface.co/bert-base-chinese)继续训练。
- SikuRoBERTa是基于[中文BERT-wwm](https://github.com/ymcui/Chinese-BERT-wwm)项目中`RoBERTa-wwm-ext`继续训练的。

## 联系我们
- Chang 649164915@qq.com
- Joe zihe.zhu@qq.com

# English version 

## Introduction

Digital humanities research needs the support of large-scale corpus and high-performance ancient Chinese natural language processing tools. The pre-training language model has greatly improved the accuracy of text mining in English and modern Chinese texts. At present, there is an urgent need for a pre-training model specifically for the automatic processing of ancient texts. 

We used the verified high-quality "Siku Quanshu" full text corpus as the training set, and based on the BERT deep language model framework, we constructed SikuBERT and SikuRoBERTa pre-training language models for intelligent processing tasks of ancient Chinese. 

We designed four downstream tasks of automatic word segmentation, segmentation punctuation, part-of-speech tagging, and named entity recognition for ancient Chinese corpus for "Zuo Zhuan" to verify the performance of the model. 

- `SikuBERT` and `SikuRoBERTa` are trained on the corpus of "`Siku Quanshu`". "Siku Quanshu", also known as "King Ding Siku Quanshu", is a large-scale series of books compiled during the Qianlong period of the Qing Dynasty. The experiment removed the original annotation part and only included the text part. The training set involved in the experiment included a total of `536,097,588` characters, and the Chinese characters in the data set were all traditional Chinese. 
- Based on the idea of Domain-Adaptive Pretraining, `SikuBERT` and `SikuRoBERTa` combine a large amount of ancient text corpus based on the BERT structure, and continue to train the BERT and RoBERTa models respectively to obtain pre-training models for the automatic processing of ancient texts. 

## News 

- 2021/5/8 model added [Huggingface Transformers](https://github.com/huggingface/transformers) pre-trained model "Family Bucket". 
- 2021/5/6 papers accepted by the 5th National Future Smart Library Development Forum
- 2021/8/20 papers will be published in "Library Forum" [Internet First](https://kns.cnki.net/kcms/detail/detail.aspx?dbcode=CAPJ&dbname=CAPJLAST&filename=TSGL20210819003&v=hVxys9Ams5mx5uSYJEe%25mmd2BJZlCnoc0g2INTV% 25mmd2F8yn71Wrz56s) 
- 2021/9/13 update sikuBERT and sikuRoberta, the newly released model has a new vocabulary that contains the original words of "Siku Quanshu", the new vocabulary has more than 8,000 words more than the original bert-base vocabulary. The performance on all tasks surpassed the former.
-  2021/9/15 related python toolkit sikufenci is officially released, which can be used for automatic word segmentation of traditional ancient books, see https://github.com/SIKU-BERT/sikufenci for the link 
- 2021/11/6 The stand-alone open source software sikuaip related to this project is officially released, providing various ancient text processing functions including word segmentation, sentence segmentation, entity recognition, text classification, etc., which can be downloaded and decompressed directly. See the link below. 
- 2021/12/10 The first NLP tool evaluation competition in the field of ancient Chinese based on this model-**EvaHan 2022** is released. For details of the competition, please visit: https://circse.github.io/LT4HALA/2022/EvaHan 

## The sikuaip version 1.0 of the ancient book intelligent processing platform for digital humanities has been officially released

Download link: https://pan.baidu.com/s/1--S-qyUedIvhBKwapQjPsA Extraction code: m36d 

Please refer to the "How to use" folder for the usage of the platform. The current version supports six functions of word segmentation, sentence segmentation, entity recognition, text classification, part-of-speech tagging and automatic punctuation. It provides two text processing modes: single text processing and corpus processing. Welcome to download and use ! 



##  How to use

### Huggingface Transformers

The `from_pretrained` method based on [Huggingface Transformers](https://github.com/huggingface/transformers) can directly obtain SikuBERT and SikuRoBERTa models online. 

- SikuBERT

```python
from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained("SIKU-BERT/sikubert")

model = AutoModel.from_pretrained("SIKU-BERT/sikubert")
```

- SikuRoBERTa

```python
from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained("SIKU-BERT/sikuroberta")

model = AutoModel.from_pretrained("SIKU-BERT/sikuroberta")
```

## Download PTM

- The model we provide is the `PyTorch` version. 

### From Huggingface

- Download directly through Huggingface's official website, and the model on the official website has been updated to the latest version simultaneously: 

  - **SikuBERT: https://huggingface.co/SIKU-BERT/sikubert**

  - **SikuRoBERTa: https://huggingface.co/SIKU-BERT/sikuroberta**

### From Google Drive

If you are not in China, We put the model on Google drive for users to download, and we are working hard to upload it, please wait~

Old version download address: 

| Model       | Link      |
| :---------- | :-------- |
| sikubert    | https://drive.google.com/drive/folders/1blElNRhouuaU-ZGA99ahud1QL7Y-7PEZ?usp=sharing |
| sikuroberta | https://drive.google.com/drive/folders/13ToN58XfsfHIIj7pjLWNgLvWAqHsb0_a?usp=sharing |

The download method of sikubert and sikuroberta with new vocabulary has been updated: 

| Model                                       | Link      |
| :------------------------------------------ | :-------- |
| sikubert_vocabtxt(Recommended download  )   | https://drive.google.com/drive/folders/1uA7m54Cz7ZhNGxFM_DsQTpElb9Ns77R5?usp=sharing |
| sikuroberta_vocabtxt(Recommended download ) | https://drive.google.com/drive/folders/1i0ldNODE1NC25Wzv0r7v1Thda8NscK3e?usp=sharing |

##  Evaluation & Results

- We use SikuBERT and SikuRoBERTa to perform automatic word segmentation, part-of-speech tagging, sentence segmentation and entity recognition experiments on the "Zuo Zhuan" corpus whose etymology is traditional Chinese. The experimental results are as follows.

|  task type   | pretrained models | （P）  | （R）  |   （F1）   |
| :----------: | :---------------: | :----: | :----: | :--------: |
|  Participle  | BERT-base-chinese | 86.99% | 88.15% |   87.56%   |
|              |      RoBERTa      | 80.90% | 84.77% |   82.79%   |
|              |     SikuBERT      | 88.62% | 89.08% |   88.84%   |
|              |    SikuRoBERTa    | 88.48% | 89.03% | **88.88%** |
|   POS tag    | BERT-base-chinese | 89.51% | 90.10% |   89.73%   |
|              |      RoBERTa      | 86.70% | 88.45% |   87.50%   |
|              |     SikuBERT      | 89.89% | 90.41% | **90.10%** |
|              |    SikuRoBERTa    | 89.74% | 90.49% |   90.06%   |
| Segmentation | BERT-base-chinese | 78.77% | 78.63% |   78.70%   |
|              |      RoBERTa      | 66.71% | 66.38% |   66.54%   |
|              |     SikuBERT      | 87.38% | 87.68% | **87.53%** |
|              |    SikuRoBERTa    | 86.81% | 87.02% |   86.91%   |


| task type | pretrained models |  entity names   | （P）  | （R）  |   （F1）   |
| :-------: | :---------------: | :-------------: | :----: | :----: | :--------: |
|    NER    | BERT-base-chinese | nr(people name) | 86.66% | 87.35% |   87.00%   |
|           |                   | ns(place name)  | 83.99% | 87.00% |   85.47%   |
|           |                   |     t(time)     | 96.96% | 95.15% |   96.05%   |
|           |                   |     avg/prf     | 86.99% | 88.15% |   87.56%   |
|           |      RoBERTa      | nr(people name) | 79.88% | 83.69% |   81.74%   |
|           |                   | ns(place name)  | 78.86% | 84.08% |   81.39%   |
|           |                   |     t(time)     | 91.45% | 91.79% |   91.62%   |
|           |                   |     avg/prf     | 80.90% | 84.77% |   82.79%   |
|           |     SikuBERT      | nr(people name) | 88.65% | 88.23% |   88.44%   |
|           |                   | ns(place name)  | 85.48% | 88.20% |   86.81%   |
|           |                   |     t(time)     | 97.34% | 95.52% |   96.42%   |
|           |                   |     avg/prf     | 88.62% | 89.08% |   88.84%   |
|           |    SikuRoBERTa    | nr(people name) | 87.74% | 88.23% | **87.98%** |
|           |                   | ns(place name)  | 86.55% | 88.73% | **87.62%** |
|           |                   |     t(time)     | 97.35% | 95.90% | **96.62%** |
|           |                   |     avg/prf     | 88.48% | 89.30% | **88.88%** |


## Citing

- If our content is helpful for your research work, please quote it in the paper. 
- GB/T 7714-2015格式：[1]王东波,刘畅,朱子赫,刘江峰,胡昊天,沈思,李斌.SikuBERT与SikuRoBERTa：面向数字人文的《四库全书》预训练模型构建及应用研究[J/OL].图书馆论坛:1-14[2021-08-21].http://kns.cnki.net/kcms/detail/44.1306.G2.20210819.2052.008.html.


## Disclaim

- The experimental results presented in the report only show the performance under a specific data set and hyperparameter combination, and cannot represent the essence of each model. The experimental results may change due to random number seeds and computing equipment. **Users can use the model arbitrarily within the scope of the license, but we are not responsible for the direct or indirect losses caused by using the content of the project. ** 


##  Acknowledgment

- SikuBERT is based on [Chinese BERT pre-training model](https://huggingface.co/bert-base-chinese) to continue training. 
- SikuRoBERTa continues training based on the `RoBERTa-wwm-ext` in the [中文BERT-wwm](https://github.com/ymcui/Chinese-BERT-wwm) project. 

## Contact us

- Chang 649164915@qq.com
- Joe zihe.zhu@qq.com
