# soft_mask_bert 文本纠错与BERT的最新结合
soft_mask_bert model for Chinese Spelling Correction in keras

本代码是对论文[Spelling Error Correction with Soft-Masked BERT](https://arxiv.org/pdf/2005.07421.pdf)结果的复现。

# 测试环境
python 3.6 + tensorflow 2.3.1 + Keras-2.4.3

# 依赖 keras_bert

- [https://github.com/CyberZHG/keras-bert](https://github.com/CyberZHG/keras-bert)
- **注意事项**：修改keras_bert layers目录的`__init__.py`, 导出`PositionEmbedding`类
  - 假设python3的安装位置为：`/usr/local/lib/python3.6/`，执行如下命令
  - `sudo vim /usr/local/lib/python3.6/site-packages/keras_bert/layers/__init__.py`
  - 把第二行修改为`from .embedding import get_embedding, TokenEmbedding, EmbeddingSimilarity, PositionEmbedding` 

# 训练&测试数据集
训练&测试数据集从下面的github获取，sgml格式
- [https://github.com/wdimmy/Automatic-Corpus-Generation](https://github.com/wdimmy/Automatic-Corpus-Generation)

# Bert中文版权重

- 官方版： [https://github.com/google-research/bert](https://github.com/google-research/bert)
- 哈工大版： [https://github.com/ymcui/Chinese-BERT-wwm](https://github.com/ymcui/Chinese-BERT-wwm)

# 推荐阅读

- 我的知乎专栏：[https://www.zhihu.com/column/c_1247512235343265792](https://www.zhihu.com/column/c_1247512235343265792)
- 及同名微信公众号：算法工程师的进阶之路

# 中文拼写纠错 (Spelling Correction)

## 背景
拼写纠错 (Spelling Correction)任务的目标是在文本中查找并更正拼写错误 (typographical errors),这类错误通常发生在外观、发音相似的字符之间.

## 示例

输入:
```
1986年毕业于国防科技大学计算机应用专业，获学时学位。
```

输出:

```
1986年毕业于国防科技大学计算机应用专业，获学士学位。
(时 -> 士)
```

## 标准评价指标
拼写纠错 (Spelling Correction)任务通常使用accuracy, precision, recall, 以及F1 score指标进行评测.这些指标可以是字符层面(character level)的或者是句子层面(sentence level)的.通常会分别对识别(Detection)和纠正(Correction)进行评估.
* 识别(Detection): 识别一段文字中所有拼写错误字符的位置，应该做到与正确参考 (gold standard) 相同.
* 纠正(Correction): 识别的错误字符以及纠正错误字符，应与正确参考 (gold standard)相同.

## <span class="t">SIGHAN Bake-off: Chinese Spelling Check Task</span>.

* 版本: SIGHAN 数据集有3个版本 ([2015](http://anthology.aclweb.org/W/W15/W15-3106.pdf), [2014](http://anthology.aclweb.org/W/W14/W14-6820.pdf), [2013](http://anthology.aclweb.org/W/W13/W13-4406.pdf))
* 综述论文: [Tseng et. al. (2015)](http://anthology.aclweb.org/W/W15/W15-3106.pdf)
* Licence: [http://nlp.ee.ncu.edu.tw/resource/csc_download.html](http://nlp.ee.ncu.edu.tw/resource/csc_download.html) (academic use only)

| test set | # sentence pairs | # characters | #拼写错误 (chars) | 字符集 | 主题 (Genre)  |
| --- | --- | --- | --- | --- | --- |
| SIGHAN 2015 ([Tseng et. al. 2015](http://aclweb.org/anthology/W15-3106))| 1,100 | 33,711 | 715 | 繁体 | 第二语言学习 (second-language learning) |
| SIGHAN 2014 ([Yu et. al. 2014](https://www.aclweb.org/anthology/W14-6820))| 1,062 | 53,114 | 792 | 繁体 | 第二语言学习 (second-language learning) |
| SIGHAN 2013 ([Wu et. al. 2013](https://www.aclweb.org/anthology/W13-4406))| 2,000 | 143,039 | 1,641 | 繁体 | 第二语言学习 (second-language learning) |

### 评价指标


* (1) 误报率 (False Positive Rate), (2) 识别正确率 (Accuracy), (3) 识别精确率 (precision), (4) 识别召回率 (recall), (5) 识别F1, (6) 纠正正确率 (Accuracy), (7) 纠正精确率 (precision), (8) 纠正 召回率 (recall), (9) 纠正 F1
* 具体实现: http://nlp.ee.ncu.edu.tw/resource/csc_download.html

### 结果

| System | False Positive Rate | Detection Accuracy | Detection Precision | Detection Recall| Detection F1| Correction Accuracy | Correction Precision | Correction Recall | Correction F1|
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Soft-Masked BERT ([Zhang et. al. 2020](https://www.aclweb.org/anthology/2020.acl-main.82))| - | 80.9 | 73.7 | 73.2 | 73.5 | 77.4 | 66.7 | 66.2 | 66.4 |
| Confusion-set ([Wang et. al. 2019](https://www.aclweb.org/anthology/P19-1578/))| - | - | 66.8 | 73.1 | 69.8 | - | 71.5 | 59.5 | 64.9 |
| FASPell ([Hong et. al. 2019](https://www.aclweb.org/anthology/D19-5522/))| - | 74.2 | 67.6 | 60.0 | 63.5 | 73.7 | 66.6 | 59.1 | 62.6 |
| CAS ([Zhang et. al. 2015](http://aclweb.org/anthology/W15-3107))| 11.6 | 70.1 | 80.3 | 53.3 | 64.0 | 69.2 | 79.7 | 51.5 | 62.5 |
上表是不同模型在结果SIGHAN 2015测试集上的结果.

### 相关资源

| Source | # sentence pairs | # chars | #拼写错误 | 字符集 | 主题 (Genre)  |
| --- | --- | --- | --- | --- | --- |
| SIGHAN 2015 Training data ([Tseng et. al. 2015](http://aclweb.org/anthology/W15-3106)) | 2,334  | 146,076 | 2,594 | 繁体 | 第二语言学习 (second-language learning) |
  | SIGHAN 2014 Training data ([Yu et. al. 2014](http://ir.itc.ntnu.edu.tw/lre/clp14csc.html)) | 6,526  | 324,342 | 10,087 | 繁体 | 第二语言学习 (second-language learning)|
| SIGHAN 2013 Training data ([Wu et. al. 2013](http://ir.itc.ntnu.edu.tw/lre/sighan7csc.html)) | 350  | 17,220 | 350 | 繁体 | 第二语言学习 (second-language learning)|

## 其他资源

| Source | # sentence pairs | # chars | #拼写错误 | 字符集 | 主题 (Genre)  |
| --- | --- | --- | --- | --- | --- |
| Synthetic training dataset ([Wang et. al. 2018](https://www.aclweb.org/anthology/P19-1578)) | 271,329 | 12M | 382,702 | 简体 | 新闻 |
