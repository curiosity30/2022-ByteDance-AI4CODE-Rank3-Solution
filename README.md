# 2022-ByteDance-AI4CODE-Rank3-Solution

**2022字节跳动技术公益创新杯-用户浏览短视频兴趣预测Rank3方案**

**Curiosity队伍方案（B榜Rank3，A榜Rank12）**

### 1.1 数据预处理

对开源baseline代码进行一定修改，读取数据，按照测试集中出现的userid和videoid对训练集进行数据筛选，最后得到2000w左右的数据作为训练集。

### 1.2 特征工程

- 原始特征（3个）
  - userid / videoid / tag
- 用户对于每个tag的反馈统计特征（8个）
  - “反馈”指用户点赞、收藏、转发的动作
  - 每个用户关于各类tag视频的曝光次数（基于全局统计）
  - 每个用户对于各类tag视频的曝光比例（训练集统计）
  - 用户对每种tag的点赞、收藏、转发的比例（训练集统计）
  - 用户点赞、收藏、转发中每种tag所占比例（训练集统计）
- 视频的反馈统计特征（3个）
  - 视频的点赞、收藏、转发的总次数（训练集统计）
- 分箱特征（3个）
  - 对 userid / videoid / tag 统计全局count特征，然后进行等频分箱
- 交叉特征（1个）
  - user + tag 交叉
- 相似度特征（1个）
  - 通过SVD分解得到video的embedding向量
  - user embedding为用户点赞和完播视频的embedding的平均池化
  - 然后计算每个曝光视频与user embedding 的余弦相似度
- kmeans聚类特征（2个）
  - 对 video embedding 通过聚类得到 48 个类别
  - 对 user embedding 通过聚类得到 16 个类别

> 总共得到21个特征，其中9个分类特征（sparse features），12个数值特征（dense features）。
> 
> 其中，数值特征通过Quantile Transformer进行标准化，然后再使用等频分箱，将其转化为分类特征。最后输入到模型中。Embedding维度设为16。

### 1.3 模型训练

使用五折 DCN 模型进行训练和预测。

模型参数如下：

```python
EMBED_DIM = 16
dnn_dropout = 0.3
hidden_units = [256, 128, 64]
LR = 1e-3
BATCH_SIZE = 8192 # 1024/2048/4096/8192/16384
epochs = 2
```

## 2 模型简要介绍

Deep&Cross Network，简称DCN模型。谷歌和斯坦福大学在2017年提出的CTR预估模型，左侧使用cross network显式构建特征交叉，右侧使用简单MLP隐式建模高阶交叉特征，最后输出点击概率。

主要优点是模型参数规模小，特征交叉操作简洁高效，模型推理速度快。

参考文献：Ruoxi Wang, Bin Fu, Gang Fu, and Mingliang Wang. 2017. Deep Cross Network for Ad Click Predictions. In Proceedings of the ADKDD'17 (ADKDD'17). Association for Computing Machinery, New York, NY, USA, Article 12, 1–7. https://doi.org/10.1145/3124749.3124754.

## 3 代码说明

### 3.1 代码运行说明

- 运行 `train.ipynb` 即可

### 3.2 代码文件说明

- `train.ipynb`: **主函数**。调用`gen_similarity.py`、`gen_kmeans_label.py`、`feature_engineer.py`、`dcn.py`、`utils.py`，完成加载数据、特征工程、模型训练和预测、生成结果文件）
- `gen_similarity.py`: 生成**相似度特征**"user_video_sim"
- `gen_kmeans_label.py`: 生成用户和视频的**kmeans聚类特征**
- `feature_engineer.py`: **特征工程函数**
- `dcn.py`: **模型代码**
- `preprocess_data.ipynb`: **预处理函数**。筛选训练集数据、对训练集和测试集进行label encode，处理结束后的训练集和测试集分别保存为`df_train_sp.pkl`和`df_test_sp.pkl`。【NOTE：**无需运行**，数据已保存至`../temp_data/`】
- `utils.py`: **工具函数**（减少内存函数等）

## 1 整体思路

- 对训练集进行**数据预处理**
- 通过**特征工程**构造总共21个特征
- 使用**五折DCN（Deep&Cross Network）模型**进行训练和预测
