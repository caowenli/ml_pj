import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import gc

plt.rcParams["font.sans-serif"] = "SimHei"  # 解决中文乱码问题
import seaborn as sns
import random

# 数据加载
train_data_path = r"/TianmaoRepeatBuyPrediction/data/data_format1/train_format1.csv"
test_data_path = r"/TianmaoRepeatBuyPrediction/data/data_format1/test_format1.csv"
user_info_path = r"/TianmaoRepeatBuyPrediction/data/data_format1/user_info_format1.csv"
user_log_path = r"/TianmaoRepeatBuyPrediction/data/data_format1/user_log_format1.csv"
train_data = pd.read_csv(train_data_path, sep=',')
test_data = pd.read_csv(test_data_path, sep=',')
user_info = pd.read_csv(user_info_path, sep=',')
user_log = pd.read_csv(user_log_path, sep=',')
# 数据探索性分析
# 1、数据基本情况分析
# 数据量：
print(train_data.shape)
print(test_data.shape)
print(user_info.shape)
print(user_log.shape)
# 数据的特征：
print(train_data.keys())
print(test_data.keys())
print(user_info.keys())
print(user_log.keys())
# 数据的类型
print(train_data.dtypes)
print(test_data.dtypes)
print(user_info.dtypes)
print(user_log.dtypes)
# 数据的缺失情况：
print(train_data.isnull().sum())
print(test_data.isnull().sum())
print(user_info.isnull().sum())
print(user_log.isnull().sum())
# 数据统计情况分析：
# 分析数据的：max、max、75%等分位情况、mean
print(train_data.max())
print(train_data.min())
print(train_data.mean())
print(test_data.max())
print(test_data.min())
print(test_data.mean())
print(user_info.max())
print(user_info.min())
print(user_info.mean())
print(user_log.max())
print(user_log.min())
print(user_log.mean())
# 数据的正负样本比例
print(train_data['label'].unique())
print(train_data['label'].value_counts())
print(train_data[train_data.label == 0]['label'].value_counts() / train_data.shape[0])
print(train_data[train_data.label == 1]['label'].value_counts() / train_data.shape[0])
# 数据重复情况
print(train_data.duplicated().sum())
print(test_data.duplicated().sum())
print(user_info.duplicated().sum())
print(user_log.duplicated().sum())
# 数据分布
# 数据的异常情况：
# 先看看数据的分布情况
# 定义异常值：正态分布、箱型图
# 找出异常值：数据选择和统计
