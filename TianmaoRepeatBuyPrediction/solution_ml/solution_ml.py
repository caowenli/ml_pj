import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import gc
from collections import Counter
import copy
import xgboost as xgb
import lightgbm as lgb

# from sklearn.model_selection import

plt.rcParams["font.sans-serif"] = "SimHei"  # 解决中文乱码问题
import seaborn as sns
import random

pd.options.display.max_columns = 5000
pd.options.display.max_rows = 5000
# 数据导入
train_data_path = r"/Users/caowenli/Desktop/ml_pj/ml/TianmaoRepeatBuyPrediction/data/data_format1/train_format1.csv"
test_data_path = r"/Users/caowenli/Desktop/ml_pj/ml/TianmaoRepeatBuyPrediction/data/data_format1/test_format1.csv"
user_info_path = r"/Users/caowenli/Desktop/ml_pj/ml/TianmaoRepeatBuyPrediction/data/data_format1/user_info_format1.csv"
user_log_path = r"/Users/caowenli/Desktop/ml_pj/ml/TianmaoRepeatBuyPrediction/data/data_format1/user_log_format1.csv"
train_data = pd.read_csv(train_data_path, sep=',')
test_data = pd.read_csv(test_data_path, sep=',')
user_info = pd.read_csv(user_info_path, sep=',')
user_log = pd.read_csv(user_log_path, sep=',')
print(test_data.shape)
print(train_data.shape)
print(user_log.shape)
print(user_info.shape)


# 数据探索性分析
# 数据预处理

def reduce_mem_usage(df, verbose=True):
    start_mem = df.memory_usage().sum() / 1024 ** 2
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(
                        np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(
                        np.int64).max:
                    df[col] = df[col].astype(np.int64)
                else:
                    if c_min > np.finfo(np.float16).min and c_max < np.finfo(
                            np.float16).max:
                        df[col] = df[col].astype(np.float16)
                    elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                        df[col] = df[col].astype(np.float32)
                    else:
                        df[col] = df[col].astype(np.float64)
                        end_mem = df.memory_usage().sum() / 1024 ** 2
                        print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))


reduce_mem_usage(test_data)
reduce_mem_usage(train_data)
reduce_mem_usage(user_log)
reduce_mem_usage(user_info)
print(train_data.shape)
print(test_data.shape)
# 合并用户的数据
train_data = train_data.merge(user_info, on=['user_id'], how='left')
test_data = test_data.merge(user_info, on=['user_id'], how='left')
del user_info
print(train_data.shape)
print(test_data.shape)
gc.collect()

# 用户日志数据按时间排序
user_log = user_log.sort_values(['user_id', 'time_stamp'])
# 合并用户日志数据各字段，新字段名为item_id
list_join_func = lambda x: " ".join([str(i) for i in x])
agg_dict = {
    'item_id': list_join_func,
    'cat_id': list_join_func,
    'seller_id': list_join_func,
    'brand_id': list_join_func,
    'time_stamp': list_join_func,
    'action_type': list_join_func
}
rename_dict = {
    'item_id': 'item_path',
    'cat_id': 'cat_path',
    'seller_id': 'seller_path',
    'brand_id': 'brand_path',
    'time_stamp': 'time_stamp_path',
    'action_type': 'action_type_path'
}


def merge_list(df_ID, join_columns, df_data, agg_dict, rename_dict):
    # 根据行进行合并
    df_data = df_data.groupby(join_columns).agg(agg_dict).reset_index().rename(
        columns=rename_dict)
    df_ID = df_ID.merge(df_data, on=join_columns, how="left")
    return df_ID


train_data = merge_list(train_data, 'user_id', user_log, agg_dict, rename_dict)
test_data = merge_list(test_data, 'user_id', user_log, agg_dict, rename_dict)
del user_log
gc.collect()
print(train_data.shape)
print(test_data.shape)


# 特征工程

def cnt_(x):
    try:
        return len(x.split(' '))
    except:
        return -1


def nunique_(x):
    try:
        return len(set(x.split(' ')))
    except:
        return -1


def max_(x):
    try:
        return np.max([float(i) for i in x.split(' ')])
    except:
        return -1


def min_(x):
    try:
        return np.min([float(i) for i in x.split(' ')])
    except:
        return -1


def std_(x):
    try:
        return np.std([float(i) for i in x.split(' ')])
    except:
        return -1


def most_n(x, n):
    try:
        return Counter(x.split(' ')).most_common(n)[n - 1][0]
    except:
        return -1


def most_n_cnt(x, n):
    try:
        return Counter(x.split(' ')).most_common(n)[n - 1][1]
    except:
        return -1


def user_cnt(df_data, single_col, name):
    df_data[name] = df_data[single_col].apply(cnt_)
    return df_data


def user_nunique(df_data, single_col, name):
    df_data[name] = df_data[single_col].apply(nunique_)
    return df_data


def user_max(df_data, single_col, name):
    df_data[name] = df_data[single_col].apply(max_)
    return df_data


def user_min(df_data, single_col, name):
    df_data[name] = df_data[single_col].apply(min_)
    return df_data


def user_std(df_data, single_col, name):
    df_data[name] = df_data[single_col].apply(std_)
    return df_data


def user_most_n(df_data, single_col, name, n=1):
    func = lambda x: most_n(x, n)
    df_data[name] = df_data[single_col].apply(func)
    return df_data


def user_most_n_cnt(df_data, single_col, name, n=1):
    func = lambda x: most_n_cnt(x, n)
    df_data[name] = df_data[single_col].apply(func)
    return df_data


# 取2000条数据举例
train_data_part = train_data.head(2000)
# 总次数
train_data_part = user_cnt(train_data_part, 'seller_path', 'user_cnt')
# 不同店铺个数
train_data_part = user_nunique(train_data_part, 'seller_path', 'seller_nunique ')
# 不同品类个数
train_data_part = user_nunique(train_data_part, 'cat_path', 'cat_nunique')
# 不同品牌个数
train_data_part = user_nunique(train_data_part, 'brand_path',
                               'brand_nunique')
# 不同商品个数
train_data_part = user_nunique(train_data_part, 'item_path', 'item_nunique')
# 活跃天数
train_data_part = user_nunique(train_data_part, 'time_stamp_path',
                               'time_stamp_nunique')
# 不同用户行为种数
train_data_part = user_nunique(train_data_part, 'action_type_path',
                               'action_type_nunique')
# 用户最喜欢的店铺
train_data_part = user_most_n(train_data_part, 'seller_path', 'seller_most_1', n=1)
# 最喜欢的类目
train_data_part = user_most_n(train_data_part, 'cat_path', 'cat_most_1', n=1)
# 最喜欢的品牌
train_data_part = user_most_n(train_data_part, 'brand_path', 'brand_most_1', n=1)
# 最常见的行为动作
train_data_part = user_most_n(train_data_part, 'action_type_path', 'action_type _1', n=1)

# 取2000条数据举例
test_data_part = test_data.head(2000)
# 总次数
test_data_part = user_cnt(test_data_part, 'seller_path', 'user_cnt')
# 不同店铺个数
test_data_part = user_nunique(test_data_part, 'seller_path', 'seller_nunique ')
# 不同品类个数
test_data_part = user_nunique(test_data_part, 'cat_path', 'cat_nunique')
# 不同品牌个数
test_data_part = user_nunique(test_data_part, 'brand_path',
                              'brand_nunique')
# 不同商品个数
test_data_part = user_nunique(test_data_part, 'item_path', 'item_nunique')
# 活跃天数
test_data_part = user_nunique(test_data_part, 'time_stamp_path',
                              'time_stamp_nunique')
# 不同用户行为种数
test_data_part = user_nunique(test_data_part, 'action_type_path',
                              'action_type_nunique')
# 用户最喜欢的店铺
test_data_part = user_most_n(test_data_part, 'seller_path', 'seller_most_1', n=1)
# 最喜欢的类目
test_data_part = user_most_n(test_data_part, 'cat_path', 'cat_most_1', n=1)
# 最喜欢的品牌
test_data_part = user_most_n(test_data_part, 'brand_path', 'brand_most_1', n=1)
# 最常见的行为动作
test_data_part = user_most_n(test_data_part, 'action_type_path', 'action_type _1', n=1)

print(train_data_part.shape)
print(test_data_part.shape)
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from scipy import sparse

tfidfVec = TfidfVectorizer(stop_words=ENGLISH_STOP_WORDS,
                           ngram_range=(1, 1),
                           max_features=100)
columns_list = ['seller_path']
for i, col in enumerate(columns_list):
    tfidfVec.fit(train_data_part[col])
    data_ = tfidfVec.transform(train_data_part[col])
    if i == 0:
        data_cat = data_
    else:
        data_cat = sparse.hstack((data_cat, data_))

import gensim

model = gensim.models.Word2Vec(
    train_data_part['seller_path'].apply(lambda x: x.split(' ')),
    size=100,
    window=5,
    min_count=5,
    workers=4)


def mean_w2v_(x, model, size=100):
    try:
        i = 0
        for word in x.split(' '):
            if word in model.wv.vocab:
                i += 1
            if i == 1:
                vec = np.zeros(size)
                vec += model.wv[word]
        return vec / i
    except:
        return np.zeros(size)


def get_mean_w2v(df_data, columns, model, size):
    data_array = []
    for index, row in df_data.iterrows():
        w2v = mean_w2v_(row[columns], model, size)
        data_array.append(w2v)
    return pd.DataFrame(data_array)


df_embeeding = get_mean_w2v(train_data_part, 'seller_path', model, 100)
df_embeeding.columns = ['embeeding_' + str(i) for i in df_embeeding.columns]

train_data_part = pd.concat([train_data_part, df_embeeding], axis=1)
print(train_data_part.shape)
print(train_data_part.keys())

model = gensim.models.Word2Vec(
    test_data_part['seller_path'].apply(lambda x: x.split(' ')),
    size=100,
    window=5,
    min_count=5,
    workers=4)


def mean_w2v_(x, model, size=100):
    try:
        i = 0
        for word in x.split(' '):
            if word in model.wv.vocab:
                i += 1
            if i == 1:
                vec = np.zeros(size)
                vec += model.wv[word]
        return vec / i
    except:
        return np.zeros(size)


def get_mean_w2v(df_data, columns, model, size):
    data_array = []
    for index, row in df_data.iterrows():
        w2v = mean_w2v_(row[columns], model, size)
        data_array.append(w2v)
    return pd.DataFrame(data_array)


df_embeeding = get_mean_w2v(test_data_part, 'seller_path', model, 100)
df_embeeding.columns = ['embeeding_' + str(i) for i in df_embeeding.columns]

test_data_part = pd.concat([test_data_part, df_embeeding], axis=1)
print(test_data_part.shape)
print(test_data_part.keys())
train_data_part=train_data_part.drop(['action_type_path',])


train_X, train_y = train_data_part.drop(['label'], axis=1), train_data_part['label']
from sklearn.model_selection import train_test_split

X_train, X_valid, y_train, y_valid = train_test_split(train_X, train_y, test_size=.3)
# 模型训练
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X_train, y_train)
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score

y_train_predict = model.predict(X_train)
y_valid_predict = model.predict(X_valid)
print(roc_auc_score(y_train, y_train_predict))
print(roc_auc_score(y_valid, y_valid_predict))
res = model.predict(test_data_part)
test_data_part['res'] = res
test_data_part.to_csv("res.csv", index=False)
# # import xgboost as xgb
# # import pandas as pd
# # from sklearn.model_selection import GridSearchCV
# # parameters = {
# #               'max_depth': [5,6,7,9,10,11,12,13],
# #               'learning_rate': [0.01, 0.02, 0.05, 0.1, 0.15],
# #               'n_estimators': [300, 500, 700,900],
# #               'subsample': [0.6, 0.7, 0.8],
# #
# # }
# #
# # xlf = xgb.XGBClassifier(booster='gbtree', objective='binary:logistic', eval_metric='auc', gamma=0.1,
# #                           max_depth=8, reg_alpha=0.2, reg_lambda=0.2, subsample=0.6,
# #                           colsample_bytree=0.3,
# #                           learning_rate=0.03,n_estimators=500)
# #
# # # 有了gridsearch我们便不需要fit函数
# # gsearch = GridSearchCV(xlf, param_grid=parameters, scoring='accuracy', cv=3)
# # gsearch.fit(X_train, y_train)
# #
# # print("Best score: %0.3f" % gsearch.best_score_)
# # print("Best parameters set:")
# # best_parameters = gsearch.best_estimator_.get_params()
# # for param_name in sorted(parameters.keys()):
# #     print("\t%s: %r" % (param_name, best_parameters[param_name]))
# import xgboost as xgb
#
# # XGBoost训练预测得分
# model = xgb.XGBClassifier(booster='gbtree', objective='binary:logistic', eval_metric='auc', gamma=0.1,
#                           max_depth=6, reg_alpha=0.2, reg_lambda=0.2, subsample=0.6,
#                           colsample_bytree=0.3,
#                           learning_rate=0.05,
#                           alpha=10, n_estimators=600)
# from sklearn.metrics import roc_auc_score
# from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score
#
# model.fit(X_train, y_train)
# y_train_predict = model.predict(X_train)
# y_valid_predict = model.predict(X_valid)
# print(roc_auc_score(y_train, y_train_predict))
# print(roc_auc_score(y_valid, y_valid_predict))
# print(model.feature_importances_)
#
# # features = ['link_pair_speed_plr', 'link_pair_speed_sax', 'link_pair_speed_dtw',
# #             'link_pair_speed_tlcc', 'link_pair_speed_cityblock',
# #             'link_pair_speed_euclidean', 'link_pair_speed_cosine',
# #             'link_pair_speed_pearsonr', 'link_pair_speed_jaccard',
# #             'link_pair_speed_mean', 'link_pair_speed_max', 'link_pair_speed_min',
# #             'link_pair_road_condition_plr', 'link_pair_road_condition_sax',
# #             'link_pair_road_condition_dtw', 'link_pair_road_condition_tlcc',
# #             'link_pair_road_condition_cityblock',
# #             'link_pair_road_condition_euclidean', 'link_pair_road_condition_cosine',
# #             'link_pair_road_condition_pearsonr', 'link_pair_road_condition_jaccard',
# #             'link_pair_road_condition_mean', 'link_pair_road_condition_max',
# #             'link_pair_road_condition_min']
# # result = permutation_importance(model, X_train, y_train, n_repeats=10, random_state=42)
# # perm_sorted_idx = result.importances_mean.argsort()
# # tree_importance_sorted_idx = np.argsort(model.feature_importances_)
# # print(tree_importance_sorted_idx)
# # print(perm_sorted_idx)
# # tree_indices = np.arange(0, len(model.feature_importances_)) + 0.5
# # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))
# # ax1.barh(tree_indices, model.feature_importances_[tree_importance_sorted_idx], height=0.7)
# # ax1.set_yticklabels(np.array(features)[tree_importance_sorted_idx])
# # ax1.set_yticks(tree_indices)
# # ax1.set_ylim((0, len(model.feature_importances_)))
# # ax2.boxplot(result.importances[perm_sorted_idx].T, vert=False, labels=np.array(features)[perm_sorted_idx])
# # fig.tight_layout()
# # plt.show()
#
# res = model.predict(data.drop(
#     columns=['s_link_dir', 's_next_link_dir', 's_is_fork_road', 's_is_signal_light', 's_next_link_dir_speed',
#              's_link_dir_speed', 's_link_dir_road_condition', 's_next_link_dir_road_condition',
#              'link_pair_speed_graph_euclidean', 'link_pair_speed_graph_cosine', 'label']))
# data['label'].value_counts()
# data['xgb_predict'] = res
# data['xgb_predict'].value_counts()
# data = data[['s_link_dir', 's_next_link_dir', 's_is_fork_road', 's_is_signal_light',
#              's_link_dir_speed', 's_next_link_dir_speed',
#              's_link_dir_road_condition', 's_next_link_dir_road_condition', 'xgb_predict']]
# data.to_csv('/Users/caowenli/Desktop/fudan_pj/data/res.csv', index=False)
# # test_data_res = model.predict(test_data.drop(
# #     columns=['s_link_dir', 's_next_link_dir', 's_is_fork_road', 's_is_signal_light', 's_next_link_dir_speed',
# #              's_link_dir_speed', 's_link_dir_road_condition', 's_next_link_dir_road_condition',
# #              'link_pair_speed_graph_euclidean',
# #              'link_pair_speed_graph_cosine', 'label']))
# # test_data['predict'] = test_data_res
# # valid = pd.read_csv('/Users/caowenli/Desktop/fudan_pj/data/gen_data_500_labeled.csv')
# # valid = valid[['s_link_dir', 's_next_link_dir', 's_is_fork_road', 's_is_signal_light', 'TRUE']]
# # data['xgb_predict'] = data['predict']
# # data_tmp = data[['s_link_dir', 's_next_link_dir', 'xgb_predict']]
# # valid_tmp = pd.merge(valid, data_tmp, how='left', on=['s_link_dir', 's_next_link_dir'])
# # valid_tmp['xgb_predict'].value_counts()
# # valid_tmp['TRUE'].value_counts()
# # from sklearn.metrics import recall_score, precision_score, confusion_matrix
# #
# # true = valid_tmp['TRUE'].tolist()
# # predict = valid_tmp['xgb_predict'].tolist()
# # print(confusion_matrix(true, predict))
# # from sklearn.metrics import recall_score, precision_score, accuracy_score, auc
# # print(accuracy_score(true, predict))
# # print(f1_score(true, predict))
# # print(roc_auc_score(true, predict))
# # print(recall_score(true, predict))
# # print(precision_score(true, predict))
