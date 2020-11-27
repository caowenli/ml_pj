import json
import pandas as pd
from collections import defaultdict

# import numpy as np
# import pandas as pd
#
# data = {'city': ['Beijing', 'Shanghai', 'Guangzhou', 'Shenzhen', 'Hangzhou', 'Chongqing'],
#         'year': [2016, 2016, 2015, 2017, 2016, 2016],
#         'population': [2100, 2300, 1000, 700, 500, 500]}
# frame = pd.DataFrame(data, columns=['year', 'city', 'population', 'debt'])
#
#
# def function(a, b):
#     if 'ing' in a and b == 2016:
#         return 1
#     else:
#         return 0
#
#
# print(frame, '\n')
# frame['test'] = frame.apply(lambda x: function(x.city, x.year), axis=1)
#
#
# 品类映射
data = pd.DataFrame()
data['a'] = data['a'].apply()


def match_category(rule_dict, category="番茄酱"):
    for k, v in rule_dict.items():
        if category == k:
            return v
    return None


def df_categroy_label(column_name, df):
    df[column_name] = df[column_name].apply(match_category(rule_dict={"番茄酱": "ZOE HOME BAKING"}, category="番茄酱"))
    return
# # 品牌名称映射
# def brand_map(category_entity, brand_entity):
#     brand_name_dict = {"展艺": "ZOE HOME BAKING",
#                        "忆霖": "YILIN",
#                        "笑厨": "XIAOCHU",
#                        "凤球唛": "P&E ",
#                        "妙多": "Mida's",
#                        "梅林": "Meilin",
#                        "味好美": "McCormick",
#                        "每食富": "Masterfood",
#                        "李锦记": "LKK",
#                        "冠利": "Kühne",
#                        "汉斯": "Hunt's",
#                        "亨氏": "Heinz",
#                        "海天": "Haitian",
#                        "呱呱": "GUA GUA",
#                        "厨邦": "Chubang",
#                        "百利": "Berry"}
#     if category_entity == "番茄酱":
#         for key in brand_name_dict.keys():
#             if brand_entity == key:
#                 return brand_name_dict[key]
#     return "undefined"
#
#
# # 品牌标签映射
# def brand_label_name_map(category_entity, brand_entity):
#     brand_label_name_dict = {"展艺": "TK-BrN-ZOE HOME BAKING final",
#                              "忆霖": "TK-BrN-YILIN final",
#                              "笑厨": "TK-BrN-XIAOCHU final",
#                              "凤球唛": "TK-BrN-P&E final",
#                              "妙多": "TK-BrN-Mida's final",
#                              "梅林": "TK-BrN-Meilin final",
#                              "味好美": "TK-BrN-McCormick final",
#                              "每食富": "TK-BrN-Masterfood final",
#                              "李锦记": "TK-BrN-LKK final",
#                              "冠利": "TK-BrN-Kühne final",
#                              "汉斯": "TK-BrN-Hunt's final",
#                              "亨氏": "TK-BrN-Heinz final",
#                              "海天": "TK-BrN-Haitian final",
#                              "呱呱": "TK-BrN-GUA GUA final",
#                              "厨邦": "TK-BrN-Chubang final",
#                              "百利": "TK-BrN-Berry final"}
#     if category_entity == "番茄酱":
#         for key in brand_label_name_dict.keys():
#             if brand_entity == key:
#                 return brand_label_name_dict[key]
#     return "undefined"
#
#
# # 品类的情感
# def category_sentimen_map(category_entity, sentiment_entity):
#     if category_entity == "番茄酱":
#         if sentiment_entity == "neg":
#             return "TK-CaS-Neg final"
#         if sentiment_entity == "pos":
#             return "TK-CaS-Pos final"
#
#
# # 品类的角度
# def category_angle_map(category_entity)
