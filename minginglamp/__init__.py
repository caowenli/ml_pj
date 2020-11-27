entity_list = ["品牌", "产品", "包装"]
relation_list = ["功效&品类", "品牌&产品", "功效&成分"]

import pandas as pd

first_list = [
    {
        "cleaned_content": "bbs,\"红姐发表于3楼您的帖子\"2019年伊始,全新一代东风风神AX7试驾体验\"由于内容丰富,已经设置成精华帖,记得继续努力写更多好的内容哦!谢谢红姐\",\"2019年伊始,全新一代东风风神AX7试驾体验\",",
        "data_publishedAt": "",
        "document_id": "98",
        "entity": [
            {
                "end_offset": 36,
                "mention": "东风风神AX7",
                "name": "东风风神AX7",
                "start_offset": 29,
                "type": "产品"
            },
            {
                "end_offset": 95,
                "mention": "东风风神",
                "name": "东风风神",
                "start_offset": 91,
                "type": "品牌"
            }
        ],
        "relation":
            [
                {
                    "object": {
                        "end_offset": 643,
                        "mention": "雅诗兰黛自然堂",
                        "name": "雅诗兰黛自然堂",
                        "start_offset": 636,
                        "type": "品牌"
                    },
                    "predicate": "IS_RELATED_TO",
                    "subject": {
                        "end_offset": 665,
                        "mention": "Thehistoryofwhoo",
                        "name": "Thehistoryofwhoo",
                        "start_offset": 649,
                        "type": "产品"
                    }
                },
                {
                    "object": {
                        "end_offset": 1007,
                        "mention": "美容",
                        "name": "美容",
                        "start_offset": 1005,
                        "type": "功效"
                    },
                    "predicate": "IS_RELATED_TO",
                    "subject": {
                        "end_offset": 940,
                        "mention": "土蜂蜜",
                        "name": "土蜂蜜",
                        "start_offset": 937,
                        "type": "成分"
                    }
                }
            ]
    },
    {
        "cleaned_content": "bbs,\"绿活泉的圣诞限量版第一眼就爱上,真的超级唯美啊,就算是为了瓶子也要入手,作为碧欧泉的真爱粉,我真的用了碧欧泉很多年了!\",\"今年份的圣诞礼物,让这个\"大叔\"来承包吧!\",",
        "data_publishedAt": "",
        "document_id": "99",
        "entity": [
            {
                "end_offset": 8,
                "mention": "绿活泉",
                "name": "绿活泉",
                "start_offset": 5,
                "type": "品牌"
            },
            {
                "end_offset": 46,
                "mention": "碧欧泉",
                "name": "碧欧泉",
                "start_offset": 43,
                "type": "品牌"
            },
            {
                "end_offset": 59,
                "mention": "碧欧泉",
                "name": "碧欧泉",
                "start_offset": 56,
                "type": "品牌"
            },
            {
                "end_offset": 36,
                "mention": "瓶子",
                "name": "瓶子",
                "start_offset": 34,
                "type": "包装"
            }
        ],
        "relation": [
            {
                "object": {
                    "end_offset": 164,
                    "mention": "放松",
                    "name": "放松",
                    "start_offset": 162,
                    "type": "功效"
                },
                "predicate": "IS_RELATED_TO",
                "subject": {
                    "end_offset": 12,
                    "mention": "乳液",
                    "name": "乳液",
                    "start_offset": 10,
                    "type": "品类"
                }
            },
            {
                "object": {
                    "end_offset": 172,
                    "mention": "强化身体的线条",
                    "name": "强化身体的线条",
                    "start_offset": 165,
                    "type": "功效"
                },
                "predicate": "IS_RELATED_TO",
                "subject": {
                    "end_offset": 12,
                    "mention": "乳液",
                    "name": "乳液",
                    "start_offset": 10,
                    "type": "品类"
                }
            }
        ]
    }
]


def parse_file_path(load_list, entity_list, relation_list):
    data_df = pd.DataFrame()
    entity_relation_dict = {}
    for entity in entity_list:
        entity_relation_dict[entity] = []
    for relation in relation_list:
        entity_relation_dict[relation] = []
    for key in entity_relation_dict.keys():
        for dict_first in load_list:
            value = []
            entity_list = dict_first['entity']
            relation_list = dict_first['relation']
            for dict_second in entity_list:
                if dict_second['type'] == key:
                    value.append(dict_second['name'])
            if len(value) == 0:
                for dict_second in relation_list:
                    object = dict_second['object']
                    subject = dict_second['subject']
                    object_type = object['type']
                    subject_type = subject['type']
                    entity_relation = object_type + "&" + subject_type
                    if entity_relation == key:
                        value.append(object['name'] + '&' + subject['name'])
            entity_relation_dict[key].append(' '.join(value))
    for k, v in entity_relation_dict.items():
        data_df[k] = v
    return data_df


def df_match_categroy(df_data, single_col, category_name, rule_dict={"番茄酱": "ZOE HOME BAKING"}):
    df_data[category_name] = df_data[single_col].map(rule_dict)
    return df_data


data = parse_file_path(first_list, entity_list, relation_list)
print(data.head(5))
data_new = df_match_categroy(data, single_col='品牌', category_name='category_name',
                             rule_dict={"东风风神": "ZOE HOME BAKING"},
                             )
print(data_new.head(5))
data.to_excel('result.xlsx')
