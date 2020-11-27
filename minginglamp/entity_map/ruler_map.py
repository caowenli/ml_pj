import json
import pandas as pd


class RulerMapper(object):
    def __init__(self, rule_file):
        self.rule_file = rule_file

    @staticmethod
    def parse_rule_file(rule_file, entity_list, relation_list,sentiment_list):
        data_df = pd.DataFrame()
        entity_relation_dict = {}
        for entity in entity_list:
            entity_relation_dict[entity] = []
        for relation in relation_list:
            entity_relation_dict[relation] = []

        with open(rule_file, 'r') as load_f:
            load_list = json.load(load_f)
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

    @staticmethod
    def df_match_category_name(df_data, single_col="品牌", category_name="category", rule_dict={"番茄酱": "Tomato Ketchup"}):
        df_data[category_name] = df_data[single_col].map(rule_dict)
        return df_data

    @staticmethod
    def df_match_category_label(df_data, single_col="品牌", category_label_name="category_label",
                                rule_dict={"番茄酱": "TK-TTL final"}):
        df_data[category_label_name] = df_data[single_col].map(rule_dict)

    @staticmethod
    def df_match_brand_name(df_data, single_col="品类&品牌", brand_name="brand",
                            rule_dict={"展艺&番茄酱": "ZOE HOME BAKING", "忆霖&番茄酱": "YILIN&番茄酱", "笑厨&番茄酱": "XIAOCHU",
                                       "凤球唛&番茄酱": "P&E ",
                                       "妙多&番茄酱": "Mida's", "梅林&番茄酱": "Meilin&番茄酱", "味好美&番茄酱": "McCormick",
                                       "每食富&番茄酱": "Masterfood",
                                       "李锦记&番茄酱": "LKK", "冠利&番茄酱": "Kühne&番茄酱", "汉斯&番茄酱": "Hunt's", "亨氏&番茄酱": "Heinz",
                                       "海天&番茄酱": "Haitian",
                                       "呱呱&番茄酱": "GUA GUA", "厨邦&番茄酱": "Chubang", "百利&番茄酱": "Berry"}):
        df_data[brand_name] = df_data[single_col].map(rule_dict)

    @staticmethod
    def df_match_brand_label(df_data, single_col="品类&品牌", brand_label="brand_label",
                             rule_dict={"展艺&番茄酱": "TK-BrN-ZOE HOME BAKING final", "忆霖&番茄酱": "TK-BrN-YILIN final",
                                        "笑厨&番茄酱": "TK-BrN-XIAOCHU final", "凤球唛&番茄酱": "TK-BrN-P&E final",
                                        "妙多&番茄酱": "TK-BrN-Mida's final", "梅林&番茄酱": "TK-BrN-Meilin final",
                                        "味好美&番茄酱": "TK-BrN-McCormick final", "每食富&番茄酱": "TK-BrN-Masterfood final",
                                        "李锦记&番茄酱": "TK-BrN-LKK final", "冠利&番茄酱": "TK-BrN-Kühne final",
                                        "汉斯&番茄酱": "TK-BrN-Hunt's final", "亨氏&番茄酱": "TK-BrN-Heinz final",
                                        "海天&番茄酱": "TK-BrN-Haitian final", "呱呱&番茄酱": "TK-BrN-GUA GUA final",
                                        "厨邦&番茄酱": "TK-BrN-Chubang final", "百利&番茄酱": "TK-BrN-Berry final"
                                        }):
        df_data[brand_label] = df_data[single_col].map(rule_dict)

    @staticmethod
    def df_category_angle_name(df_data, single_col="品类&品牌", brand_label="brand_label",
                               rule_dict={"展艺&番茄酱": "TK-BrN-ZOE HOME BAKING final", "忆霖&番茄酱": "TK-BrN-YILIN final",
                                          "笑厨&番茄酱": "TK-BrN-XIAOCHU final", "凤球唛&番茄酱": "TK-BrN-P&E final",
                                          "妙多&番茄酱": "TK-BrN-Mida's final", "梅林&番茄酱": "TK-BrN-Meilin final",
                                          "味好美&番茄酱": "TK-BrN-McCormick final", "每食富&番茄酱": "TK-BrN-Masterfood final",
                                          "李锦记&番茄酱": "TK-BrN-LKK final", "冠利&番茄酱": "TK-BrN-Kühne final",
                                          "汉斯&番茄酱": "TK-BrN-Hunt's final", "亨氏&番茄酱": "TK-BrN-Heinz final",
                                          "海天&番茄酱": "TK-BrN-Haitian final", "呱呱&番茄酱": "TK-BrN-GUA GUA final",
                                          "厨邦&番茄酱": "TK-BrN-Chubang final", "百利&番茄酱": "TK-BrN-Berry final"
                                          }):

    def paser_output(self, output_file):
        return

    def tk_social_logits(self, output):
        std_data = self.paser_output(output_file=xfsafas)
        rule = self.paser_output(self.rule_file)

        res = self.match_category(std_data, rule, category=xxxx)


mapper = RulerMapper(xxx)
mapper.tk_social_logits()
