from titanic.titanic_pytorch.model import Model
from titanic.titanic_pytorch.data_helper import preprocessing
import torch
import pandas as pd
import numpy as np
import json
import datetime

if __name__ == "__main__":
    # # 读取用户在命令行输入的信息
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--config_path", help="config path of models")
    # args = parser.parse_args()
    #
    # with open(args.config_path, "r") as fr:
    #     config = json.load(fr)
    config_path = '/Users/caowenli/Desktop/ml_pj/ml/titanic/titanic_pytorch/config.json'
    with open(config_path, "r") as fr:
        config = json.load(fr)
    test_data_path = config['test_data_path']
    load_model_path = config['load_model_path']
    result_path = config['result_path']
    y_name = config['y_name']
    nowtime = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    dftest_raw = pd.read_csv(test_data_path)
    x_test = preprocessing(dftest_raw).values
    print(dftest_raw.shape)
    print("x_test.shape=", x_test.shape)
    # 加载模型的模型并上线预测
    net_clone = Model(n_feature=15, n_hidden=10, n_output=1)
    net_clone.load_state_dict(torch.load(load_model_path))
    y_pred_probs = net_clone(torch.tensor(x_test).float()).data
    y_pred = torch.where(y_pred_probs > 0.5, torch.ones_like(y_pred_probs), torch.zeros_like(y_pred_probs))
    print("type{},value{}".format(type(y_pred), y_pred))
    print("type   {},\nvalue  {}".format(type(y_pred), y_pred))
    y_pred_np = y_pred.numpy()
    y_pred_np = y_pred_np.astype(np.int16)
    print(y_pred_np)
    print(type(x_test))
    dftest_raw[y_name] = y_pred_np
    print(dftest_raw.head(20))
    dftest_raw.to_csv(result_path + '_' + str(nowtime) + '.csv', index=None)
