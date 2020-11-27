import os
import datetime
import gc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
from titanic.titanic_pytorch.data_helper import preprocessing
from titanic.titanic_pytorch.model import Model
from sklearn.model_selection import train_test_split
from torchkeras import summary
import argparse
import json

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
    train_data_path = config['train_data_path']
    y_name = config['y_name']
    test_size = config['test_size']
    batch_size = config['batch_size']
    n_feature = config['n_feature']
    n_hidden = config['n_hidden']
    n_output = config['n_output']
    lr = config['lr']
    epochs = config['epochs']
    log_step_freq = config['log_step_freq']
    save_model_path = config['save_model_path']
    nowtime = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print("\n" + "===========" * 8 + "%s" % nowtime)

    # 数据导入
    dftrain_raw = pd.read_csv(train_data_path)
    print(dftrain_raw.shape)
    x_train = preprocessing(dftrain_raw).values
    y_train = dftrain_raw[[y_name]].values
    # 数据分割
    x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=test_size,
                                                          stratify=y_train,  # 按照标签来分层采样
                                                          shuffle=True,  # 是否先打乱数据的顺序再划分
                                                          random_state=1
                                                          )
    print("x_train.shape =", x_train.shape)
    print("y_train.shape =", y_train.shape)
    print("x_valid.shape =", x_valid.shape)
    print("y_valid.shape=", y_valid.shape)

    # del dftrain_raw, dftest_raw
    # gc.collect()
    # 使用DataLoader和TensorDataset封装成可以迭代的数据管道
    dl_train = DataLoader(TensorDataset(torch.tensor(x_train).float(), torch.tensor(y_train).float()),
                          shuffle=True, batch_size=batch_size)
    dl_valid = DataLoader(TensorDataset(torch.tensor(x_valid).float(), torch.tensor(y_valid).float()),
                          shuffle=False, batch_size=batch_size)
    # 测试数据管道
    for features, labels in dl_train:
        print(features, labels)
        break
    for features, labels in dl_valid:
        print(features, labels)
        break

    net = Model(n_feature=n_feature, n_hidden=n_hidden, n_output=n_output)
    print(net)

    summary(net, input_shape=(n_feature,))

    # 训练模型
    # 1、定义损失函数、优化函数、评估指标
    from sklearn.metrics import accuracy_score

    loss_func = nn.BCELoss()
    optimizer = torch.optim.Adam(params=net.parameters(), lr=lr)
    metric_func = lambda y_pred, y_true: accuracy_score(y_true.data.numpy(), y_pred.data.numpy() > 0.5)
    metric_name = "accuracy"
    # 2、训练模型

    dfhistory = pd.DataFrame(columns=["epoch", "loss", metric_name, "val_loss", "val_" + metric_name])
    print("Start Training...")
    nowtime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print("==========" * 8 + "%s" % nowtime)

    for epoch in range(1, epochs + 1):

        # 1，训练循环-------------------------------------------------
        net.train()
        loss_sum = 0.0
        metric_sum = 0.0
        step = 1

        for step, (features, labels) in enumerate(dl_train, 1):

            # 梯度清零
            optimizer.zero_grad()

            # 正向传播求损失
            predictions = net(features)
            loss = loss_func(predictions, labels)
            metric = metric_func(predictions, labels)

            # 反向传播求梯度
            loss.backward()
            optimizer.step()

            # 打印batch级别日志
            loss_sum += loss.item()
            metric_sum += metric.item()
            if step % log_step_freq == 0:
                print(("[step = %d] loss: %.3f, " + metric_name + ": %.3f") %
                      (step, loss_sum / step, metric_sum / step))

        # 2，验证循环-------------------------------------------------
        net.eval()
        val_loss_sum = 0.0
        val_metric_sum = 0.0
        val_step = 1

        for val_step, (features, labels) in enumerate(dl_valid, 1):
            # 关闭梯度计算
            with torch.no_grad():
                predictions = net(features)
                val_loss = loss_func(predictions, labels)
                val_metric = metric_func(predictions, labels)
            val_loss_sum += val_loss.item()
            val_metric_sum += val_metric.item()

        # 3，记录日志-------------------------------------------------
        info = (epoch, loss_sum / step, metric_sum / step,
                val_loss_sum / val_step, val_metric_sum / val_step)
        dfhistory.loc[epoch - 1] = info

        # 打印epoch级别日志
        print((
                      "\nEPOCH = %d, loss = %.3f," + metric_name + "  = %.3f, val_loss = %.3f, " + "val_" + metric_name + " = %.3f") % info)
        nowtime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print("\n" + "==========" * 8 + "%s" % nowtime)

    print('Finished Training...')


    def plot_metric(dfhistory, metric):
        train_metrics = dfhistory[metric]
        val_metrics = dfhistory['val_' + metric]
        epochs = range(1, len(train_metrics) + 1)
        plt.plot(epochs, train_metrics, 'bo--')
        plt.plot(epochs, val_metrics, 'ro-')
        plt.title('Training and validation ' + metric)
        plt.xlabel("Epochs")
        plt.ylabel(metric)
        plt.legend(["train_" + metric, 'val_' + metric])
        plt.show()


    plot_metric(dfhistory, "loss")
    plot_metric(dfhistory, "accuracy")
    # 保存模型
    print(net.state_dict().keys())
    torch.save(net.state_dict(), save_model_path + '_' + str(nowtime) + ".pkl")
