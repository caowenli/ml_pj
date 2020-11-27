import os
import datetime
import gc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torchvision
import warnings

nowtime = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
print("\n" + "===========" * 8 + "%s" % nowtime)

# 数据导入
train_data_path = r"/dataset/titanic/train.csv"
test_data_path = r"/dataset/titanic/test.csv"
dftrain_raw = pd.read_csv(train_data_path)
dftest_raw = pd.read_csv(test_data_path)
print(dftest_raw.shape)
print(dftrain_raw.shape)


# 特征工程
def preprocessing(dfdata):
    dfresult = pd.DataFrame()

    # Pclass
    dfPclass = pd.get_dummies(dfdata['Pclass'])
    dfPclass.columns = ['Pclass_' + str(x) for x in dfPclass.columns]
    dfresult = pd.concat([dfresult, dfPclass], axis=1)

    # Sex
    dfSex = pd.get_dummies(dfdata['Sex'])
    dfresult = pd.concat([dfresult, dfSex], axis=1)

    # Age
    dfresult['Age'] = dfdata['Age'].fillna(0)
    dfresult['Age_null'] = pd.isna(dfdata['Age']).astype('int32')

    # SibSp,Parch,Fare
    dfresult['SibSp'] = dfdata['SibSp']
    dfresult['Parch'] = dfdata['Parch']
    dfresult['Fare'] = dfdata['Fare']

    # Carbin
    dfresult['Cabin_null'] = pd.isna(dfdata['Cabin']).astype('int32')

    # Embarked
    dfEmbarked = pd.get_dummies(dfdata['Embarked'], dummy_na=True)
    dfEmbarked.columns = ['Embarked_' + str(x) for x in dfEmbarked.columns]
    dfresult = pd.concat([dfresult, dfEmbarked], axis=1)

    return (dfresult)


x_train = preprocessing(dftrain_raw).values
y_train = dftrain_raw[['Survived']].values
x_test = preprocessing(dftest_raw).values
from sklearn.model_selection import train_test_split

x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.2,
                                                      stratify=y_train,  # 按照标签来分层采样
                                                      shuffle=True,  # 是否先打乱数据的顺序再划分
                                                      random_state=1
                                                      )
print("x_train.shape =", x_train.shape)
print("y_train.shape =", y_train.shape)
print("x_valid.shape =", x_valid.shape)
print("y_valid.shape=", y_valid.shape)
print("x_test.shape=", x_test.shape)
# del dftrain_raw, dftest_raw
# gc.collect()
# 使用DataLoader和TensorDataset封装成可以迭代的数据管道
dl_train = DataLoader(TensorDataset(torch.tensor(x_train).float(), torch.tensor(y_train).float()),
                      shuffle=True, batch_size=8)
dl_valid = DataLoader(TensorDataset(torch.tensor(x_valid).float(), torch.tensor(y_valid).float()),
                      shuffle=False, batch_size=8)
# 测试数据管道
for features, labels in dl_train:
    print(features, labels)
    break
for features, labels in dl_valid:
    print(features, labels)
    break


# 定义BP神经网络
class Model(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(Model, self).__init__()
        self.hidden = torch.nn.Linear(n_feature, n_hidden)
        self.out = torch.nn.Linear(n_hidden, n_output)

    def forward(self, x):
        x = torch.relu(self.hidden(x))
        x = torch.sigmoid(self.out(x))
        return x


net = Model(n_feature=15, n_hidden=10, n_output=1)
print(net)
from torchkeras import summary

summary(net, input_shape=(15,))

# 训练模型
# 1、定义损失函数、优化函数、评估指标
from sklearn.metrics import accuracy_score

loss_func = nn.BCELoss()
optimizer = torch.optim.Adam(params=net.parameters(), lr=0.01)
metric_func = lambda y_pred, y_true: accuracy_score(y_true.data.numpy(), y_pred.data.numpy() > 0.5)
metric_name = "accuracy"
# 2、训练模型
epochs = 50
log_step_freq = 30
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

# 模型评估
import matplotlib.pyplot as plt


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
#
# # 使用模型
# # 预测概率
# y_pred_probs = net(torch.tensor(x_test).float()).data
# print("type   {},\nvalue  {}".format(type(y_pred_probs), y_pred_probs))
# value = y_pred_probs.numpy()
# print(value)
# print(type(x_test))
# dftest_raw['Survived'] = value
# print(dftest_raw.head(5))
# # 预测类别
# y_pred = torch.where(y_pred_probs > 0.5, torch.ones_like(y_pred_probs), torch.zeros_like(y_pred_probs))
# print("type{},value{}".format(type(y_pred), y_pred))
# print("type   {},\nvalue  {}".format(type(y_pred), y_pred))
# y_pred_np = y_pred.numpy()
# y_pred_np = y_pred_np.astype(np.int16)
# print(y_pred_np)
# print(type(x_test))
# dftest_raw['Survived'] = y_pred_np
# print(dftest_raw.head(20))

# 保存模型
print(net.state_dict().keys())
# 保存模型参数
torch.save(net.state_dict(), "/Users/caowenli/Desktop/ml_pj/ml/titanic/net_parameter.pkl")

# 加载模型的模型并上线预测
net_clone = Model(n_feature=15, n_hidden=10, n_output=1)
net_clone.load_state_dict(torch.load("/Users/caowenli/Desktop/ml_pj/ml/titanic/net_parameter.pkl"))
y_pred_probs = net_clone(torch.tensor(x_test).float()).data
y_pred = torch.where(y_pred_probs > 0.5, torch.ones_like(y_pred_probs), torch.zeros_like(y_pred_probs))
print("type{},value{}".format(type(y_pred), y_pred))
print("type   {},\nvalue  {}".format(type(y_pred), y_pred))
y_pred_np = y_pred.numpy()
y_pred_np = y_pred_np.astype(np.int16)
print(y_pred_np)
print(type(x_test))
dftest_raw['Survived'] = y_pred_np
print(dftest_raw.head(20))
