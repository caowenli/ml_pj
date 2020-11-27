import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras import models, layers
import tensorflow as tf
from tensorflow.keras.layers import Embedding, Dense, Dropout, Bidirectional, LSTM

from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Embedding, Dense, Dropout, Bidirectional, LSTM

import datetime

nowtime = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
print("\n" + "===========" * 8 + "%s" % nowtime)

# 数据导入
train_data_path = r"/Users/caowenli/Desktop/ml_pj/ml/titanic/data/train.csv"
test_data_path = r"/Users/caowenli/Desktop/ml_pj/ml/titanic/data/test.csv"

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
# 任意方式定义网络
n_features = 15
n_hidden = 10
n_output = 1
tf.keras.backend.clear_session()
inputs = layers.Input(shape=(n_features,))
hidden1 = layers.Dense(n_hidden, activation='relu')(inputs)
outputs = layers.Dense(n_output, activation='sigmoid')(hidden1)
model = models.Model(inputs=inputs, outputs=outputs)
print(model.summary())
# 二分类问题选择二元交叉熵损失函数
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['acc'])

history = model.fit(x_train, y_train,
                    batch_size=64,
                    epochs=30,
                    validation_split=0.2  # 分割一部分训练数据用于验证
                    )
import matplotlib.pyplot as plt


def plot_metric(history, metric):
    train_metrics = history.history[metric]
    val_metrics = history.history['val_' + metric]
    epochs = range(1, len(train_metrics) + 1)
    plt.plot(epochs, train_metrics, 'bo--')
    plt.plot(epochs, val_metrics, 'ro-')
    plt.title('Training and validation ' + metric)
    plt.xlabel("Epochs")
    plt.ylabel(metric)
    plt.legend(["train_" + metric, 'val_' + metric])
    plt.show()


plot_metric(history, "loss")
plot_metric(history, "acc")
print(model.evaluate(x=x_valid, y=y_valid))
# 预测概率
model.predict(x_test[0:10])
# 预测类别
model.predict_classes(x_test[0:10])
# model(tf.constant(x_test[0:10].values,dtype = tf.float32)) #等价写法
# 保存权重，该方式仅仅保存权重张量
model.save_weights('tf_model_weights.ckpt', save_format="tf")
# 保存模型结构与模型参数到文件,该方式保存的模型具有跨平台性便于部署

model.save('tf_model_savedmodel', save_format="tf")
print('export saved model.')

model_loaded = tf.keras.models.load_model('tf_model_savedmodel')
model_loaded.evaluate(x_valid, y_valid)
