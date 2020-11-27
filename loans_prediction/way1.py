import pandas as pd
import numpy as np
import xgboost as xgb
import os
import datetime
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import lightgbm as lgb

sns.set_style("dark")
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)


def read_data(filename):
    df = pd.read_csv(filename)
    print("{}读取数据{}".format(now(), filename))
    print("{}查看数据{}行，{}列".format(now(), df.shape[0], df.shape[1]))
    return df


def now():
    tmp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return tmp


# 查看数据的缺失量和数据的类型
def checking_na(df):
    try:
        if (isinstance(df, pd.DataFrame)):
            df_na_bool = pd.concat([df.isnull().any(), df.isnull().sum(), (df.isnull().sum() / df.shape[0]) * 100],
                                   axis=1, keys=['df_bool', 'df_amt', 'missing_ratio_percent'])
            df_na_bool = df_na_bool.loc[df_na_bool['df_bool'] == True]
            return df_na_bool
        else:
            print("{}: The input is not panda DataFrame".format(now()))
    except (UnboundLocalError, RuntimeError):
        print("{}: Something is wrong".format(now()))


loan_data = read_data("/Users/caowenli/PycharmProjects/ml/loans_prediction/Loan payments data.csv")
print("\n\n")
print(checking_na(loan_data))
print(loan_data.head(5))
print(loan_data.keys())
print(loan_data.dtypes)

features = ['Loan_ID', 'loan_status', 'Principal', 'terms', 'effective_date',
            'due_date', 'paid_off_time', 'past_due_days', 'age', 'education',
            'Gender']
print(loan_data.loan_status.unique())

fig = plt.figure(figsize=(5, 5))
ax = sns.countplot(loan_data.loan_status)
ax.set_title("Count of Loan Status")
for p in ax.patches:
    ax.annotate(str(format(int(p.get_height()), ',d')), (p.get_x(), p.get_height() * 1.01))
plt.show()
# 时间数据处理
loan_data['paid_off_date'] = pd.DatetimeIndex(loan_data.paid_off_time).normalize()
loan_data['day_to_pay'] = (pd.DatetimeIndex(loan_data.paid_off_time).normalize() - pd.DatetimeIndex(
    loan_data.effective_date).normalize()) / np.timedelta64(1, 'D')
# 类型数据离散化
status_map = {"PAIDOFF": 1, "COLLECTION": 2, "COLLECTION_PAIDOFF": 2}
loan_data['loan_status_trgt'] = loan_data['loan_status'].map(status_map)
dummies = pd.get_dummies(loan_data['education']).rename(columns=lambda x: 'is_' + str(x))
loan_data = pd.concat([loan_data, dummies], axis=1)
loan_data = loan_data.drop(['education'], axis=1)
dummies = pd.get_dummies(loan_data['Gender']).rename(columns=lambda x: 'is_' + str(x))
loan_data = pd.concat([loan_data, dummies], axis=1)
loan_data = loan_data.drop(['Gender'], axis=1)
loan_data = loan_data.drop(
    ['Loan_ID', 'loan_status', 'effective_date', 'due_date', 'paid_off_time', 'past_due_days', 'paid_off_date',
     'day_to_pay'], axis=1)
dummy_var = ['is_female', 'is_Master or Above']
loan_data = loan_data.drop(dummy_var, axis=1)
print(loan_data.head(5))
x = loan_data.drop(['loan_status_trgt'], axis=1)
y = loan_data.loan_status_trgt
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=7)
# models = xgb.XGBClassifier()
# models.fit(X_train, y_train)
# y_pred = models.predict(X_test)
# predictions = [round(value) for value in y_pred]
# accuracy = accuracy_score(y_test, predictions)
# print("Accuracy: %.2f%%" % (accuracy * 100.0))
model = lgb.LGBMClassifier()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
predictions = [round(value) for value in y_pred]
accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))
