#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
import copy
import warnings
import gc
from sklearn import preprocessing
import xgboost as xgb

# In[2]:


pd.set_option('display.max_columns', None)
data_path = './data/'
result_path = './res/'
# In[3]:


# 读取数据
age_train = pd.read_csv(data_path + "age_train.csv", names=['uid', 'age_group'])
age_test = pd.read_csv(data_path + "age_test.csv", names=['uid'])
count = age_test.size
user_basic_info = pd.read_csv(data_path + "user_basic_info.csv", names=['uid', 'gender', 'city', 'prodName',
                                                                        'ramCapacity', 'ramLeftRation', 'romCapacity',
                                                                        'romLeftRation', 'color', 'fontSize', 'ct',
                                                                        'carrier', 'os'])
user_behavior_info = pd.read_csv(data_path + "user_behavior_info.csv", names=['uid', 'bootTimes', 'AFuncTimes',
                                                                              'BFuncTimes', 'CFuncTimes', 'DFuncTimes',
                                                                              'EFuncTimes', 'FFuncTimes', 'FFuncSum'])
user_app_actived = pd.read_csv(data_path + "user_app_actived.csv", names=['uid', 'appId'])
# user_app_usage = pd.read_csv("user_app_usage.csv")
app_info = pd.read_csv(data_path + "app_info.csv", names=['appId', 'category'])


# In[2]:


# 处理数据量较大的user_app_usage.csv，结合app_info.csv简单统计得到appuseProcessed.csv作为特征
def f(x):
    s = x.value_counts()
    return np.nan if len(s) == 0 else s.index[0]


def processUserAppUsage():
    print("processUserAppUsage start")
    resTable = pd.DataFrame()
    reader = pd.read_csv(data_path + "user_app_usage.csv", names=['uid', 'appId', 'duration', 'times', 'use_date'],
                         iterator=True)
    last_df = pd.DataFrame()
    
    app_info = pd.read_csv(data_path + "app_info.csv", names=['appId', 'category'])
    cats = list(set(app_info['category']))
    category2id = dict(zip(sorted(cats), range(0, len(cats))))
    id2category = dict(zip(range(0, len(cats)), sorted(cats)))
    app_info['category'] = app_info['category'].apply(lambda x: category2id[x])
    i = 1
    
    while True:
        try:
            print("index: {}".format(i))
            i += 1
            df = reader.get_chunk(1000000)
            df = pd.concat([last_df, df])
            idx = df.shape[0]-1
            last_user = df.iat[idx, 0]
            while df.iat[idx, 0] == last_user:
                idx -= 1
            last_df = df[idx+1:]
            df = df[:idx+1]

            now_df = pd.DataFrame()
            now_df['uid'] = df['uid'].unique()
            now_df = now_df.merge(df.groupby('uid')['appId'].count().to_frame(), how='left', on='uid')
            now_df = now_df.merge(df.groupby('uid')['appId', 'use_date'].agg(['nunique']), how='left', on='uid')
            now_df = now_df.merge(df.groupby('uid')['duration', 'times'].agg(['mean', 'max', 'std']), how='left',
                                  on='uid')

            now_df.columns = ['uid', 'usage_cnt', 'usage_appid_cnt', 'usage_date_cnt', 'duration_mean', 'duration_max',
                              'duration_std', 'times_mean', 'times_max', 'times_std']

            df = df.merge(app_info, how='left', on='appId')
            now_df = now_df.merge(df.groupby('uid')['category'].nunique().to_frame(), how='left', on='uid')
            # print(df.groupby(['uid'])['category'].value_counts().index[0])
            now_df['usage_most_used_category'] = df.groupby(['uid'])['category'].transform(f)
            resTable = pd.concat([resTable, now_df])
        except StopIteration:
            break
    
    resTable.to_csv("appuseProcessed.csv", index=0)
    
    print("Iterator is stopped")


# In[5]:


processUserAppUsage()


# In[4]:


# 将user_basic_info.csv 和 user_behavior_info.csv中的字符值编码成可以训练的数值类型，合并
class2id = {}
id2class = {}


def mergeBasicTables(baseTable):
    resTable = baseTable.merge(user_basic_info, how='left', on='uid', suffixes=('_base0', '_ubaf'))
    resTable = resTable.merge(user_behavior_info, how='left', on='uid', suffixes=('_base1', '_ubef'))

    cat_columns = ['city', 'prodName', 'color', 'os', 'fontSize', 'gender']
    resTable.drop(['ct', 'carrier'], axis=1, inplace=True)
    for c in cat_columns:
        resTable[c] = resTable[c].apply(lambda x: x if type(x) == str else str(x))
        sort_temp = sorted(list(set(resTable[c])))  
        class2id[c+'2id'] = dict(zip(sort_temp, range(1, len(sort_temp)+1)))
        id2class['id2'+c] = dict(zip(range(1, len(sort_temp)+1), sort_temp))
        resTable[c] = resTable[c].apply(lambda x: class2id[c+'2id'][x])
        
    return resTable


# In[5]:


# 处理app使用相关数据
# 对user_app_actived.csv简单统计
# 将之前训练的appuseProcess.csv进行合并
def mergeAppData(baseTable):
    resTable = baseTable.merge(user_app_actived, how='left', on='uid')
    resTable['appId'] = resTable['appId'].apply(lambda x: len(list(x.split('#'))))
    appusedTable = pd.read_csv("appuseProcessed.csv")
    resTable = resTable.merge(appusedTable, how='left', on='uid')
    resTable[['category', 'usage_most_used_category']] = resTable[['category', 'usage_most_used_category']].fillna(41)
    resTable = resTable.fillna(0)
    # print(resTable[:5])
    return resTable


# In[6]:


# 合并用户基本特征以及app使用相关特征，作为训练集和测试集
df_train = mergeAppData(mergeBasicTables(age_train))
df_test = mergeAppData(mergeBasicTables(age_test))
print(df_train.shape)
print(df_test.shape)


# In[7]:

# 训练模型

# In[8]:


print("训练模型：")
params = {
    'booster': 'gbtree',
    'objective': 'multi:softmax',
    'num_class': 6,
    'stratified': True,
    'max_depth': 4,  # 5  4
    'min_child_weight': 1,
    'gamma': 0,  # 3
    'subsample': 1,  # 0.8
    'colsample_bytree': 1,  # 0.6
    'lambda': 2,  # 3
    'eta': 0.05,  # 0.05
    'seed': 0,  # 20
    'silent': 1,
    'eval_metric': 'merror'
}

X = df_train.drop(['age_group', 'uid'], axis=1)
y = df_train['age_group']
uid = df_test['uid']
test = df_test.drop('uid', axis=1)

trainFeature = df_train.drop(['age_group', 'uid'], axis=1)
trainLabel = df_train.age_group - 1
testFeature = test


def xgbCV(trainFeature, trainLabel, params, rounds):
    print("cv")
    dtrain = xgb.DMatrix(trainFeature, label=trainLabel)
    # params['scale_pos_weights '] = (float)(len(trainLabel[trainLabel == 0]))/(float)(len(trainLabel[trainLabel == 1]))
    num_round = rounds
    print('run cv: ' + 'round: ' + str(rounds))
    res = xgb.cv(params, dtrain, num_round, verbose_eval=100, early_stopping_rounds=200, nfold=3)
    return res


def xgbPredict(trainFeature, trainLabel, testFeature, rounds, params):
    print("predict")
    # params['scale_pos_weights '] = (float)(len(trainLabel[trainLabel == 0])) / len(trainLabel[trainLabel == 1])

    dtrain = xgb.DMatrix(trainFeature.values, label=trainLabel)
    dtest = xgb.DMatrix(testFeature.values)

    watchlist = [(dtrain, 'train')]
    num_round = rounds

    model = xgb.train(params, dtrain, num_round, watchlist, verbose_eval=50)
    predict = model.predict(dtest)
    return model, predict + 1


cv_res = xgbCV(trainFeature, trainLabel, params, 1)

model, predict = xgbPredict(trainFeature, trainLabel, testFeature, 1, params)

cv_pred = []
y_test = predict

sub = pd.DataFrame()
sub['id'] = uid
sub['label'] = predict
sub.head()

sub.to_csv(result_path + 'submission.csv', index=False, header=0)

print("finish")





