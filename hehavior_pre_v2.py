#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import matplotlib.pyplot as plt
import types
import numpy as np
import math

data_path = './data/'
res_path = './res/'

user_behavior_info = pd.read_csv(data_path + "user_behavior_info.csv", names=['uid', 'bootTimes', 'AFuncTimes',
                                                                              'BFuncTimes', 'CFuncTimes', 'DFuncTimes',
                                                                              'EFuncTimes', 'FFuncTimes', 'FFuncSum'])


# 数据删除
user_behavior_info.drop(['FFuncTimes', 'AFuncTimes', 'BFuncTimes', 'EFuncTimes'], axis=1, inplace=True)
# 数据补齐
df1 = user_behavior_info[user_behavior_info['bootTimes'] != 0]
# df2 = user_behavior_info[user_behavior_info['AFuncTimes'] != 0]
# df3 = user_behavior_info[user_behavior_info['BFuncTimes'] != 0]
df4 = user_behavior_info[user_behavior_info['CFuncTimes'] != 0]
df5 = user_behavior_info[user_behavior_info['DFuncTimes'] != 0]
# df6 = user_behavior_info[user_behavior_info['EFuncTimes'] != 0]
df7 = user_behavior_info[user_behavior_info['FFuncSum'] != 0]

bootTimes_mean = int(math.ceil(df1.bootTimes.mean()))
# AFuncTimes_mean = int(math.ceil(df2.AFuncTimes.mean()))
# BFuncTimes_mean = int(math.ceil(df3.BFuncTimes.mean()))
CFuncTimes_mean = int(math.ceil(df4.CFuncTimes.mean()))
DFuncTimes_mean = int(math.ceil(df5.DFuncTimes.mean()))
# EFuncTimes_mean = int(math.ceil(df6.EFuncTimes.mean()))
GFuncTimes_mean = int(math.ceil(df7.FFuncSum.mean()))

print(type(user_behavior_info.bootTimes[0]), type(bootTimes_mean))

user_behavior_info.loc[user_behavior_info['bootTimes'] == 0, 'bootTimes'] = bootTimes_mean
# user_behavior_info.loc[user_behavior_info['AFuncTimes'] == 0, 'AFuncTimes'] = AFuncTimes_mean
# user_behavior_info.loc[user_behavior_info['BFuncTimes'] == 0, 'BFuncTimes'] = BFuncTimes_mean
user_behavior_info.loc[user_behavior_info['CFuncTimes'] == 0, 'CFuncTimes'] = CFuncTimes_mean
user_behavior_info.loc[user_behavior_info['DFuncTimes'] == 0, 'DFuncTimes'] = DFuncTimes_mean
# user_behavior_info.loc[user_behavior_info['EFuncTimes'] == 0, 'EFuncTimes'] = EFuncTimes_mean
user_behavior_info.loc[user_behavior_info['FFuncSum'] == 0, 'FFuncSum'] = GFuncTimes_mean

print(type(df1.uid[0]), type(user_behavior_info.uid[0]))

user_behavior_info.to_csv('behavior_process.csv', index=0, header=0)

# 数据分组


def group(strs, thre):
    # print(user_behavior_info.__getitem__(strs).value_counts())
    user_behavior_info = pd.read_csv("behavior_process.csv", names=['uid', 'bootTimes', 'CFuncTimes', 'DFuncTimes',
                                                                    'FFuncSum'])
    dup = user_behavior_info.drop_duplicates(strs)[strs].values
    cnt = dup.size
    group = [0] * cnt
    count = []
    print(cnt)
    for i in range(cnt):
        print(i)
        count.append(user_behavior_info.loc[user_behavior_info.__getitem__(strs) == dup[i]].__getitem__(strs).count())

    now_df = pd.DataFrame()
    now_df[strs] = dup
    now_df['prod_cnt'] = count
    now_df['group'] = group
    # print(user_behavior_info.prodName.count())
    cnt1 = cnt - now_df.loc[(now_df.prod_cnt < thre), strs].count()
    print(cnt1, cnt - cnt1)
    # input()
    now_df.loc[now_df.prod_cnt < thre, 'group'] = cnt1
    num = 0
    for index, row in now_df.iterrows():
        if row['group'] == 0:
            now_df.loc[index, 'group'] = num
            num += 1

    now_df.drop(['prod_cnt'], axis=1, inplace=True)
    # print(now_df)
    now_df = user_behavior_info.merge(now_df, how='left', on=strs)
    now_df[strs] = now_df.group
    now_df.drop(['group'], axis=1, inplace=True)
    print(now_df[strs])
    # input()
    # print(new_df.shape, user_behavior_info.shape)
    now_df.to_csv('behavior_process.csv', index=0, header=0)
    print(strs + ' finished')


# group('city', 5000)

names = ['bootTimes', 'CFuncTimes', 'DFuncTimes', 'FFuncSum']
for row in names:
    group(row, 20000)



