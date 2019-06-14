#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import matplotlib.pyplot as plt
import types
import numpy as np
import math

data_path = './data/'
res_path = './res/'


user_basic_info = pd.read_csv(data_path + "user_basic_info.csv", names=['uid', 'gender', 'city', 'prodName',
                                                                        'ramCapacity', 'ramLeftRation', 'romCapacity',
                                                                        'romLeftRation', 'color', 'fontSize', 'ct',
                                                                        'carrier', 'os'])
# 内容补齐
user_basic_info.loc[user_basic_info.city.isnull(), 'city'] = 'c0043'
#
mean1 = math.ceil(user_basic_info.ramCapacity.mean())
user_basic_info.loc[user_basic_info.ramCapacity.isnull(), 'ramCapacity'] = mean1
#
mean2 = round(user_basic_info.ramLeftRation.mean(), 2)
user_basic_info.loc[user_basic_info.ramLeftRation.isnull(), 'ramLeftRation'] = mean2
#
mean3 = round(user_basic_info.romCapacity.mean())
user_basic_info.loc[user_basic_info.romCapacity.isnull(), 'romCapacity'] = mean3
#
mean4 = round(user_basic_info.romLeftRation.mean(), 2)
user_basic_info.loc[user_basic_info.romLeftRation.isnull(), 'romLeftRation'] = mean4
#
mean5 = round(user_basic_info.fontSize.mean(), 2)
user_basic_info.loc[user_basic_info.fontSize.isnull(), 'fontSize'] = mean5
#
user_basic_info.loc[user_basic_info.ct.isnull(), 'ct'] = '4g#wifi'
#
mean6 = round(user_basic_info.os.mean(), 1)
user_basic_info.loc[user_basic_info.os.isnull(), 'os'] = mean6

user_basic_info.to_csv('user_basic_info_processed.csv', index=0, header=0)


def group(strs, thre):
    # print(user_basic_info.__getitem__(strs).value_counts())
    user_basic_info = pd.read_csv("user_basic_info_processed.csv", names=['uid', 'gender', 'city', 'prodName',
                                                                          'ramCapacity', 'ramLeftRation', 'romCapacity',
                                                                          'romLeftRation', 'color', 'fontSize', 'ct',
                                                                          'carrier', 'os'])
    dup = user_basic_info.drop_duplicates(strs)[strs].values
    cnt = dup.size
    group = [0] * cnt

    count = []
    print(cnt)
    for i in range(cnt):
        print(i)
        count.append(user_basic_info.loc[user_basic_info.__getitem__(strs) == dup[i]].__getitem__(strs).count())

    now_df = pd.DataFrame()
    now_df[strs] = dup
    now_df['prod_cnt'] = count
    now_df['group'] = group
    # print(user_basic_info.prodName.count())
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
    now_df = user_basic_info.merge(now_df, how='left', on=strs)
    now_df[strs] = now_df.group
    now_df.drop(['group'], axis=1, inplace=True)
    print(now_df[strs])
    # input()
    # print(new_df.shape, user_basic_info.shape)
    now_df.to_csv('user_basic_info_processed.csv', index=0, header=0)
    print(strs + ' finished')


group('carrier', 0)
group('city', 0)
group('ct', 0)
names = ['prodName', 'ramLeftRation', 'romLeftRation', 'color', 'fontSize', 'os']
for row in names:
    group(row, 500)

