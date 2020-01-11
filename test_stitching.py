# -*- coding: utf-8 -*-
# @Time    : 2020/1/10 9:19
# @Author  : Sherlock June
# @Email   : 1738502793@qq.com
# @File    : test_stitching.py
# @Software: PyCharm

import sys
sys.path.append(r'../')
import pandas as pd
pd.set_option('expand_frame_repr', False)  # 数据超过总宽度后，是否折叠显示，看一下更多的数据

def stitching(part_1, part_2):
    data = pd.concat([part_1, part_2])
    # 多次检查项目进行拼接
    vid_table = data.groupby(['vid', 'table_id']).size().reset_index()  #针对病人id（vid）和检查项目（table_id），进行重复组合
    print("----------1-----------")
    # print(vid_table)
    vid_table['new_index'] = vid_table['vid'] + '_' + vid_table['table_id']  #组合生成新的index
    print("----------2-----------")
    # print(vid_table[0])  #似乎vid_table[0]只能针对groupby之后的数据才能进行输出最后一列，我的想法是输出的不是列，而是size的情况
    vid_table_dup = vid_table[vid_table[0] > 1]['new_index']  #拎出重复大于1的列数，组成待删除组
    print("----------3-----------")
    # print(vid_table_dup)
    data['new_index'] = data['vid'] + '_' + data['table_id']#重新给原始拼接的data进行组合
    print("----------4-----------")
    # print(data.head())
    dup_part = data[data['new_index'].isin(list(vid_table_dup))]#根据vid_table_dup找出的重复，进行查询，找出原始data中需要删除的部分
    print("----------5-----------")
    # print(dup_part.head())
    dup_part = dup_part.sort_values(['vid', 'table_id'])#重新排序，更新原始data需要删除的部分组成的新组合
    print("----------6-----------")
    # print(dup_part.head(20))
    # print(dup_part['field_results'])
    unique_part = data[~data['new_index'].isin(list(vid_table_dup))]#找到data非重复部分，组合为unique_part
    print("----------7-----------")
    # print(unique_part.head(20))


    # 重复数据的拼接操作
    def merge_table(df):                 #针对dup_part，将其重复的检查项目，但是不同的结果进行拼接
        df['field_results'] = df['field_results'].astype(str)
        if df.shape[0] > 1:
            merge_df = " ".join(list(df['field_results']))
        else:
            merge_df = df['field_results'].values[0]
        return merge_df
    data_dup = dup_part.groupby(['vid', 'table_id']).apply(merge_table).reset_index()  #通过apply调用merge_table方法，拼接
    print("----------8-----------")
    # print(data_dup.head(20))
    data_dup.rename(columns={0: 'field_results'}, inplace=True)   #拼接完之后需要对列名进行重命名
    print("----------9-----------")
    # print(data_dup.head(20))
    data_res = pd.concat([data_dup, unique_part[['vid', 'table_id', 'field_results']]])  #重复部分数据（已整合）和非重复的数据拼接
    print("----------10-----------")
    # print(data_res.head(20))
    data = data_res.pivot(index='vid', columns='table_id')['field_results'].reset_index()  #行列转置
    print("----------11-----------")
    # print(data.head(20))
    data.to_csv('../data/data_all.csv', index=False)  #初步处理后的df保存为csv

part_1 = pd.read_csv('../data/meinian_round1_data_part1_20180408.txt', sep='$')
print("--------读取meinian_round1_data_part1_20180408.txt-------")
part_2 = pd.read_csv('../data/meinian_round1_data_part2_20180408.txt', sep='$')
print("--------读取meinian_round1_data_part2_20180408.txt-------")
stitching(part_1, part_2)