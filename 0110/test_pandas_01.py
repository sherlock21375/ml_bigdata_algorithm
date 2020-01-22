import pandas as pd
import numpy as np

df = pd.DataFrame(np.arange(20).reshape(5,4),index=[1,3,4,6,8])
print(df)
print(df.reset_index(drop=True))

df = pd.DataFrame({'key1':list('aabba'),
                  'key2': ['one','two','one','two','one'],
                  'data1': np.random.randn(5),
                  'data2': np.random.randn(5)})
print(df)

for i in df.groupby('key1'):
    print(i)

print(df.groupby('key1').size())

for i in df.groupby(['key1','key2']):
    print(i)  #似乎只能通过这样一种方式进行数据的读取工作

df_size = df.groupby(['key1','key2']).size()
print(df_size[0])


d = df.apply(lambda x:x.describe())
print(d)


#填补空值
df = pd.DataFrame({'A':['bob','sos','bob','sos','bob','sos','bob','bob'],
              'B':['one','one','two','three','two','two','one','three'],
              'C':[3,1,4,1,5,9,None,6],
              'D':[1,2,3,None,5,6,7,8]})
grouped = df.groupby('A')

for name,group in grouped:
    print(name)
    print(group)

def fill_none(one_group):
   return one_group.fillna(one_group.mean()) # 把平均值填充到空值里面


d = grouped.apply(fill_none)
print(d)