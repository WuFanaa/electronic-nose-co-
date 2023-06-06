import pandas as pd
import numpy as np

data_1 = np.array(pd.read_csv('1.txt',encoding='gbk'))
lable_1 = data_1[121:,1]
data_1 = data_1[121:,2:12]
data = []
lables = []
for i in range(len(data_1)):
    if i%180 == 0:
        temp1 = data_1[i:i+120,:]
        temp2 = lable_1[i+20:i+40]
        temp2 = round(np.mean(temp2),2)
        data.append(temp1)
        lables.append(temp2)
file_len = 2699
for i in range(3):
    sample_point = []
    url = str(i+2) + '.txt'
    temp_data = np.array(pd.read_csv(url,encoding='gbk'))
    for i in range(file_len):
        sample_point.append(int((i/file_len)*len(temp_data)))
    temp_data = np.array(temp_data[sample_point])
    for i in range(len(temp_data)):
        if i%180 == 0:
            temp1 = data_1[i:i+120,:]
            temp2 = lable_1[i+20:i+40]
            temp2 = round(np.mean(temp2),2)
            data.append(temp1)
            lables.append(temp2)

data = np.reshape(data,(120*60,10))
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
data = scaler.fit_transform(data)
data = np.reshape(data,(60,120,10,1))
lables = np.array(lables)
np.save('data',data)
np.save('lables',lables)