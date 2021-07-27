'''
종합 1,2,3등 무한야끼 1인분 + 비트 우수상
4, 5등 커피
6등 이하 나가리

문제 설명
시, 고, 저, 종가 거래량만으로 컬럼 구성
행은 2011년 1월 3일 이후 데이터로 구성

삼성과 sk 각각 5개의 컬럼씩
앙상블하고
1. 금요일 종가 맞추기
2. 월요일 시가 맞추기

쉬는 시간:
오전 매시 20분 ~ 30분
오후 매시 정각 ~ 10분
'''
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import Dense, LSTM, SimpleRNN, GRU, Conv1D, GlobalAvgPool1D, Flatten, Input, concatenate
from tensorflow.keras.models import Model, load_model, Sequential
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler, RobustScaler, StandardScaler, PowerTransformer, QuantileTransformer, OneHotEncoder
import time
import pandas as pd
from tensorflow.python.keras.layers.core import Dropout
import datetime

# 1. 데이터
samsung = pd.read_csv('D:/study/samsung/data/삼성전자 주가 20210721.csv', encoding='CP949')

sk = pd.read_csv('D:/study/samsung/data/SK주가 20210721.csv', encoding='CP949')

# 날짜 포맷 변경
samsung['일자'] = pd.to_datetime(samsung['일자'], format="%Y/%m/%d")
sk['일자'] = pd.to_datetime(sk['일자'], format="%Y/%m/%d")

# 날짜 추출
mask = samsung['일자'] >= '2011/01/03'
samsung = samsung.loc[mask]
mask = sk['일자'] >= '2011/01/03'
sk = sk.loc[mask]

# samsung = samsung[:2500]
# sk = sk[:2500]

# 날짜 오름차순으로 변경
samsung = samsung.sort_index(ascending=False)
sk = sk.sort_index(ascending=False)

# 일자, 시가, 고가, 저가, 종가, 거래량 열 추출
samsung = samsung.iloc[:,[1,2,3,4, 10]]
sk = sk.iloc[:, [1,2,3,4, 10]]

print(samsung.isnull().sum())
print(sk.isnull().sum())

samsung.dropna()
sk.dropna()

samsung = samsung.to_numpy()
sk = sk.to_numpy()

size = 5

def split_x(dataset, size):
    aaa = []
    for i in range(len(dataset) - size + 1):
        subset = dataset[i : (i + size),:]
        aaa.append(subset)
    return np.array(aaa)

x1 = split_x(samsung, size)
x2 = split_x(sk, size)
x1_predict = x1[x1.shape[0]-3:x1.shape[0], :, :] # predict는 전처리 완료 수정x.
x2_predict = x2[x2.shape[0]-3:x2.shape[0], :, :] # predict는 전처리 완료 수정x.
x1 = x1[:x1.shape[0]-3, :, :]
x2 = x2[:x2.shape[0]-3, :, :]
y = samsung[7:,0]

x1 = x1.reshape(x1.shape[0] * 5, 5)
x1_predict = x1_predict.reshape(x1_predict.shape[0] * 5, 5)
x2 = x2.reshape(x2.shape[0] * 5, 5)
x2_predict = x2_predict.reshape(x2_predict.shape[0] * 5, 5)

print('x1 :', y)

scaler1 = StandardScaler()
x1 = scaler1.fit_transform(x1)
x1_predict = scaler1.transform(x1_predict)

scaler2 = StandardScaler()
x2 = scaler2.fit_transform(x2)
x2_predict = scaler2.transform(x2_predict)

x1 = x1.reshape(2594, 5, 5)
x1_predict = x1_predict.reshape(3, 5, 5)
x2 = x2.reshape(2594, 5, 5)
x2_predict = x2_predict.reshape(3, 5, 5)


# print(x1.shape)
# print()
# print('x_predict', x1_predict[:,:])
# print('x 마지막 데이터', x1[-1,:])
# print('y 마지막 데이터', y[-1])
# print('x shape', x1.shape)
# print('y shape', y.shape)

# print('y', y[0])

# (N, 20, 5) = (N,)

# print(x1_train.shape)
# # 삼성전자 4-5일 정도로 잘라서 분석하기 
# # 모델 : 20일치로 나누기
input1 = Input(shape=(5,5))
# xx1 = SimpleRNN(units=256, activation='relu')(input1)
xx1 = Conv1D(64, activation='relu', kernel_size=5)(input1)
xx1 = Conv1D(8, activation='relu', kernel_size=1)(xx1)
xx1 = Flatten()(xx1)
xx1 = Dense(256, activation='relu')(xx1)
xx1 = Dense(256, activation='relu')(xx1)
xx1 = Dense(128, activation='relu')(xx1)
xx1 = Dense(128, activation='relu')(xx1)
xx1 = Dense(128, activation='relu')(xx1)
xx1 = Dense(64, activation='relu')(xx1)
out1 = Dense(64, activation='relu')(xx1)

input11 = Input(shape=(5,5))
# xx2 = SimpleRNN(units=256, activation='relu')(input11)
xx2 = Conv1D(64, activation='relu', kernel_size=5)(input11)
xx2 = Conv1D(32, activation='relu', kernel_size=1)(xx2)
xx2 = Conv1D(8, activation='relu', kernel_size=1)(xx2)
xx2 = Flatten()(xx2)
xx2 = Dropout(0.1)(xx2)
xx2 = Dense(256, activation='relu')(xx2)
xx2 = Dropout(0.1)(xx2)
xx2 = Dense(128, activation='relu')(xx2)
XX2 = Dense(64, activation='relu')(xx2)
out2 = Dense(64, activation='relu')(xx2)

mergexx = concatenate([out1, out2])
mergexx = Dense(256, activation='relu')(mergexx)
mergexx = Dense(256, activation='relu')(mergexx)
mergexx = Dense(128, activation='relu')(mergexx)
mergexx = Dense(128, activation='relu')(mergexx)
mergexx = Dense(64, activation='relu')(mergexx)
mergexx = Dense(32, activation='relu')(mergexx)
mergexx = Dense(8, activation='relu')(mergexx)
mergexx = Dense(8, activation='relu')(mergexx)
output = Dense(1)(mergexx)

# #########3
date = datetime.datetime.now()
date_time = date.strftime("%m%d_%H%M")

filepath = './samsung/_save/ModelCheckPoint/'
filename = '.{epoch:04d}-{loss: .4f}.hdf5'
modelpath = "".join([filepath, "KCS_", date_time, "_", filename])

# ##################################################################
model = Model(inputs=[input1, input11], outputs=output)

es = EarlyStopping(monitor='loss', patience=20, mode='min', verbose=1,
                    restore_best_weights=True)

mcp = ModelCheckpoint(monitor='loss', mode='min', verbose=1, save_best_only=True,
                        filepath= modelpath)
model.compile(loss='mse', optimizer='adam')
model.fit([x1, x2], y, epochs=500, batch_size=16, verbose=1, callbacks=[es, mcp])

# model.save('./samsung/_save/ModelCheckPoint/save_model2_1.h5')

model = load_model('./samsung/_save/ModelCheckPoint/save_model2_2.h5')
# model : 80333.28 : x loss :  1753347.75

# model2: 79991 : x

# model3 : 79492.48 loss :  1437712.625

# model4 : 79215.5 loss :  1017948.8125

# model5 : 81760.43 : x

result = model.evaluate([x1, x2], y)
print('loss : ', result)

y_predict = model.predict([x1_predict, x2_predict])
print('y_predict : ', y_predict)