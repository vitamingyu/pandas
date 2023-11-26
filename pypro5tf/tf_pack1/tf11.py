# 다중회귀분석
# feature 간 단위의 차이가 큰 경우 정규화/ 표준화 작업이 모델의 성능을 향상

import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import MinMaxScaler, minmax_scale, StandardScaler, RobustScaler
# robu는 이상치가 많을 때 씀

data = pd.read_csv('../testdata/Advertising.csv')
print(data.head(2))
del data['no']
print(data.head(2))

fdata = data[['tv','radio','newspaper']]
ldata = data.iloc[:, [3]]
print(fdata[:2])
print(ldata[:2])

# 정규화 (관찰값 - 최소값) / (최대값 - 최소값)
# 방법1) minmax 클래스로 씀
# scaler = MinMaxScaler(feature_range=(0,1))
# fdata = scaler.fit_transform(fdata)
# print(fdata[:2])

# 방법2)  minmax 객체를 씀
fedata = minmax_scale(fdata, axis=0, copy=True)  # copy=true 원본데이터는 보존
print(fedata[:2])

# train/ test split
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(fedata, ldata, shuffle=True, test_size=0.3, random_state=123)
# 시간의 순서에 따라 바뀌는 시계열 데이터면 shuffle쓰면 안됨
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

model = Sequential()
model.add(Dense(20, input_dim=3, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1, activation='linear'))

model.compile(optimizer='adam', loss='mse', metrics=['mse'])
print(model.summary())

from keras.utils import plot_model
plot_model(model, 'tf11.png')

history = model.fit(x_train, y_train, epochs=100, batch_size=32, verbose=2, validation_split=0.2)
# validation_data=(val_xtest, val_ytrain)

loss = model.evaluate(x_test, y_test, batch_size=32, verbose=0)
print(loss)

# history
print(history.history['loss'])
print(history.history['mse'])
print(history.history['val_loss'])
print(history.history['val_mse'])

# loss 시각화
import matplotlib.pyplot as plt
plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.legend()
plt.show()

from  sklearn.metrics import r2_score
print('설명력 : ', r2_score(y_test, model.predict(x_test)))

# predict
pred = model.predict(x_test[:3])
print('예측값 : ', pred.flatten())
print('실제값 : ', y_test[:3])

# 선형회귀 분석 모델의 충족조건 : 독립성, 정규성, 선형성, 등분산성, 다중공선성 확인
