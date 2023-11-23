# keras 모듈(라이브러리)을 사용하여 네트워크 구성
# 논리회로 분류 모델

import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Activation
import numpy as np
from keras.optimizers import SGD, RMSprop, Adam

# 1) 데이터 수집 및 가공
x = np.array([[0,0], [0,1], [1,0], [1,1]])
y = np.array([0,1,1,0])  # xor

# 2) 모델 구성(설정)
model = Sequential()

# ------------
# model.add(units=5, input_dim=2)
# model.add(Activation('relu'))  # 활성화함수(sigmoid, relu, ...) 중 하나
# relu = sigmoid와 tanh가 갖는 Gradient Vanishing(기울기 소멸문제) 문제를 해결하기 위한 함수이다
# - Back Propagation에서 계산결과와 정답과의 오차를 통해 가중치를 수정하는데,
# 입력층으로 갈수록 기울기가 작아져 가중치들이 업데이트되지 않아 최적의 모델을 찾을 수 없는 문제입니다.
# model.add(units=1)
# model.add(Activation('sigmoid'))  # 노드를 추가함
# -------------
# 간결하게
model.add(Dense(units=5, input_dim=2, activation='relu'))  # 첫번째 히든레이어에선 노드 5개로 빠져나감
# 중간에 있는 히든레이어에서 sigmoid를 쓸 수도 있지만 기울기소실문제 때문에 relu를 써줌
model.add(Dense(units=1, activation='sigmoid'))  # 두번째 레이어, 출력층에서는 1개로 빠져나감.
# 빠져나갈 땐 이항분류일 경우 sigmoid, 다항분류일 경우 softmax를 써줌

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
history = model.fit(x, y, epochs=1000, batch_size=1, verbose=0)
loss_metrics = model.evaluate(x,y)
print(loss_metrics)

pred = (model.predict(x) > 0.5).astype('int32')
print('예측결과 : ', pred.flatten())  # 차원을 떨굼

print(model.summary())

print()
print(model.input)
print(model.output)
print(model.weights)
print('**'*20)
print(history.history)
print(history.history['loss'])
print(history.history['accuracy'])

# 시각화
import matplotlib.pyplot as plt
plt.plot(history.history['loss'], label='train loss')
plt.xlabel('epochs')
plt.legend()
plt.show()

plt.plot(history.history['accuracy'], label='train accuracy')
plt.xlabel('epochs')
plt.legend()
plt.show()

import pandas as pd
pd.DataFrame(history.history)['loss'].plot(figsize=(8, 5))
plt.show()
