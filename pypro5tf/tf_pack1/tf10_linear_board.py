# 다중선형회귀분석
# tensoboard : 머신러닝 실험을 위한 시각화 툴킷(toolkit)입니다.
# TensoBoard를 사용하면 손실 및 정확도와 같은 측정 항목을 추적 및 시각화되는 것, 모델 그래프를 시각화하는 것, 히스토그램을 보는 것,
# 이미지를 출력하는 것 등이 가능합니다.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras import optimizers

# 5명이 치른 세 번의 시험점수로 다음 번 시험 점수 예측
x_data = np.array([[70,85,80], [71,89,78],[50,85,60],[55,25,50],[50,35,10]])    # x_data는 각각의 특성(feature)을 나타냄
y_data = np.array([73,82,72,50,34])     # 출력 데이터 y_data는 모델이 예측해야 하는 값

print('1) Sequential api ---------')
# 여러 개의 레이어로 구성된 Sequential 모델을 정의
# 각 Dense 층은 모델의 계층을 나타냄. / 층을 여러 개로 쌓아가는 것은 모델이 더 복잡한 패턴을 학습할 수 있도록 하는데 도움.
# 첫 번째 층은 입력 특성과 연결되어 있는데, 모델이 학습하는 동안 제공되는 데이터의 특징을 나타냄.
# 나머지 층들은 이러한 입력을 기반으로 모델이 학습해야 할 내부 표현을 학습
model = Sequential()
model.add(Dense(units=6, input_dim=3, activation='linear', name='a'))    # 레이어의 뉴런은 6개, 독립 변수 feature가 3개 (입력 차원은 3차원)
model.add(Dense(units=3, activation='linear', name='b'))    # 리니어는 sigmoid를 태우거나 하지않고 들어온대로 그냥 내보냄
model.add(Dense(units=1, activation='linear', name='c'))    # activation='linear' - 활성화 함수로 선형 함수를 사용
# 입력 차원이 3이고 출력 차원이 1인 간단한 선형 회귀 모델
# 선형 가중치를 적용하여 예측을 수행하는 구조
print(model.summary())  # 모델의 구조와 파라미터 개수 등을 요약

opti = optimizers.Adam(learning_rate=0.01)  # Adam 옵티마이저를 생성, 학습률 0.01로 설정 / 학습률이 너무 작으면 느리게 진행, 너무 크면 최적의 지점을 지나칠 수 있음 0.01은 일반적인 학습률
model.compile(optimizer=opti, loss='mse', metrics=['mse'])  # 손실 함수로 평균 제곱 오차를 사용, 평가 지표도 평균 제곱 오차를 사용
history = model.fit(x_data, y_data, batch_size=1, epochs=50, verbose=2) # 모델을 주어진 데이터(x_data, y_data)로 학습
# batch_size=1: 한 번에 처리되는 데이터 샘플의 개수를 1로 설정하여 미니 배치 경사 하강법을 사용
# epochs=50: 전체 데이터셋에 대해 학습을 반복하는 횟수를 50으로 설정
# verbose=2: 학습 진행 상황을 자세히 출력 / verbose는 학습 과정 중에 출력되는 정보의 양을 조절하는 매개변수
# 0 = 출력이 없음 / 1 = 학습 진행 상황을 나타내는 진행 막대와 함께 출력 / 2 = 미니 배치마다의 손실 값 등을 자세하게 출력
print(history.history['loss'])

plt.plot(history.history['loss'])   # 학습 중에 기록된 손실(오차)의 변화를 그래프
plt.xlabel('epochs')
plt.ylabel('loss')
plt.show()

# 마지막 출력 값
# Epoch 50/50
# 5/5 - 0s - loss: 21.7005 - mse: 21.7005 - 12ms/epoch - 2ms/step
# 에폭 수, loss - 현재까지의 전체 손실 값(모델이 학습한 전체 데이터 손실 값. 모델의 예측이 실제 값과 얼마나 차이가 나는지)
# mse - 현재까지의 평균 제곱 오차(예측 값과 실제 값 간의 평균 제곱 차이를 나타내는 지표), 에폭당 평균 소요 시간, 각 배치 단계당 평균 소요 시간

loss_metrics = model.evaluate(x_data, y_data, batch_size=1, verbose=0)  # 모델을 평가 / x_data와 y_data는 각각 입력 데이터와 실제 값, batch_size=1은 배치 크기를 1
print('loss_metrics : ', loss_metrics)  # 모델의 평가 결과인 손실(loss) 및 지정한 메트릭(여기서는 MSE)을 출력
from sklearn.metrics import r2_score
print('설명력 : ', r2_score(y_data, model.predict(x_data)))    #  R-squared(결정 계수)를 출력
# 모델이 데이터를 얼마나 잘 설명하는지에 대한 지표로, 1에 가까울수록 좋다.
print()



print('2) functional api ---------')
from keras.layers import Input
from keras.models import Model

inputs = Input(shape=(3,))  # 입력층을 정의. shape=(3,)는 입력 데이터의 크기가 3이라는 것을 의미
output1 = Dense(6, activation='linear', name='a')(inputs)   # 각각 모델의 층을 나타냄. 각 층은 이전 층의 출력을 입력으로 받아 층을 구성
output2 = Dense(3, activation='linear', name='b')(output1)
output3 = Dense(1, activation='linear', name='c')(output2)
model2 = Model(inputs, output3) # model 클래스를 사용하여 모델을 생성. 입력과 출력을 지정하여 모델을 정의
print(model2.summary())

opti = optimizers.Adam(learning_rate=0.01)
model2.compile(optimizer=opti, loss='mse', metrics=['mse'])

# Tensorboard
from keras.callbacks import TensorBoard


tb = TensorBoard(
    log_dir="./my",
    histogram_freq=0,
    write_graph=True,
    write_images=False,
    write_steps_per_second=False,
    update_freq='epoch',
    profile_batch=2,
    embeddings_freq=1
)

history = model2.fit(x_data, y_data, batch_size=1, epochs=50, verbose=1, callbacks=[tb])
print(history.history['loss'])

loss_metrics = model2.evaluate(x_data, y_data, batch_size=1, verbose=0)
print('loss_metrics : ', loss_metrics)
print('설명력 : ', r2_score(y_data, model2.predict(x_data)))

# 새로운 값 예측
x_new = np.array([[30,35,30],[5,7,88]])     # 새로운 입력 데이터를 생성
print('예상 점수 : ', model2.predict(x_new).flatten())  # 새로운 입력에 대한 모델의 예측 결과를 출력
# predict() : 훈련된 모델을 사용하여 새로운 입력 데이터에 대한 예측을 생성하는 메서드
# flatten() : 2D 배열을 1D 배열로 변환 / 결과를 더 편리하게 활용하기 위한 목적
