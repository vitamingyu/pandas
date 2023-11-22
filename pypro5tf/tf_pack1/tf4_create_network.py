# keras 모듈(라이브러리)을 사용하여 네트워크 구성
# 논리회로 분류 모델

import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Activation
import numpy as np

# 1) 데이터 수집 및 가공
x = np.array([[0,0], [0,1], [1,0], [1,1]])
y = np.array([0,1,1,1])

# 2) 모델 구성(설정)
# model = Sequential([
#     Dense(input_dim=2, units=1),   #입력값이 2개가 들어와 노드 1개를 거쳐 출력값 1개로 나옴
#     Activation('sigmoid')
# ])

#줄여쓰기
# model = Sequential()
# model.add(Dense(units=1, input_dim=2))
# model.add(Activation('sigmoid'))

#더 줄여쓰기
model = Sequential()
model.add(Dense(units=1, input_dim=2, activation='sigmoid'))

# 3) 모델 학습 과정 설정(컴파일)
model.compile(optimizer='sgd', loss='binary_crossentropy', metrics=['accuracy'])

# 4) 모델 학습시키기
model.fit(x, y, epochs=500, batch_size=1, verbose=0)
# epochs 학습횟수, batch_size 문제지와 답은 매번 1문제풀 때마다 맞춤

# 5) 모델 평가
loss_metrics = model.evaluate(x, y, batch_size=1, verbose=0)

# 6) 학습 결과 확인 : 예측값 출력
# pred = model.predict(x, batch_size=1, verbose=0)
pred = (model.predict(x) > 0.5).astype('int32')
print('예측결과 : ', pred.flatten())  # 차원을 떨굼

