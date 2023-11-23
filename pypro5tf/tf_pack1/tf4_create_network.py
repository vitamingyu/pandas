# keras 모듈(라이브러리)을 사용하여 네트워크 구성
# 논리회로 분류 모델

import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Activation
import numpy as np
from keras.optimizers import SGD, RMSprop, Adam

# 1) 데이터 수집 및 가공
x = np.array([[0,0], [0,1], [1,0], [1,1]])
y = np.array([0,1,1,0])

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
# model.compile(optimizer='sgd', loss='binary_crossentropy', metrics=['accuracy'])
# model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
# model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
# model.compile(optimizer=SGD(learning_rate=0.01), loss='binary_crossentropy', metrics=['accuracy'])
# model.compile(optimizer=SGD(learning_rate=0.01, momentum=0.9), loss='binary_crossentropy', metrics=['accuracy'])
# model.compile(optimizer=RMSprop(learning_rate=0.01), loss='binary_crossentropy', metrics=['accuracy'])
model.compile(optimizer=Adam(learning_rate=0.01), loss='binary_crossentropy', metrics=['accuracy'])
# classification(분류)에선 accuracy. regression일땐 MSE써줌
# metrics는 성능을 확인하기 위해 써줌
# optimizer : 입력데이터와 손실함수를 업데이트하는 메커니즘이다. 손실함수의 최소값을 찾는 알고리즘을 옵티마이저라 한다
# - adam은 선형, 비선형 모두 쓸 수 있음. 발전순서 : sgd -> rmsprop -> adam
# optimizer='sgd'는 클래스를 객체화해서 쓰는 것이고 optimizer=SGD()이거는 클래스째로 쓰는거임
# 클래스를 써주면 기본값을 수정할 수 있음
# learning_rate=0.01이 기본값임

# 4) 모델 학습시키기(train data) : 더 나은 표현(w갱신)을 찾는 자동화 과정
model.fit(x, y, epochs=500, batch_size=1, verbose=0)
# batch_size : 훈련데이터를 여러개의 작은 묶음(batch)으로 만들어 가중치(w)를 갱신. 1 epoch시 사용하는 dataset의 크기
# epochs 학습횟수, batch_size 문제지와 답은 매번 1문제 풀 때마다 맞춤

# 5) 모델 평가(test data)
loss_metrics = model.evaluate(x, y, batch_size=1, verbose=0)
print(loss_metrics)

# 6) 학습 결과 확인 : 예측값 출력
# pred = model.predict(x, batch_size=1, verbose=0)
pred = (model.predict(x) > 0.5).astype('int32')
print('예측결과 : ', pred.flatten())  # 차원을 떨굼

# 7. 모델 저장
model.save('tf4model.h5')  # hdf5

# 8. 모델 읽기
from keras.models import load_model
mymodel = load_model('tf4model.h5')

mypred = (mymodel.predict(x) > 0.5).astype('int32')
print('예측결과 : ', mypred.flatten())  # 차원을 떨굼
