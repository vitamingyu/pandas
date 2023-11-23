# 단순선형 회귀 - 경사하강법 함수 사용 1.x
import tensorflow.compat.v1 as tf   # tensorflow 1.x 소스 실행 시
tf.disable_v2_behavior()            # tensorflow 1.x 소스 실행 시
import matplotlib.pyplot as plt

x_data = [1.,2.,3.,4.,5.]
y_data = [1.2,2.0,3.0,3.5,5.5]

x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)
w = tf.Variable(tf.random_normal([1]))
b = tf.Variable(tf.random_normal([1]))

hypothesis = x * w + b
cost = tf.reduce_mean(tf.square(hypothesis - y))

print('\n경사하강법 메소드 사용------------')

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(cost)

sess = tf.Session()   # Launch the graph in a session.
sess.run(tf.global_variables_initializer())

w_val = []
cost_val = []

for i in range(501):
    _, curr_cost, curr_w, curr_b = sess.run([train, cost, w, b], {x:x_data, y:y_data})
    w_val.append(curr_w)
    cost_val.append(curr_cost)

    if i  % 10 == 0:
        print(str(i) + ' cost:' + str(curr_cost) + ' weight:' + str(curr_w) +' b:' + str(curr_b))

plt.plot(w_val, cost_val)
plt.xlabel('w')
plt.ylabel('cost')
plt.show()

print('--회귀분석 모델로 Y 값 예측------------------')
print(sess.run(hypothesis, feed_dict={x:[5]}))        # [5.0563836]
print(sess.run(hypothesis, feed_dict={x:[2.5]}))      # [2.5046895]
print(sess.run(hypothesis, feed_dict={x:[1.5, 3.3]})) # [1.4840119 3.3212316]

print('-------------------')
# 단순선형회귀 - 경사하강법 함수 사용 2.x
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping
from keras import optimizers
import numpy as np

model = Sequential()
model.add(Dense(units=1, input_dim=1, activation='linear'))
model.compile(optimizer='sgd', loss='mse', metrics=['mse'])  # mse : 평균 제곱 오차(mean squared error)
cb_early_stopping = EarlyStopping(monitor='val_loss', patience=100)
model.fit(x_data, y_data, batch_size=100, epochs=5000, verbose=2, callbacks=[cb_early_stopping])
# monitor='val_loss': 검증 손실을 모니터링합니다.
# patience=100: 검증 손실이 100 에폭 동안 향상되지 않으면 학습을 조기 중단
# 이렇게 설정된 EarlyStopping은 검증 손실이 100 에폭 동안 향상되지 않으면 학습을 중단
# 과적합 방지와 시간단축의 장점이 있다
print(model.evaluate(x_data, y_data))

pred = model.predict(x_data)
print('예측값 : ', pred.flatten())

import matplotlib.pyplot as plt
plt.plot(x_data, y_data, 'ro')
plt.plot(x_data, pred, 'b')
plt.show()

# 결정계수
from sklearn.metrics import r2_score
print('설명력(참고일 뿐 절대적으로 믿으면 안된다) : ', r2_score(y_data, pred))

# 새로운 값으로 예측
new_x = [1.5, 2.5, 3.3]
print('새로운 값 예측 결과 : ', model.predict(new_x).flatten())
