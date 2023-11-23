# 단순선형회귀 모델 작성
# 1) keras의 내장 api 사용 - Sequential : 다음번 예제
# 3) GradientTape 객체를 이용해 모델을 구현 - 유연하게 복잡한 로직을 처리할 수 있다.
# Tensorflow는 GradientTape을 이용하여 쉽게 오차 역전파를 수행할 수 있다.

import tensorflow as tf
import numpy as np
from keras.optimizers import SGD, RMSprop, Adam

w = tf.Variable(tf.random.normal((1,)))
b = tf.Variable(tf.random.normal((1,)))
print(w.numpy(), b.numpy())
opti = SGD()  # 경사하강법. RMSprop, Adam도 쓸 수 있음

@tf.function
def trainModel(x,y):
    with tf.GradientTape() as tape:
        hypo = tf.add(tf.multiply(w, x), b)  # wx + b
        loss = tf.reduce_mean(tf.square(tf.subtract(hypo, y)))  # cost function
    grad = tape.gradient(loss, [w, b])  # 자동 미분(loss를 w와 b로 미분(그래디언트)
    # w = w - 러닝레이트 * w에 대한 손실의 편미분. b역시 마찬가지
    # 'tape.gradient(loss, [w, b])는 이 편미분 값을 계산하고, 최적화 알고리즘을 통해 w와 b를 업데이트합니다.
    opti.apply_gradients(zip(grad, [w, b]))  # zip : 튜플의 형태로 차례로 접근할 수 있는 반복자를 반환
    # 경사 하강법 알고리즘을 통해 모델의 가중치 w와 편향 b를 업데이트하는 코드입니다.
    return loss  # 손실함수를 통해 MSE의 값이 반환되는듯

x = [1., 2., 3., 4., 5.]
y = [1.2, 2.0, 3.0, 3.5, 5.5]
print('x와 y의 상관계수 : ', np.corrcoef(x, y))

w_val = []
cost_val = []

for i in range(1, 101):
    loss_val = trainModel(x, y)
    cost_val.append(loss_val.numpy())
    w_val.append(w.numpy())
    if i % 10 == 0:
        print(i,'번 째 MSE(낮아지는 중 ~ ) : ', loss_val)

print('cost_val(MSE값들 모두 나타냄) : ', cost_val)
print('w_val : ', w_val)

import matplotlib.pyplot as plt
plt.plot(w_val, cost_val, 'o')
plt.xlabel('w')
plt.ylabel('cost')
plt.show()

print('cost가 최소일 때 w: ', w.numpy())
print('cost가 최소일 때 b: ', b.numpy())
y_pred = tf.multiply(x, w) + b  # 선형회귀식 완성
print('예측값 : ', y_pred.numpy())

plt.plot(x, y, 'ro', label='real y')  # 빨강색 o
plt.plot(x, y_pred, 'b-', label='pred')  # 파랑색 실선
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()

# 새 값으로 예측
new_x = [3.5, 9.0]
new_pred = tf.multiply(new_x, w) + b
print('예측 결과값 : ', new_pred.numpy())
