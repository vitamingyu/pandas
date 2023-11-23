# cost와 accuracy/r2score는 반비례한다.
import numpy as np
import math

real = np.array([10, 3, 3, 2, 11])
# pred = np.array([11, 5, 2, 4, 3])  실제값과 차이가 적음 3.8 -> 차이가 적으니 좋은 모델이다
pred = np.array([15, 8, 2, 9, 2])  # 실제값과 차이가 큼 36.2
cost = 0
for i in range(5):
    cost += math.pow(pred[i] - real[i], 2)  # 제곱
    print(cost)

print(cost / len(pred))

print('------------------')
import tensorflow as tf
import matplotlib.pyplot as plt

x = [1.,2.,3.,4.,5.]
y = [1.,2.,3.,4.,5.]
b = 0
# hypothesis = w*x + b
# cost = tf.reduce_sum(tf.pow(hypothesis - y, 2)) / len(x)
# cost = tf.reduce_mean(tf.pow(hypothesis - y, 2))  이 두개가 tf에서 cost를 minimize하는 방법

w_val = []
cost_val = []

for i in range(-30, 50):
    feed_w = i * 0.1  # 뒤 0.1은 learning rate
    hypothesis = tf.multiply(feed_w, x) + b  # w*x + b
    cost = tf.reduce_mean(tf.square(hypothesis - y))
    cost_val.append(cost)
    w_val.append(feed_w)
    print(str(i) + '번 수행, ' + 'cost : ' + str(cost.numpy()) + ', weight : ' + str(feed_w))

plt.plot(w_val, cost_val)
plt.xlabel('w')
plt.ylabel('cost')
plt.legend()
plt.show()
