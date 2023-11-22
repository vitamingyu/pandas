# 구글에서 만든 딥러닝 프로그램을 쉽게 구현할 수 있도록 기능을 제공하는 라이브러리이다.
# 텐서플로 자체는 기본적으로 C++로 구현이 되나 파이썬, 자바, 고(go) 등 다양한 언어를 지원한다
# 하지만 파이썬을 최우선으로 지원하며 대부분의 편한 기능들을 파이썬 라이브러리로만 구현 되어 있어서 python으로 개발하는 것을 추천~

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf

# print(tf.__version__)
# print('GPU사용 가능' if tf.test.is_gpu_available() else 'GPU사용 불가능 ㅠ')
# print(tf.config.list_physical_devices('GPU'))

print('tensor : 수치용 컨테이너다. 임의의 차원을 가지는 행렬의 일반화된 모습이다. 계산 그래프 구조를 갖음. 병렬연산이 기본')
print(1, type(1))  # 1 <class 'int'>
print(tf.constant(1), type(tf.constant(1)))  # tf.Tensor(1, shape=(), dtype=int32) <class 'tensorflow.python.framework.ops.EagerTensor'> 0-d tensor : scalar
print(tf.constant([1]), ' ', tf.rank(tf.constant([1])))  # 1-d tensor : vector  1차원
print(tf.constant([[1]]))  # 2-d tensor : matrix  2차원
print()
a = tf.constant([1,2])
b = tf.constant([3,4])
c = a + b
print(c)
d = tf.add(a,b)
print(d)
print()
print(7)  # 파이썬
# 아래는 7이라는 값이 다 텐서로 바뀜. 4가지 방법
print(tf.convert_to_tensor(7, dtype=tf.float32))
print(tf.cast(7, dtype=tf.float32))
print(tf.constant(7.0))
print(tf.constant(7, dtype=tf.float32))

print()
import numpy as np
arr = np.array([1,2])
print(arr, type(arr))
tfarr = tf.add(arr, 5)
print(tfarr)
print(tfarr.numpy())  # 텐서를 다시 넘파이로 바꾸고 싶다면
print(np.add(tfarr, 3))
