# tf.constant() : 텐서를 직접 기억
# tf.Variable() : 텐서가 저장된 주소를 참조
import tensorflow as tf
import numpy as np

node1 = tf.constant(3, dtype=tf.float32)
node2 = tf.constant(4.0)
print(node1)
print(node2)
imsi = tf.add(node1, node2)
print(imsi)

print()
node3 = tf.Variable(3, dtype=tf.float32)
node4 = tf.Variable(4.0)
print(node3)
print(node4)
node4.assign_add(node3)
print(node4)

print()
a = tf.constant(5)
b = tf.constant(10)
c = tf.multiply(a ,b)
result = tf.cond(a < b, lambda :tf.add(10, c), lambda :tf.square(a))  # 거짓이면 제곱: square
# lambda는 익명함수 정의
print('result : ', result.numpy())

print('-----')
v = tf.Variable(1)

@tf.function  # Graph 환경에서 처리가 됨
def find_next_func():
    v.assign(v + 1)
    if tf.equal(v % 2, 0):  # 나눈 나머지가 0이면 아래 조건 수행
        v.assign(v + 10)

find_next_func()
print(v.numpy())
print(type(find_next_func))  # <class 'NoneType'>

# <class 'function'>
# 아래는 @tf.function  # Graph 환경에서 처리가 됨 이후 타입
# <class 'tensorflow.python.eager.polymorphic_function.polymorphic_function.Function'>

print('func1 -------------')
def func1():
    imsi = tf.constant(0)  # imsi =0로 써도 아래서 어차피 형변환 되니 똑같은 결과
    su = 1
    for _ in range(3):
        imsi = tf.add(imsi, su)
    return imsi

kbs = func1()
print(kbs.numpy(), ' ', np.array(kbs))  # 두개 똑같이 tensor를 ndarray로 바꾸는 것

print('func2 -------------')
imsi = tf.constant(0)
@tf.function
def func2():
    # imsi = tf.constant(0)
    global imsi  # 함수내에선 로컬만 먹으니 global을 써줘서 전역변수를 써준다
    su = 1
    for _ in range(3):
        # imsi = tf.add(imsi, su)  컨스턴트여서 아래 3가지 방법이 다 됨
        # imsi = imsi + su
        imsi += su
    return imsi

kbs = func2()
print(kbs.numpy(), ' ', np.array(kbs))  # 두개 똑같이 tensor를 ndarray로 바꾸는 것

print('func3 -------------')
imsi = tf.Variable(0)
@tf.function
def func3():
    # imsi = tf.Variable(0)  # autograph에서는 Variable()은 함수 밖에서 선언
    su = 1
    for _ in range(3):
        imsi.assign_add(su)  # variable은 assign_add로 해야됨
        # imsi = imsi + su  # variable은 이 연산 안됨
        # imsi += su  # variable은 이 연산 안됨.    variable은 assign을 통해 누적연산을 해야됨
    return imsi

kbs = func3()
print(kbs.numpy(), ' ', np.array(kbs))  # 두개 똑같이 tensor를 ndarray로 바꾸는 것

print('구구단 출력 --------------')
@tf.function
def gugu1(dan):
    su = 0
    for _ in range(9):
        su = tf.add(su, 1)
        # print(su.numpy())  autograph쓰면 오류남
        print(su)
        # 오토그래프는 연산만 하는 곳임. 그런데 su.numpy()같이 형변환은 안 되는거임
        print('{} * {} = {}'.format(dan, su, dan*su))  # 오류는 안되지만 서식이 있는 출력 안 먹음. 오토그래프 빼면 됨
        # tf.print(su) 이는 그래프 내부에서 출력
        # tf.print(dan*su)

gugu1(3)

print('-------------')
# 내장함수 : 일반적으로 numpy 지원함수를 그대로 사용. + 알파
# ... 중 reduce~ 함수
ar = [[1., 2.], [3.,4.]]  # 실수연산ㅇ르 위해 .을 찍어줌
print(tf.reduce_sum(ar))  # 2차원이었는데 차원을 떨어뜨리고 합을 구함
print(tf.reduce_sum(ar).numpy())
print(tf.reduce_mean(ar).numpy())
print(tf.reduce_mean(ar, axis=0).numpy())  # 열기준  [2. 3.]
print(tf.reduce_mean(ar, axis=1).numpy())  # 행기준  [1.5 3.5] 넘파이 슬라이싱과 헷갈리지 말자!

# one_hot encoding
print(tf.one_hot([0, 1, 3, 2, 0], depth=4))
print(tf.one_hot([2,10,2,5,4], depth=11))
