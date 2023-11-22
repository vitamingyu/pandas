# 변수 선언 후 사용하기
# tf.Variable()

import tensorflow as tf

print(tf.constant(1.0))  # 고정된 자료를 기억

# 값이 동적으로 계속 변하면 Variable에 담아둠
f = tf.Variable(1.0)  # 변수형 텐서 : scalr라고 생각하면 됨
v = tf.Variable(tf.ones(2,))  # 1-d 텐서 : vector라 생각하면 됨 정확한 명칭은 0-d텐서가 맞음
m = tf.Variable(tf.ones(2,1))  # 2-d 텐서 : matix라 생각
print(f)
print(v)
print(m)

# 치환
v1 = tf.Variable(1)
v1.assign(10)  # 치환
print('v1 : ', v1, v1.numpy())

print()
w = tf.Variable(tf.ones(shape=(1,)))
b = tf.Variable(tf.ones(shape=(1,)))
w.assign([2])
b.assign([3])

# 아래는 만들어진 텐서를 연산을 파이썬의 가상환경에서 함. 그래서 속도가 느림
def func1(x):
    return w * x + b

print(func1(3))

@tf.function  # auto graph 기능 : 별도의 텐서 영역에서 연산을 수행. tf.Graph + tf.Session
def func2(x):
    return w * x + b

print(func2(3))

print('Variable의 치환 / 누적')
aa = tf.ones(2,1)
print(aa.numpy())
m = tf.Variable(tf.zeros(2,1))
print(m.numpy)
m.assign(aa)  # 치환
print(m.numpy())

m.assign_add(aa)
print(m.numpy())

m.assign_sub(aa)
print(m.numpy())

print('---------')
g1 = tf.Graph()

with g1.as_default():
    c1 = tf.constant(1, name='c_one')  # c1은 name이 c_one이란 이름을 갖고 정수 1을 가진 상수(고정된 값을 기억)다.
    print(c1)
    print(type(c1))
    print(c1.op)
    print('-----------')
    print(g1.as_graph_def())

print('~'*60)
g2 = tf.Graph()

with g2.as_default():
    v1 =tf.Variable(initial_value=1, name='v1')
    print(v1)
    print(type(v1))
    print(v1.op)

print(g2.as_graph_def())





