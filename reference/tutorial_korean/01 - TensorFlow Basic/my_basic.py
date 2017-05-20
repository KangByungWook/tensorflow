# -*- coding: utf-8 -*-
# 2017.02.12

import tensorflow as tf 
print "01 - Basic.py 시작"

hello = tf.constant('hello, TensorFlow!')

a = tf.constant(10)
b = tf.constant(32)
c = a + b

# tf.placeholder: 계산 실행 시 입력값을 받는 편수로 사용
# None은 크기가 정해지지 않았음을 의미
X = tf.placeholder("float", [None, 3])

# tf.Varialbe: 그래프를 계산하면서 최적화 할 변수들. 신경망을 좌우하는 값
# tf.random_normal: 각 변수들의 초기값을 정규분포 랜덤 값으로 초기화
# name: 나중에 텐서보드등으로 값의 변화를 추적하거나 살펴보기 쉽게 하기 위해 이름을 붙임
W = tf.Variable(tf.random_normal([3,2]), name='Weights')
b = tf.Variable(tf.random_normal([2,1]), name='Bias')

x_data = [[1,2,3], [4,5,6]]

# 입력값과 변수들을 계산할때 수식을 작성
# tf.matmaul 처럼 mat* 로 되어있는 함수로 행렬 계산을 수행
expr = tf.matmul(X, W) + b

# 그래프를 실행할 세션을 구성
sess = tf.Session()

# sess.run: 설정한 텐서 그래프(변수나 수식 등등)를 실행합니다
# 최초에 tf.global_variables_initializer() 를 한번 실행해야 합니다
sess.run(tf.global_variables_initializer())

# 위에서 변수와 수식들을 정의했지만, 실행이 정의한 시점에서 실행되는 것은 아님
# sess.run함수를 이용해 실제 값을 넣어주면 그때 계산된다
# 모델을 구성하는 것과 실행하는 것을 분리하여 프로그램을 깔끔하게 작성할 수 있다
print "==contants=="
print sess.run(hello)
print "a + b = c =", sess.run(c)
print "=== x_data ==="
print x_data
print "=== W ==="
print sess.run(W)
print "=== b ==="
print sess.run(b)
print "=== expr ==="
# expr 수식에는 X라는 입력값이 필요하다!!
# 따라서 expr 실행시에는 이 변수에 대한 실제 입력값을 넣어주어야 한다
print sess.run(expr, feed_dict={X: x_data})

# 세션을 닫는다
sess.close()

print "01 - Basic.py 끝"

