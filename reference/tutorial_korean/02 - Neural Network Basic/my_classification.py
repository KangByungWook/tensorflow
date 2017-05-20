# -*- coding: utf-8 -*-
# 털과 날개가 있는지 없는지에 따라 포유류인지 조류인지 분류하는 신경망 모델

import tensorflow as tf 
import numpy as np 

# [털, 날개]
x_data = np.array([[0,0], [1,0], [1,1], [0,0], [0,0], [0,1]])

# [기타, 포유류, 조류]
# 다음과 같은 형식을 one-hot 형식의 데이터라고 한다
# 한종류에만 해당하는 것을 의미하는듯...
y_data = np.array([
                  [1,0,0], # 기타
                  [0,1,0], # 포유류
                  [0,0,1], # 조류
                  [1,0,0],
                  [1,0,0],
                  [0,0,1]
                  ])

################
# 신경망 모델 구성
################
X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

# 첫번째 가중치의 차원은 2차원으로 [특성, 히든 레이어의 뉴런 갯수] -> [2,10]으로 정한다
W1 = tf.Variable(tf.random_uniform([2,10], -1., 1.))
# 두번째 가중치의 차원을 [첫번째의 히든 레이어의 뉴런 갯수, 분류 갯수] -> [10,3]으로 정한다
W2 = tf.Variable(tf.random_uniform([10,3], -1., 1.))

# 편향을 각각 각 레이어의 아웃풋 갯수로 설정
# b1 은 히든 레이어의 뉴런 갯수로, b2는 최종 결과값 즉 분류 갯수인 3으로 설정
b1 = tf.Variable(tf.zeros([10]))
b2 = tf.Variable(tf.zeros([3]))

# 신경망의 히든 레이어에 가중치 W1과 편향 b1을 적용
L = tf.add(tf.matmul(X, W1), b1)
# 가중치와 펴향을 이용해 계산한 결과 값에
# 텐서플로우에서 기본으로 제공하는 활성화 함수인 ReLU 함수를 적용
L = tf.nn.relu(L)

# 최종적인 아웃풋을 계산
# 히든 레이어에 두번째 가중치 W2와 편향 b2를 적용하여 3개의 출력값을 만든다
model = tf.add(tf.matmul(L, W2), b2)
# 마지막으로 softmax함수를 이용하여 출력값을 사용하기 쉽게 만든다
# softmax 함수는 결과값을 전체합이 1인 확률로 만들어주는 함수

model = tf.nn.softmax(model)

# 신경망을 최적화하기 위한 비용 함수를 작성
# 각 개별 결과에 대한 합을 구한 뒤 평균을 낸다
# 전체 합이 아닌, 개별 결과를 구한뒤 평균을 내는 방식을 사용하기 위해 axis옵션 사용
# axis 옵션이 없으면 -1.09처럼 총합인 스칼라값으로 출력
# 예측값과 실제값 사이의 확률 분포 차이를 비용으로 계산한 것
# 이것을 Cross-entropy라고 한다
cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(model), axis=1))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train_op = optimizer.minimize(cost)

####################
# 신경망 모델 학습
####################
init = tf.global_variables_initializer()
sess  = tf.Session()
sess.run(init)

for step in xrange(100):
  sess.run(train_op, feed_dict={X: x_data, Y: y_data})

  if (step + 1) % 10 == 0:
    print (step + 1), sess.run(cost, feed_dict={X:x_data, Y:y_data})


#####################
# 결과 확인
# 0: 기타 1: 포유류, 2:조류
#####################
# tf.argmax: 예측값과 실제값의 행렬에서 tf.argmax를 이용해 가장 큰 값을 가져옴
prediction = tf.argmax(model, 1)
target = tf.argmax(Y, 1)
print '예측값:', sess.run(prediction, feed_dict={X: x_data})
print '실제값:', sess.run(target, feed_dict={Y: y_data})

check_prediction = tf.equal(prediction, target)
accuracy = tf.reduce_mean(tf.cast(check_prediction, tf.float32))
print '정확도: %.2f' % sess.run(accuracy * 100, feed_dict={X: x_data, Y: y_data})
