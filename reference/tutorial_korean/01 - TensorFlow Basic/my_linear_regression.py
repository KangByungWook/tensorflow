# -*- coding: utf-8 -*-
# X 와 Y의 상관관계를 분석하는 기초적인 선형회귀 모델을 만들고 실행해봅니다.

import tensorflow as tf

x_data = [1,2,3]
y_data = [1,2,3]

# random_uniform -> 2,3번째 숫자 사이의 임의의 값
# 예측할 변수값들
W = tf.Variable(tf.random_uniform([1], -1.0, -1.0))
b = tf.Variable(tf.random_uniform([1], -1.0, -1.0))

# placeholder -> 직접 채워주어야 할 부분
X = tf.placeholder(tf.float32, name="X")
Y = tf.placeholder(tf.float32, name="Y")

# X와 Y의 상관관계 분석을 위한 가설 수식 작성
# h = W * X + b
# W와 X가 행렬이 아니므로 tf.matmul이 아닌 tf.mul을 사용
hyphothesis = tf.add(tf.mul(W, X), b)

# 손실 함수를 작성
# mean(h - Y)^2 : 예측값과 실제값의 거리를 비용(손실) 함수로 정한단
cost = tf.reduce_mean(tf.square(hyphothesis - Y))
# 텐서플로우에 기본적으로 포함되어있는 함수를 이용하여 겨아 하강법 최적화 수행
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
# 비용을 최소화하는 것이 최종 목표
train_op = optimizer.minimize(cost)
# 모델 생성 완료!!

# 새션을 생성하고 초기화
with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())

  # 최적화를 100번 수행
  for step in xrange(100):
    # sess.run 을 통해 train_op와 cost 그래프를 계산
    # 이 떄, 가설 수식에 넣어야할 실제값을 feed_dict를 통해 전달
    _, cost_val = sess.run([train_op, cost], feed_dict={X: x_data, Y: y_data})

    print step, cost_val, sess.run(W), sess.run(b)
  print "\n=== Test ==="
  # 최적화가 완료된 모델에 테스트 값을 넣고 결과가 잘 나오는지 확인
  print "X: 5, Y:", sess.run(hyphothesis, feed_dict={X: 5})
  print "X: 2.5, Y:", sess.run(hyphothesis, feed_dict={X: 2.5})