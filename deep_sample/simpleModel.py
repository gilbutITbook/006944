#!/usr/bin/env python
#coding:utf-8

import numpy as np
import chainer.functions as F
import chainer.links as L
from chainer import Variable, optimizers 

#모델 정의
model = L.Linear(1,1)
optimizer = optimizers.SGD()
optimizer.setup(model)

#학습시킬 횟수
times = 50

#입력 벡터
x = Variable(np.array([[1]], dtype=np.float32))

#정답 벡터
t = Variable(np.array([[2]], dtype=np.float32))

#학습 그룹
for i in range(0, times):
	#경사를 초기화
	optimizer.zero_grads()

	#여기에서 모델로 예측시킨다 
	y=model(x)

	#모델이 내린 답을 표시
	print(y.data)

	#손실을 계산한다 
	loss=F.mean_squared_error(y, t)

	#역전파한다 
	loss.backward()

	#optimizer를 갱신한다
	optimizer.update()
