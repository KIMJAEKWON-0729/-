#선형 회귀의 목적 : 절편과 기울기 찾기 

#load_diabetes()

from sklearn.datasets import load_diabetes

diabetes = load_diabetes()

#입력과 타깃 데이터의 크기 확인 하기 

print(diabetes.data.shape, diabetes.target.shape)
#(442,10) (442,)

#442개의 행과 10개의 열로 구성되어 있음을 알수 있다 

#입력 데이터 자세히 보기 

diabetes.data[0:3]

import matplotlib.pyplot as plt

#산점도 
plt.scatter(diabetes.data[:,2],diabetes.target)
#plt.show()

#훈련 데이터 

x = diabetes.data[:,2]

y = diabetes.target

#경사 하강법 모델이 데이터를 잘 표현할수 있도록 기울기를 사용하여 모델을 조금씩 조정하는 최적화 알고리즘 

#딥러닝에서는  기울기 a를 종종 가중치를 의미하는 w나 계수를 의미하는 0 으로 표기 y는 y hat  

#훈련 데이터에 잘 맞는 w와 b를 잘 찾는 방법

#1.무작위로 w와 b를 정한다 
#2. x 에서  샘플 하나를 선택하여 y hat을 계산합니다(무작위로 모델 예측하기 )
#3. y hat과 선택한 샘플의 진짜 y를 비교합니다
#4. y hat 과 y 가 더 가까워 지도록 w,b를 조정합니다
#5. 모든 샘플을 처리할 때까지 다시 2~4 항목을 반복 

# w와 b 초기화 하기 
w = 1.0
b = 1.0

#훈련 데이터의 첫 번째 샘플 데이터로 y hat 얻기 

y_hat = x[0]*w +b
print(y_hat)
#1.06
print(y[0])
#151.0

#w값 조절해 예측값 바꾸기

w_inc = w+0.1
y_hat_inc = x[0]*w_inc +b

print(y_hat_inc)
#기존 y_hat 보다 조금더 증가 

#w값 조정한 후 예측값 증가 정도 확인하기 
#w 가 0.1 만큼 증가 했을때 y_hat이 얼마나 증가했는지 계산 

w_rate = (y_hat_inc - y_hat)/(w_inc-w)

print(w_rate)