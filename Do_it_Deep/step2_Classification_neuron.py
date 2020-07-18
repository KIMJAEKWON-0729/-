from sklearn.datasets import load_breast_cancer
import numpy as np 
cancer = load_breast_cancer()

#입력 데이터 확인
print(cancer.data.shape,cancer.target.shape)

#cancer에는 569개의 샘플과 30개의 특성 첫 3개 샘플 출력

print(cancer.data[:3])

#실수범위의 값이며 양수로 이루어져 있다 산점도로 표현하기는 어려우므로 박스플롯을 이용하여 각 특서으이 사분위 값을 나타내기 

import matplotlib.pyplot as plt

plt.boxplot(cancer.data)
plt.xlabel('feature')
plt.ylabel('value')
#plt.show()
#박스플롯 4 14 24 번째 특성이 다른특성보다 값의 분포가 훨씬 크다는것을 확인 
#값이 여러개일 경우 괄호 두개 
print(cancer.feature_names[[3,13,23]])

#타깃 데이터 확인하기 
#넘파이의 unique()함수를 사용하면 고유한 값을 찾아 반환 이때 return_counts매개 변수를 True로 지정하면 고유한 값이 등장하는 횟수까지 세어 반환 

print(np.unique(cancer.target,return_counts = True))
#(array([0, 1]), array([212, 357]))
#212개의 음성 클래스 와 357개의 양성 클래스 

#훈련 데이터 세트 저장하기 ------------------------------------------------------------------

x = cancer.data

y = cancer.target

#훈련 데이터 세트를 나누기 전에 음성 클래스가 어느 한쪽에 몰리지 않도록 골고루 섞어야 한다 

#train_test_split 함수로 훈련 데이터 세트 나누기 
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,y,stratify = y, test_size = 0.2,
random_state = 42)
#stratify = y 는 타깃을 골고루 섞기 위해 

#결과 확인하기

print(x_train.shape,x_test.shape)
#(455, 30) (114, 30)

#3,unique()함수로 훈련 세트의 타깃 확인하기 
print(np.unique(y_train,return_counts=True))

#(array([0, 1]), array([170, 285]))

#로지스틱 회귀 구현하기 ------------------------------------------------------

class LogisticNeuron:
        def __init__(self):
            self.w = None
            self.b = None
        def forpass(self,x):
            #직선 방정식 계산
            z = np.sum(x*self.w) + self.b
            return z
        def backprop(self, x, err):
            w_grad = x *err
            b_grad = 1*err
            return w_grad, b_grad
        def activation(self,z):
            a = 1/(1+np.exp(-z))
            return a
        def fit(self,x,y, epochs = 100):
            self.w = np.ones(x.shape[1])
            self.b = 0
            for i in range(epochs):
                for x_i ,y_i,in zip(x,y):
                    z = self.forpass(x_i)
                    a = self.activation(z)
                    err = -(y_i-a)
                    w_grad, b_grad = self.backprop(x_i,err)
                    self.w -= w_grad
                    self.b -= b_grad
        #예측하는 메서드 구성하기 
        def predict(self,x):
            z =[self.forpass(x_i) for x_i in x]
            a = self.activation(np.array(z))
            return a >0.5
#모델 훈련하기 -------------------------------------------------------

neuron = LogisticNeuron()

neuron.fit(x_train, y_train)

#테스트 세트 사용해 모델의 정확도 평가하기

print(np.mean(neuron.predict(x_test) == y_test))
#0.8245614035087719

#내일도하기