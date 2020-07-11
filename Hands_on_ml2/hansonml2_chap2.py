#!/usr/bin/env python
# coding: utf-8



#데이터 다운로드 

#데이터를 다운로드 하는 함수를 준비하면 특히 데이터가 정기적으로 바뀌는 경우에 유용하다 

import os 
import tarfile
import urllib

import pandas as pd

os.getcwd()

download_root = "https://raw.githubusercontent.com/ageron/handson-ml2/master/"
HOUSING_PATH = os.path.join("datasets","housing")
HOUSING_URL = download_root +"datasets/housing/housing.tgz"

def fetch_housing_data(housing_url = HOUSING_URL, housing_path = HOUSING_PATH):
    os.makedirs(housing_path,exist_ok=True)
    tgz_path = os.path.join(housing_path,"housing.tgz")
    urllib.request.urlretrieve(housing_url,tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path = housing_path)
    housing_tgz.close()
#fetch_housing_data 를 호출하면 현재 작업공간에 datasets/housing 디렉터리르 만들고 
#housing.tgz 파일을 내려받고 같은 디렉터리에 압축을 풀어 housing.csv 파일을 만듭니다




import pandas as pd

#def load_housing_data(housing_path = HOUSING_PATH):
 #   csv_path = os.path.join(housing_path,"housing.csv")
  #  return pd.read_csv(csv_path)

#데이터 구조 훑어보기
housing = pd.read_csv("housing.csv")

housing.head()

#info 메서드는 데이터에 대한 간략한 설명과 전채 행수 데이터 타입 확인 
housing.info()
#RangeIndex: 20640 entries, 0 to 20639
#머신러닝 프로젝트치고는 작은사이즈
#total_bedrooms 특성은 20433 개만 널값이 아니다 207개는 특성을 가지고 있지 않다 
#ocean_proximity 필드만 배고 모든 틀성이 숫자형 
#ocean_proximity 변수 특성 확인 




housing["ocean_proximity"].value_counts( )




#describe()메서드는 숫자형 특성의 요약정보를 보여준다 
#summary
housing.describe()




#데이터의 형태를 빠르게 검토하는 다른방법은 각 숫자형 틀성을 히스토그램으로 그려보는것 
#주피터 노트북의 매직 명령

import matplotlib.pyplot as plt

housing.hist(bins=50,figsize=(20,15))

#plt.show()




#housing_median_age 와 median_house_value의 최대값들이 크다 한정해야 한다 

#테스트 세트 만들기 
import numpy as np 

def split_train_test(data,test_ratio):
    shuffled_indices = np.random.permutation(len(data))
    #permutation은 array를 복사하여 리턴 
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data.iloc[train_indices],data.iloc[test_indices]
#20프로 테스트 셋
train_set, test_set = split_train_test(housing , 0.2)
print(len(train_set)
)
print(len(test_set))




#위 코드를 쓸경우 다시 불러올떄 새롭게 섞인다 난수값이 필요 그리고 두 경우 데이터셋이 업데이트 될경우 샘플링이 이상해 진다 

from zlib import crc32

def test_set_check(identifier, test_ratio):
    return crc32(np.int64(identifier)) & 0xffffffff <test_ratio * 2 * 32
#비트 연산을 하는 이유는 파이썬2와 호환성 유지를 위해 

def split_train_test_by_id(data, test_ratio,id_column):
    ids = data[id_column]
    in_test_set = ids.apply(lambda id_ : test_set_check(id_,test_ratio))
    return data.loc[~in_test_set],data.loc[in_test_set]




# 주택 데이터 셋에는 식별자 컬럼이 없다 대신 행의 인덱스를 ID로 사용 

housing.reset_index()
#index 열이추가된 데이터 프레임이 반환 된다 

housing_with_id = housing.reset_index()

train_set, test_set = split_train_test_by_id(housing_with_id,0.2,"index")


#

housing_with_id["id"] =  housing["longitude"] * 1000 + housing["latitude"]

train_set, test_set = split_train_test_by_id(housing_with_id,0.2,"id")




#train_test_split  난수 초깃값을지정할수 있다 행의 개수가 같은 여러개의 데이터셋을 넘겨서 같은 인덱스를 기반으로 나눌수 있다 




from sklearn.model_selection import train_test_split

train_set, test_set = train_test_split(housing, test_size = 0.2, random_state = 42)




#테스트 세트 안 변수 하나가 중요하다 는 가정  이 테스트 세트가 전체 데이터 셋 에 있는 여러 소득 카테고리를 잘 대표 해야 한다 
# 이 셋안에서는 median_income 이 중요하다고 한다  

#위 히스토 그램 확인후 계급 설정 

housing["income_cat"] = pd.cut(housing["median_income"],
                              bins = [0,1.5,3.0,4.5,6,np.inf],
                              labels = [1,2,3,4,5])

housing["income_cat"].hist()



from sklearn.model_selection import StratifiedShuffleSplit

#StratifiedShuffleSplit 층위 무작위 추출을 통한 train test

split = StratifiedShuffleSplit(n_splits = 1, test_size= 0.2, random_state= 42)

for train_index, test_index in split.split(housing, housing["income_cat"]):
                                    strat_train_set = housing.loc[train_index]
                                    strat_test_set = housing.loc[test_index]



#소득 카테고리의 비율확인 
strat_test_set["income_cat"].value_counts()/len(strat_test_set)



#파생 변수를 만드는 목적이 아니기에 카테고리에 따라 비율을 나눠준뒤 INCOME_CAT 특성 삭제 

for set_ in (strat_train_set,strat_test_set):
    set_.drop("income_cat",axis=1,inplace = True)



# 데이터 이해를 위한 탐색과 시각화

#훈련세트를 손상시키지 않기 위해 복사본을 만들어 사용 

housing =  strat_test_set.copy()

#지리적 데이터 시각화 

housing.plot(kind = "scatter", x ="longitude",y = "latitude")
#kind 그래프 종류 # x y 는  y

#캘리포니아 지역의 위경도를 잘 나타 내지만 패턴을 찾아보기는 힘들다 

#alpha 옵션 조정 

housing.plot(kind = "scatter",x = "longitude", y = "latitude",alpha = 0.1)

#매개 변수 를 넣어 보기 s(scale) 는 인구 c(color) 색상은 가격 

housing.plot(kind = 'scatter', x = 'longitude', y = 'latitude', alpha = 0.4,
            s = housing['population']/100, c = 'median_house_value',cmap = plt.get_cmap('jet'),colorbar = True, 
            figsize = (10,7))

#상관관계 조사 

#데이터의 셋이 너무 크지 않으므로 모든 특성가느이 표준 상관계수(피어슨)를  corr()메서드를 이용해 쉽계 계산할수 있다

corr_matrix = housing.corr()

print(corr_matrix['median_house_value'].sort_values(ascending = False))

#데이터 정제 

housing = strat_train_set.drop("median_house_value",axis = 1)
print(housing)

housing_labels = strat_train_set["median_house_value"].copy()

print(housing_labels)

#total_bedrooms na 값 
#해당구역 제거 

print(housing.dropna(subset= ["total_bedrooms"])) 

#전체 특성을 삭제 

print(housing.drop("total_bedrooms",axis = 1))
#중간값으로 대체 

median = housing["total_bedrooms"].median()

print(housing["total_bedrooms"].fillna(median,inplace = True))

#sklearn 의 simpleimputer는 누락된 값을 손쇱게 다루도록 해준다 

from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy = "median")

#수치형 특성에서만 계산될수 있다 텍스트 특성 ocean_proximity 제외