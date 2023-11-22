#이 데이터는 어느 지역의 2011년 1월 ~ 2012년 12월 까지 날짜/시간. 기온, 습도, 풍속 등의 정보를 바탕으로
# 1시간 간격의 자전거 대여횟수가 기록되어 있다.
#train / test로 분류 한 후 대여횟수에 중요도가 높은 칼럼을 판단하여 feature를 선택한 후,
# 대여횟수에 대한 회귀 예측을 하시오. (배점:10)
#칼럼 정보 :
# 'datetime', 'season'(사계절:1,2,3,4),  'holiday'(공휴일(1)과 평일(0)), 'workingday'(근무일(1)과 비근무일(0)),
#  'weather'(4종류:Clear(1), Mist(2), Snow or Rain(3), Heavy Rain(4)),
#  'temp'(섭씨온도), 'atemp'(체감온도), 'humidity'(습도), 'windspeed'(풍속),
#  'casual'(비회원 대여량), 'registered'(회원 대여량), 'count'(총대여량)
#참고 : casual + registered 가 count 임.

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

from sklearn.metrics import accuracy_score, classification_report
from xgboost import plot_importance
from sklearn.model_selection import train_test_split
import xgboost
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB


df = pd.read_csv("bike_dataset.csv")
# print(df.head(3), df.shape, df.columns)  # (8124, 23)
# print(df['class'].unique())

df['datehour'] = df['datetime'].str[11:13].astype(int)
feature = df.drop(['count', 'casual', 'registered', 'datetime'], axis=1)
feature = pd.get_dummies(feature)
feature['datehour'] = df['datehour']
# print(feature['datehour'])
# print(feature.columns)

label = df['count']

x_train, x_test, y_train, y_test = train_test_split(feature, label, test_size=0.2, random_state=128)

model = LinearRegression()
model.fit(x_train, y_train)

pred = model.predict(x_train)
print(pred)


"""
model = xgboost.XGBClassifier(booster='gbtree', max_depth=6, n_estimators=500).fit(x_train, y_train)  # 0.9385
# print(model)
pred = model.predict(x_test)

from sklearn import metrics
acc = metrics.accuracy_score(y_test, pred)
print('acc : ', acc)
# print(metrics.classification_report(y_test, pred))

fig, ax = plt.subplots(figsize=(10, 12))  # xgboost에서만 적용됨
plot_importance(model, ax=ax)
plt.show()

real_features = feature[['stalk-root_b', 'gill-size_b', 'odor_n', 'spore-print-color_w', 'gill-color_w', 'bruises_f']]

x_train, x_test, y_train, y_test = train_test_split(real_features, label, test_size=0.2, random_state=159)
# model
gmodel = GaussianNB()
print(gmodel)
gmodel.fit(x_train, y_train)

pred = gmodel.predict(x_test)
print('예측값 : ', pred[:10])
print('실제값 : ', y_test[:10])

print('정확도 : ', accuracy_score(y_test, pred))
print('분류보고서 : \n', classification_report(y_test, pred))

# print('새값으로 예측')
# import numpy as np
# myweather = np.array([[2,12,0], [22,34,50],[1,2,3,]])
# print('예측 결과 : ', gmodel.predict(myweather))
"""