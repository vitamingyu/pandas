from sklearn import svm, metrics
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('../testdata/Heart.csv')
df = df.dropna(subset=['Ca'])

feature = df.drop(['Unnamed: 0', 'AHD', 'ChestPain', 'Thal'], axis=1)
print(feature.head(3), feature.shape)
# print(feature.max(axis=0))  # 열(특성)별 최댓값

# print(feature.columns)
# print(df.head(3), df.shape)
# print(df.info())
feature = feature.dropna(subset=['Ca'])
# print(feature.isnull().sum())

AHD = df['AHD']
# print(AHD[:3])


# train / test
x_train, x_test, y_train, y_test = train_test_split(feature, AHD, test_size=0.3, random_state=158)
# print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

# scaler = StandardScaler()
# x_train_scaled = scaler.fit_transform(x_train)
# x_test_scaled = scaler.transform(x_test)

# 표준화 작업
# scaler = StandardScaler()
# # scaled_features = scaler.fit_transform(feature)
# # feature = pd.DataFrame(scaled_features, columns=feature.columns)

model = svm.SVC().fit(x_train, y_train)
print(model)

pred = model.predict(x_test)
print('예측값 : ', pred[:10])
print('실제값 : ', y_test[:10].values)

ac_score=metrics.accuracy_score(y_test, pred)
print('분류 정확도 : ', ac_score)

from sklearn import model_selection
cross_vali = model_selection.cross_val_score(model, feature, AHD, cv = 3)
print('각각의 검증 정확도 : ', cross_vali)
print('평균 검증 정확도 ', cross_vali.mean())

print()

# 새로운 값으로 분류 예측
newdata = pd.DataFrame({'Age':[80, 59, 70],'Sex':[1,0,1],'RestBP':[200,120,160],'Chol':[450,250,280],'Fbs':[1,0,0]
                        ,'RestECG':[2,2,2],'MaxHR':[190,108,129],'ExAng':[1,0,1], 'Oldpeak':[6.0,1.5,2.6], 'Slope':[3,2,2], 'Ca':[3.0,3.0,2.0]})

# print(newdata)
newPred = model.predict(newdata)
print('새로운 예측 결과 : ', newPred)

