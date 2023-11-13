# 단순 선형 회귀 : iris dataset, ols()
# 상관관계가 약한 경우와 강한 경우를 나눠 분석 모델을 작성 후 비교

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf

iris = sns.load_dataset('iris')
print(iris.head(3), iris.shape)
print(iris.iloc[:,0:4].corr())

# 상관관계가 약한 경우  sepal_length, sepal_width : -0.117570
result1 = smf.ols(formula='sepal_length ~ sepal_width', data=iris).fit()
print('result1 모델 정보 : ', result1.summary())
print('result1 R_squared : ', result1.rsquared)  # 0.0138
print('result1 p-value : ', result1.pvalues[1])  # 0.15189 > 0.05 이므로 모델은 유의하지 않다

plt.scatter(iris.sepal_width, iris.sepal_length)
plt.plot(iris.sepal_width, result1.predict(), color='r')
plt.show()
# 의미없는 결과가 도출되었다.

# 상관관계가 강한 경우  sepal_length, petal_length : 0.871754
result2 = smf.ols(formula='sepal_length ~ petal_length', data=iris).fit()
print('result2 모델 정보 : ', result2.summary())
print('result2 R_squared : ', result2.rsquared)  # 0.7599
print('result2 p-value : ', result2.pvalues[1])  # 1.04e-47 > 0.05 이므로 유의한 모델이다. 귀무가설 기각

plt.scatter(iris.petal_length, iris.sepal_length)
plt.plot(iris.petal_length, result2.predict(), color='r')
plt.show()

print('실제 값 : ', iris.sepal_length[:10].values)
print('예측 값 : ', result2.predict()[:10])
# 새로운 petal_length로 sepal_length를 예측 가능
new_data = pd.DataFrame({'petal_length':[1.1, 0.5, 5.0]})  # petal_length가 1.1이면, 0.5면, 5.0이면 sepal_length가 뭐가 나오는데?
y_pred = result2.predict(new_data)
print('예측결과 : ', y_pred.values)

print('다중선형회귀 : 독립변수가 복수')
# result3 = smf.ols(formula='sepal_length ~ sepal_width + petal_length + petal_width', data=iris).fit()
column_select = "+".join(iris.columns.difference(['sepal_length', 'species']))
print(column_select)
result3 = smf.ols(formula='sepal_length ~ ' + column_select, data=iris).fit()
print('result3 모델 정보 : ', result3.summary())
