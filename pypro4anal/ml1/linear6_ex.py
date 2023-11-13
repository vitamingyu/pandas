import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import statsmodels.api
import numpy as np
import seaborn as sns
plt.rc('font', family='malgun gothic')

# Sales 변수에 영향을 주는 변수들을 선택하여 선형회귀분석을 실시한다.
# 변수 선택은 모델.summary() 함수를 활용하여 타당한 변수만 임의적으로 선택한다.
# 회귀분석모형의 적절성을 위한 조건도 체크하시오.
# 완성된 모델로 Sales를 예측.

url = 'https://raw.githubusercontent.com/pykwon/python/master/testdata_utf8/Carseats.csv'
data = pd.read_csv(url, usecols=[0,2,3,5,7])
# print(data.head(3), data.shape)  # (400, 8)
# print(data.info()) object인 애 버려
# print(data.summary())
# print(data.columns)
# print(data[['ShelveLoc', 'Age', 'Education', 'Urban', 'US']])
# print(data.loc[:,['Sales','CompPrice','Income','Advertising','Population']].corr())
# print(data.loc[:, ['Sales','Price','Age','Education']].corr())

# 다중 선형 회귀 모델
# print('r : \n', data.corr())  # 0.782224
# lm_mul = smf.ols(formula='Sales ~ Income+Advertising+Price+Age + CompPrice + Population + Education', data = data).fit()  # Population : 0.857, Education : 0.282 제외
lm_mul = smf.ols(formula='Sales ~ Income + Advertising + Price + Age', data = data).fit()
print(lm_mul.summary())  # Prob (F-statistic):  1.33e-38, Adj.R-squared: 0.364

# 단순 선형 회귀 모델
print(lm_mul.params)
print(lm_mul.pvalues[1])  # 0.00802
print(lm_mul.rsquared)  # 0.364 36%정도 설명

# 잔차 먼저 구해
print('잔차항 구하기')
fitted = lm_mul.predict(data.iloc[:, 0:5])
print(fitted)
residual = data['Sales'] - fitted  # 실제값 - 예측값 꼴로 적은 것임, 뜻하는 바는 얘가 곧 잔차
print('\nresidual : ', residual, '\n',sum(residual))  #  -5.141664871644025e-12 0에 근접함 그래서 제곱해준다는데


# . 선형성 : 독립변수(feature)의 변화에 따라 종속변수도 일정 크기로 변화해야 한다. => 예측값과 잔차가 비슷하게 유지
sns.regplot(x=fitted, y=residual, lowess=True, line_kws={'color':'red'})  # lowess=True:비모수적 최적모델 추정(로컬 가증 선형회귀)
plt.plot([fitted.min(), fitted.max()],[0, 0], '--', color='blue')  # 리밋값 주고 x값 y값 주고
plt.show()  # 이정도면 평평하게 가고 있다고 봐 만족


# . 정규성 : 잔차항(오차항)이 정규분포를 따라야 한다. q_q plot을 사용
import scipy.stats
ssz = scipy.stats.zscore(residual)
(x, y), _ = scipy.stats.probplot(ssz)
sns.scatterplot(x=x, y=y)
plt.plot([-3,3],[-3,3], '--', color='blue')
plt.show()
print('정규성 : ', scipy.stats.shapiro(residual))  # pvalue=0.21268504858016968 정규성 만족


# . 독립성 : 독립변수의 값이 서로 관련되지 않아야 한다. 잔차가 자기상관(인접 관측치와 독립이어야 함)이 있는지 확인 필요
# 자기상관은 Durbin-Watson 지수 d를 이용하여 검정한다
# d 값은 0~4 사이에 나오며 2에 가까울수록 자기상관이 없이 독립이며, 독립인 경우 회귀 분석을 사용할 수 있다.
# DW 값이 0 또는 4에 근사하면 잔차들이 자기상관이 있고, 계수(t, f, R^2) 값을 증가시켜(그러면 p값이 작아지니) 유의하지 않은 결과를 유의한 결과로 왜곡시킬 수 있다
print('Durbin-watson:', 1.931)  # 독립성 만족


# . 등분산성 : 그룹간의 분산이 유사해야 한다. 독립변수의 모든 값에 대한 오차들의 분산은 일정해야 한다.
# 분산은 모든 잔차에 대해 동일해야 한다. 잔차(y)축 및 예상 값(x축)의 산점도를 사용하여 이 가정을 테스트할 수 있다.
# 결과 산점도는 플롯에서 임의로 플롯된 점의 수평 밴드로 나타나야 한다.
sns.regplot(x=fitted, y=np.sqrt(np.abs(ssz)), lowess=True, line_kws={'color':'red'})
plt.show()  # 적색 실선이 수평선을 그림


# . 다중공선성 : 다중회귀 분석 시 두 개 이상의 독립변수 간에 강한 상관관계가 있어서는 안된다.
# VIF(Variance Inflation Factor, 분산 팽창 지수)
# VIF는 예측변수들이 상관성이 있을 때 추정회귀 계수의 산포 크기를 측정하는 것이며, 산포가 커질수록 회귀모형은 신뢰할 수 없게 된다.
# VIF 값이 1 근방에 있으면 다중공선성이 없어 모형을 신뢰할 수 있으며 만약 VIF 값이 10 이상이 되면 매우 높은 다중공선성이 있기 때문에 변수 선택을 신중히 고려해야 합니다.
from statsmodels.stats.outliers_influence import variance_inflation_factor
# lm_mul = smf.ols(formula='sales ~ tv + radio', data = advdf).fit() 이거 위에서 구해둠
# print(variance_inflation_factor(data.values, 4))
vifdf = pd.DataFrame()
vifdf['Variable'] = data.columns[1:]  # 첫 번째 열은 종속변수이므로 제외
vifdf['VIF'] = [variance_inflation_factor(data.values, i) for i in range(1, data.shape[1])]

print(vifdf)
# Variable        VIF
# 0       Income   6.959300
# 1  Advertising   2.194469
# 2        Price  10.333421
# 3          Age   8.506083 Price는 10이 넘어 다중공산성이 있어 보임  -> Price는 독립변수로 써야할지 고민을 해볼 필요가 있음. 영향력이 큰 변수라면 고민이 필요
# 독립변수가 더 많은 경우, 예를 들어 남편의 수입과 아내의 수입이 서로 상관성이 높다면 두 개의 변수를 더해 가족 수업이라는 새로운 변수를 작성하거나
# 주성분분석을 이용하여 하나의 변수로 만들어 작업할 수 있다.

print("참고 : Cook's distance - 극단값을 나타내는 지표 이해 ---")
from statsmodels.stats.outliers_influence import OLSInfluence
cd, _ = OLSInfluence(lm_mul).cooks_distance  # 극단값을 나타내는 지표를 반환
print(cd.sort_values(ascending=False).head())

statsmodels.api.graphics.influence_plot(lm_mul, criterion='cooks')
plt.show()  # 원의 크기가 특별히 큰 데이터는 이상값(outlinear)이라 볼 수 있다.

# 모델 검증이 끝난 경우 모델을 저장
# import pickle
# with open('linear6m.model', 'wb') as obj:
#     pickle.dump(lm_mul, obj)
# # 위는 저장
#
# # 아래는 읽어옴
# with open('linear6m.model', 'rb') as obj:
#     mymodel = pickle.load(obj)  # mymodel이라는 이름으로 불러오기

# 방법2 메모리를 덜 잡아먹어 속도가 더 빠름
import joblib
joblib.dump(lm_mul, 'linear6m.model') # 확장자는 내가 정하는거임 정해진거 없음. 저장
mymodel = joblib.load('linear6m.model')

# 예측2 : 새로운 tv, radio값으로 sales를 추정
x_new2 = pd.DataFrame({'Price':[200.0, 250.0, 170.0], 'Advertising':[37.8, 45.3, 55.0],
                       'Age':[33, 44, 39], 'Income':[35.0, 72.0, 70.0]})
new_pred2 = mymodel.predict(x_new2)
print('sales 추정값 : ', new_pred2.values)  #[ 7.05072271  4.95349361 10.92284472]