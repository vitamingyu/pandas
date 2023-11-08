# 추론 통계 분석 중 가설검정 : 단일 표본 t-검정(one-sample t-test)
# 정규분포의(모집단) 표본에 대해 기댓값을 조사(평균차이 사용)하는 검정방법
# 예) 새우깡 과자 무게가 진짜 120그램이 맞는가?

# 실습1) 어느 남성 집단의 평균키 검정
# 귀무 : 남성의 평균 키는 177.0 (모집단의 평균)이다.
# 대립 : 남성의 평균 키는 177.0이 아니다. (양측 검정)

import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

one_sample = [167.0, 182.7, 169.6, 176.8, 185.0]
# plt.boxplot(one_sample)
# plt.xlabel('data')
# plt.ylabel('height')
# plt.grid()
# plt.show()

print(np.array(one_sample).mean())  # 176.21
print(np.array(one_sample).mean() - 177.0)  # -0.78

print('정규성 확인 : ', stats.shapiro(one_sample))  # pvalue=0.54 > 0.05 정규성 만족
result = stats.ttest_1samp(one_sample, 177.0)
print(result)  # TtestResult(statistic=-0.22139444
print('statistic(t값):%.5f, pvalue:%.5f'%result)  # statistic(t값):-0.22139, pvalue:0.83563
# 해석 : pvalue:0.83563 > 0.05 이므로 귀무가설 채택. 수집된 자료는 우연히 발생된 것이라 할 수 있다.

print('----'*30)
# 실습2)
# A중학교 1학년 1반 학생들의 시험결과가 담긴 파일을 읽어 처리 (국어 점수 평균 검정) student.csv
# 귀무 : 학생들의 국어 점수 평균은 80.0이다.
# 대립 : 학생들의 국어 점수 평균은 80.0이 아니다.

data = pd.read_csv('../testdata/student.csv')
print(data.head(3))
print(data.describe())
print(np.mean(data.국어))  # 72.9

result2 = stats.ttest_1samp(data.국어, 80.0)
print(result2)  # TtestResult(statistic=-1.332180
print('statistic(t값):%.5f, pvalue:%.5f'%result2)  # statistic(t값):-1.33218, pvalue:0.19856
# 해석 : pvalue:0.1985 > 0.05 이므로 귀무가설 채택. 수집된 자료는 우연히 발생된 것이라 할 수 있다.

print('*'*50)
# 실습예제 3)
# 여아 신생아 몸무게의 평균 검정 수행 babyboom.csv
# 여아 신생아의 몸무게는 평균이 2800(g)으로 알려져 왔으나 이보다 더 크다는 주장이 나왔다.
# 표본으로 여아 18명을 뽑아 체중을 측정하였다고 할 때 새로운 주장이 맞는지 검정해 보자.

# 귀무가설 : 여아 신생아의 몸무게는 평균이 2800g이다.
# 대립가설 : 여아 신생아의 몸무게는 평균이 2800g보다 크다

data2 = pd.read_csv('../testdata/babyboom.csv')
print(data2.head(3), len(data2))  # 44
fdata = data2[data2['gender'] == 1]
print(fdata.head(3), len(fdata))  # 18
print(np.mean(fdata['weight']))

# 정규성 확인
print(stats.shapiro(fdata.iloc[:, 2]))  # pvalue=0.01798 < 0.05 이므로 정규성 만족 못함

# 정규성 확인 시각화 1
stats.probplot(fdata.iloc[:, 2], plot=plt)  # q-q plot
plt.show()

# 정규성 확인 시각화 2: histogram
sns.displot(fdata.iloc[:, 2], kde=True)
plt.show()

result3 = stats.ttest_1samp(fdata.weight, 2800)
print(result3)
print('statistic(t값):%.5f, pvalue:%.5f'%result3)
# 해석 : pvalue : 0.03927 < 0.05 이므로 귀무가설 기각. 여아 신생아의 뭄무게 평균은 2800보다 크다