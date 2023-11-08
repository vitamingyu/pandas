# 추론 통계 분석 중 가설검정 : 독립표본 t-검정(independent two-sample t-test)
# 비교 집단이 두개인 경우 평균이 단순히 유의한 차이가 있는지를 검정
# ex) 오리온과 롯데에서 생산된 초코파이의 무게가 같은지 비교

import scipy.stats as stats
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 서로 독립인 두 집단의 평균 차이 검정(independent samples t-test)
# 남녀의 성적, A반과 B반의 키, 경기도와 충청도의 소득 따위의 서로 독립인 두 집단에서 얻은 표본을 독립표본(two sample)이라고 한다.
# 실습) 남녀 두 집단 간 파이썬 시험의 평균 차이 검정. 남녀의 시험 점수 평균이 우연히 같을 확률은?

# 귀무 : 두 집단 간 파이썬 시험의 평균에 차이가 없다.
# 대립 : 두 집단 간 파이썬 시험의 평균에 차이가 있다.

male = [75, 85, 100, 72.5, 86.5]
female = [63.2, 76, 52, 100, 70]

print(np.mean(male), np.mean(female)) # 83.8 72.24

two_sample = stats.ttest_ind(male, female, equal_var=True, alternative='two-sided')
# equal_var=True : 정규성 만족, alternative='two_sided' : 등분산성 만족
print(two_sample) # TtestResult(statistic=1.233193, pvalue=0.252507, df=8.0)
# 해석 : pvalue=0.2525 > 0.05이므로 귀무 채택

# 참고 : Effect-size(효과 크기) - 두 집단 평균 차이를 물리적으로 세 등급으로 분리해 표현
# 효과 크기 (Cohen's d) = 두 표본 집단의 평균 차이 / 추정된 표준편차
imsi = male + female
print(imsi)
print((np.mean(male) - np.mean(female))/np.std(imsi))
# 0.799328이므로 효과크기가 커서 두 그룹의 평균에 차이가 크다고 할 수 있다.

# 실습 2 : 두가지 교육 방법에 따른평균시험 점수에 대한 검정 수행 two_sample.csv
# 귀무 : 두 가지 교육방법에 따른 평균시험 점수에 차이가 없다.
# 대립 : 두 가지 교육 방법에 따른 평균 시험 점수에 차이가 있다.

data = pd.read_csv("../testdata/two_sample.csv")
print(data.head(3), data.shape) # (50, 5)
# print(data.isnull().sum()) # score : 2개의 nan값 확인됨, / 제거 또는 특정 값으로 대체 요구

ms = data[['method','score']]
print(ms.head(2))
print(ms['method'].unique())

# 교육 방법별 데이터 추출
m1 = ms[ms['method'] == 1]
m2 = ms[ms['method'] == 2]
print(m1.head(2))
print(m2.head(2))

# 교육 방법에서 점수 추출
score1 = m1['score']
score2 = m2['score']
print(score1.describe()) # 요약통계량 출력
print(score2.describe()) # 요약통계량 출력
print(score1.isnull().sum())
print(score2.isnull().sum())
# score2 = score2.fillna(0)
score1 = score1.fillna(score1.mean())
score2 = score2.fillna(score2.mean())

# 정규성 확인
import seaborn as sns
sns.histplot(score1, kde=True, color='r')  # 커널 밀도를 나타냄
sns.histplot(score2, kde=True, color='y')
plt.show()

print(stats.shapiro(score1))
print(stats.shapiro(score2))

# 등분산성
print(stats.levene(score1, score2).pvalue)   # 모수 0.4568427112977609 > 0.05 이므로 등분산성 만족
print(stats.fligner(score1, score2).pvalue)  # 모수 0.44323735267062647
print(stats.bartlett(score1, score2).pvalue) # 비모수 0.26789717886602216

result = stats.ttest_ind(score1, score2)
print('t-value : %.5f, p-value : %.5f'%result) # t-value : -0.19649, p-value : 0.84505
# 판정 : p-value > 0.05 귀무가설 채택 두가지 교육방법에 따른 평균시험 점수에 차이가 없다

print('등분산성을 만족한 경우 : ', stats.ttest_ind(score1, score2).pvalue)
print('등분산성을 만족한 경우 : ', stats.ttest_ind(score1, score2, equal_var=True).pvalue)
print('등분산성을 만족하지 못한 경우 : ', stats.ttest_ind(score1, score2, equal_var=False).pvalue)
print()
print('정규성을 만족한 경우 : ', stats.ttest_ind(score1, score2).pvalue)
print('정규성을 만족하지 못한 경우 : ', stats.wilcoxon(score1, score2).pvalue)  # 두 집단의 크기가 같은 경우
print('정규성을 만족하지 못한 경우 : ', stats.mannwhitneyu(score1, score2).pvalue)  # 두 집단의 크기가 다른 경우
