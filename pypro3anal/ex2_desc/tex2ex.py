import numpy as np
import scipy.stats as stats
import pandas as pd


# 귀무 : 새로운 백열전구는 평균 수명이 300이다
# 대립 : 새로운 백열전구는 평균 수명이 300아니다
sample = [305, 280, 296, 313, 287, 240, 259, 266, 318, 280, 325, 295, 315, 278]

print(np.array(sample).mean())  # 289.78
print(np.array(sample).mean() - 300.0)  # -10.22

print('정규성 확인 : ', stats.shapiro(sample))  # pvalue=0.820 > 0.05 정규성 만족
result = stats.ttest_1samp(sample, 300.0)
print(result)  # TtestResult(statistic=-1.556
print('statistic(t값):%.5f, pvalue:%.5f'%result)  # statistic(t값):-1.55644, pvalue:0.14361
# 해석 : pvalue:0.14361 > 0.05 이므로 귀무가설 채택. 수집된 자료는 우연히 발생된 것이라 할 수 있다.

print('----'*30)

# 실습2)
# A중학교 1학년 1반 학생들의 시험결과가 담긴 파일을 읽어 처리 (국어 점수 평균 검정) one_sample.csv
# 귀무 : 노트북 평균 사용 시간은 5.2시간이다
# 대립 : 노트북 평균 사용 시간은 5.2시간이 아니다

data = pd.read_csv('../testdata/one_sample.csv')
print(data['time'].unique())
data['time'] = data['time'].str.replace(" ", "")
data = data[data['time'] != '']
print(data['time'].unique())

# 'time' 열을 부동 소수점으로 변환
data['time'] = data['time'].astype(float)

print(np.mean(data['time']))  # 5.556
result = stats.ttest_1samp(data['time'], 5.2)
print(result) # TtestResult(statistic=3.94605956, pvalue=0.0001416669, df=108)
print('statistic(t값):%.5f, pvalue:%.5f'%result)  # statistic(t값):3.94606, pvalue:0.00014
# 해석 : pvalue:0.00014 < 0.05 이므로 귀무가설 기각. 수집된 자료는 우연히 발생된 것이라 할 수 있다.

print('----'*30)
# 실습예제 3)
# beauty.csv

# 귀무가설 : 미용 요금 평균은 15000원이다
# 대립가설 : 미용 요금 평균은 15000원이 아니다
data2 = pd.read_excel("../testdata/beauty.xls")
data2.to_csv("beauty.csv",index=False)
data2 = pd.read_csv("beauty.csv")

# data2['세종'].fillna(data[['서울', '부산', '대구', '인천', '광주', '대전', '울산', '경기', '강원', '충북', '충남', '전북', '전남', '경북', '경남', '제주']].mean(axis=1), inplace=True)
data2 = data2.dropna(axis=1)
print(data2)
data2 = data2.drop(['번호', '품목'], axis=1)
print(np.mean(data2.T.iloc[:,0]))  # 18311.875

result3 = stats.ttest_1samp(data2.iloc[0], popmean=15000)
print(result3)
print('statistic(t값):%.5f, pvalue:%.5f'%result3)
# 해석 : pvalue : 0.00001 < 0.05 이므로 귀무가설 기각. 전국 평균 비용 요금이 15000원이 아니다
