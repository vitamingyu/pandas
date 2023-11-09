import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd


data = {
    'kind': [1, 2, 3, 4, 2, 1, 3, 4, 2, 1, 2, 3, 4, 1, 2, 1, 1, 3, 4, 2],
    'quantity': [64, 72, 68, 77, 56, np.NAN, 95, 78, 55, 91, 63, 49, 70, 80, 90, 33, 44, 55, 66, 77]
}

df = pd.DataFrame(data)
df['quantity'].fillna(df.quantity.mean(), inplace=True)
# print(df)

gr1 = df[df['kind'] == 1]['quantity']
gr2 = df[df['kind'] == 2]['quantity']
gr3 = df[df['kind'] == 3]['quantity']
gr4 = df[df['kind'] == 4]['quantity']
print(np.mean(gr1))  # 63.2543
print(np.mean(gr2))  # 68.8333
print(np.mean(gr3))  # 66.75
print(np.mean(gr4))  # 72.75

# 정규성
print(stats.shapiro(gr1).pvalue)  # 0.8680412
print(stats.shapiro(gr2).pvalue)  # 0.59239
print(stats.shapiro(gr3).pvalue)  # 0.4860
print(stats.shapiro(gr4).pvalue)  # 0.41621
print()

# 등분산성
print(stats.bartlett(gr1, gr2, gr3, gr4).pvalue)  # 0.193420

f_statistic, p_value = stats.f_oneway(gr1, gr2, gr3, gr4)
print('f_statistic:{}, p_value:{}'.format(f_statistic, p_value))
# f_statistic:0.26693511759829797, p_value:0.8482436666841788
# 귀무기각 실패, 기름의 종류에 따라 빵에 흡수된 기름의 양은 차이가 없다

# 사후 검정
tkResult = pairwise_tukeyhsd(endog=df.quantity, groups=df.kind)
print(tkResult)

tkResult.plot_simultaneous(xlabel='quantity', ylabel='kind')
# plt.show()


# 문제2
import MySQLdb
import pickle
import sys

try:
    with open('mydb.dat', mode='rb') as obj:
        config = pickle.load(obj)

except Exception as e:
    print('연결 오류 :', e)
    sys.exit()

try:
    conn = MySQLdb.connect(**config)
    cursor = conn.cursor()
    sql = """
        select buser_name, jikwon_pay
        from jikwon j join 
        buser b on b.buser_no=j.buser_num
    """
    cursor.execute(sql)

    df = pd.DataFrame(cursor.fetchall(), columns=['부서명', '연봉'])
    # print(df.isnull().sum())  # null값 있는지 확인, 근데 없네
    df.dropna(subset=['연봉'])
    print(df)
    print()
    b총 = df[df['부서명'] == '총무부']['연봉']
    b전 = df[df['부서명'] == '전산부']['연봉']
    b영 = df[df['부서명'] == '영업부']['연봉']
    b관 = df[df['부서명'] == '관리부']['연봉']

    # 정규성
    # 두 개의 표본이 같은 분포를 따르는지 정규성 확인
    print('정규성')
    print(stats.ks_2samp(gr1, gr2).pvalue)  # 0.9307 > 0.05 이므로 정규성 만족
    print(stats.ks_2samp(gr1, gr3).pvalue)  # 0.9238 > 0.05 이므로 정규성 만족
    print(stats.ks_2samp(gr1, gr4).pvalue)  # 0.5523 > 0.05 이므로 정규성 만족
    print(stats.ks_2samp(gr2, gr3).pvalue)  # 0.9238 > 0.05 이므로 정규성 만족
    print(stats.ks_2samp(gr2, gr4).pvalue)  # 0.5523 > 0.05 이므로 정규성 만족
    print(stats.ks_2samp(gr3, gr4).pvalue)  # 0.7714 > 0.05 이므로 정규성 만족
    print()

    # 등분산성
    print(stats.bartlett(b총, b관, b영, b전).pvalue)  # 0.629095 등분산성 만족
    print()
    print(stats.f_oneway(gr1, gr2, gr3, gr4))  # F_onewayResult(statistic=0.26693511759829797, pvalue=0.8482436666841788)
    # pvalue=0.848 > 0.05 이므로 귀무가설 기각 실패. 총무부, 영업부, 전산부, 관리부 직원의 연봉의 평균 차이가 없다.


except Exception as e:
    print('처리 오류 :', e)
finally:
    cursor.close()
    conn.close()
