from scipy import stats
import pandas as pd
import numpy as np

# [two-sample t 검정 : 문제2]
# 아래와 같은 자료 중에서 남자와 여자를 각각 15명씩 무작위로 비복원 추출하여 혈관 내의 콜레스테롤 양에 차이가 있는지를 검정하시오.
남자 = [0.9, 2.2, 1.6, 2.8, 4.2, 3.7, 2.6, 2.9, 3.3, 1.2, 3.2, 2.7, 3.8, 4.5, 4, 2.2, 0.8, 0.5, 0.3, 5.3, 5.7, 2.3, 9.8]
여자 = [1.4, 2.7, 2.1, 1.8, 3.3, 3.2, 1.6, 1.9, 2.3, 2.5, 2.3, 1.4, 2.6, 3.5, 2.1, 6.6, 7.7, 8.8, 6.6, 6.4]

# 귀무: 남녀 두 집단 간 콜레스테롤 양에 차이가 없다
# 대립: 남녀 두 집단 간 콜레스테롤 양에 차이가 있다.
import random

male = random.sample(남자, 15)
female = random.sample(여자, 15)
print('남자의 정규성 ', stats.shapiro(male).pvalue)
print('여자의 정규성 ', stats.shapiro(female).pvalue)  # 둘 다 정규성을 만족하지 않는다.

print('남성의 콜레스테롤 평균:{}, 여성의 콜레스테롤 평균:{}'.format(np.mean(male), np.mean(female)))
two_sample = stats.wilcoxon(male, female).pvalue
print(two_sample)  # 0.096197
print('pvalue값= {}이므로 0.05보다 크기에 귀무가설 채택'.format(two_sample))
print()

# [two-sample t 검정 : 문제3]
# DB에 저장된 jikwon 테이블에서 총무부, 영업부 직원의 연봉의 평균에 차이가 존재하는지 검정하시오.
# 연봉이 없는 직원은 해당 부서의 평균연봉으로 채워준다.

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
        select jikwon_name ,buser_name,jikwon_pay
        from jikwon j join 
        buser b on b.buser_no=j.buser_num
    """
    cursor.execute(sql)

    df = pd.DataFrame(cursor.fetchall(), columns=['직원명', '부서명', '연봉'])
    jikwon = df.fillna(df['연봉'].mean())
    print(jikwon)
    print()
    총무부직원들 = jikwon[jikwon['부서명'] == '총무부']
    영업부직원들 = jikwon[jikwon['부서명'] == '영업부']
    print(총무부직원들)
    print()
    print(영업부직원들)
    print()

    pay1 = 총무부직원들['연봉']
    pay2 = 영업부직원들['연봉']

    # 귀무가설: 총무부, 영업부 직원의 연봉의 평균에 차이가 존재하지 않는다.
    # 대립가설: 총무부, 영업부 직원의 연봉의 평균에 차이가 존재한다.

    print(np.mean(pay1), np.mean(pay2))
    print()

    import seaborn as sns

    print('정규성 확인')
    print(stats.shapiro(pay1).pvalue)  # 0.02604489028453827 < 0.05이므로 정규성 불만족
    print(stats.shapiro(pay2).pvalue)  # 0.025608452036976814 <0.05이므로 정규성 불만족

    print('정규성을 만족하지 못했기에, 비모수적 방법 사용 :', stats.mannwhitneyu(pay1, pay2).pvalue)

    if stats.mannwhitneyu(pay1, pay2).pvalue > 0.05:
        print('pvalue값= {}이므로 0.05보다 크기에 귀무가설(총무부, 영업부 직원의 연봉의 평균에 차이가 존재하지 않는다.) 채택'.format(
            stats.mannwhitneyu(pay1, pay2).pvalue))
    else:
        print('pvalue값= {}이므로 0.05보다 작기에 대립가설(총무부, 영업부 직원의 연봉의 평균에 차이가 존재한다.) 채택'.format(
            stats.mannwhitneyu(pay1, pay2).pvalue))

    print()

except Exception as e:
    print('처리 오류 :', e)
finally:
    cursor.close()
    conn.close()

print()
# [대응표본 t 검정 : 문제4]
# 어느 학급의 교사는 매년 학기 내 치뤄지는 시험성적의 결과가 실력의 차이없이 비슷하게 유지되고 있다고 말하고 있다.
# 이 때, 올해의 해당 학급의 중간고사 성적과 기말고사 성적은 다음과 같다. 점수는 학생 번호 순으로 배열되어 있다.
#    중간 : 80, 75, 85, 50, 60, 75, 45, 70, 90, 95, 85, 80
#    기말 : 90, 70, 90, 65, 80, 85, 65, 75, 80, 90, 95, 95
# 그렇다면 이 학급의 학업능력이 변화했다고 이야기 할 수 있는가?

중간 = [80, 75, 85, 50, 60, 75, 45, 70, 90, 95, 85, 80]
기말 = [90, 70, 90, 65, 80, 85, 65, 75, 80, 90, 95, 95]

# 귀무가설: 중간고사 성적과 기말고사에 기반하여 이 학급의 학업능력은 변화가 없다
# 대립가설: 중간고사 성적과 기말고사에 기반하여 이 학급의 학업능력은 변화했다

print(np.mean(중간))
print(np.mean(기말))
print()

print('정규성 확인')
print(stats.shapiro(중간).pvalue)  # 0.368144154548645> 0.05이므로 정규성 만족
print(stats.shapiro(기말).pvalue)  # 0.19300280511379242>0.05이므로 정규성 만족

pair_sample = stats.ttest_rel(중간, 기말)
print('t value : %.5f, p-value : %.5f' % pair_sample)

if pair_sample.pvalue > 0.05:
    print('pvalue값= {}이므로 0.05보다 크기에 귀무가설(중간고사 성적과 기말고사에 기반하여 이 학급의 학업능력은 변화없다.) 채택'.format(
        stats.mannwhitneyu(pay1, pay2).pvalue))
else:
    print('pvalue값= {}이므로 0.05보다 작기에 대립가설(중간고사 성적과 기말고사에 기반하여 이 학급의 학업능력은 변화했다.) 채택'.format(
        stats.mannwhitneyu(pay1, pay2).pvalue))





