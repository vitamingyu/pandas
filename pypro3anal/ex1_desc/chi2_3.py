# 이원카이제곱 검정 - 두 개 이상의 집단(범주형 변인)을 대상으로, 독립성, 동질성 검정

# 독립성(관련성)검정
# - 동일 집단의 두 변인 (학력수준과 대학진학 여부)을 대상으로 관련성이 있는가 없는가?
# - 독립성 검정은 두 변수 사이의 연관성을 검정한다.

# 실습 : 교육수준과 흡연율 간의 관련성 분석 : smoke.csv
# 귀무 : 교육수준과 흡연율 간에 관련이 없다 : 독립이다
# 대립 : 교육수준과 흡연율 간에 관련이 있다 : 독립이 아니다

import pandas as pd
import scipy.stats as stats

data1 = pd.read_csv("../testdata/smoke.csv")  # 표본자료
print(data1.head(3), data1.shape)  # 355, 2
print(data1['education'].unique())  # [1 2 3] 독립
print(data1['smoking'].unique())  # [1 2 3] 종속
#교차표 작성
ctab = pd.crosstab(index=data1['education'], columns=data1['smoking'])
ctab.index=['대학원졸','대졸','고졸']
ctab.columns=['과흡연','보통','노담']
print(ctab)
#ctab = pd.crosstab(index=data1['education'], columns=data1['smoking'], normalize=True)
#print(ctab)  # 비율로 나옴

chi2, p, dof, _ = stats.chi2_contingency(ctab)
print(f'chi2:{chi2}, p:{p}, dof:{dof}')
# chi2:18.910915739853955, p:0.0008182572832162924, dof:4
# 판정 : p가 0.05보다 작기때문에 유의한 수준(α=0.05)에서 귀무가설을 기각한다.
# 발생된 데이터는 우연히 발생된 자료가 아니다.
# 교육수준과 흡연율 간에 관련이 있다.
# 참고 : Yeats의 연속성 보정(자유도가 1일 때 보정이 필요) 근데 다 해준다는거 같기도