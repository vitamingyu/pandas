import numpy as np
import pandas as pd
from pandas.core.frame import DataFrame

# 문1 표준정규분포를 따르는 9 X 4 형태의 DataFrame을 생성하시오
# a에서 생성한 DataFrame의 칼럼 이름을 - No1, No2, No3, No4로 지정하시오
a = DataFrame(np.random.randn(9,4), columns=['No1','No2','No3','No4'])
print(a)
# 각 컬럼의 평균을 구하시오. mean() 함수와 axis 속성 사용
print(a.mean(axis=0))
print()

# 문2
#  DataFrame으로 자료를 만드시오. colume(열) name은 numbers, row(행) name은 a~d이고 값은 10~40
a = DataFrame([10, 20, 30, 40], columns=['numbers'], index=['a','b','c','d'])
print(a)
print()
#  c row의 값을 가져오시오
print(a.iloc[[2]])
#  a, d row들의 값을 가져오시오
print(a.iloc[[0,3]],'\n')
# numbers의 합을 구하시오.
print(a['numbers'].sum(),'\n')
#numbers의 값들을 각각 제곱하시오
print(a**2,'\n')
#floats 라는 이름의 칼럼을 추가하시오. 값은 1.5, 2.5, 3.5, 4.5    아래 결과가 나와야 함.')
a['floats']=[1.5,2.5,3.5,4.5]
print(a)

#names라는 이름의 다음과 같은 칼럼을 위의 결과에 또 추가하시오. Series 클래스 사용.')
a['names'] = pd.Series(('길동', '오정', '팔계', '오공'), index=['d','a','b','c'])
print(a)