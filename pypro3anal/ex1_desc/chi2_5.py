# 이원카이제곱 검정
import pandas as pd
import scipy.stats

# 독립성 검정
data = pd.read_csv("https://raw.githubusercontent.com/pykwon/python/master/testdata_utf8/home_task.txt",\
                   sep='\t', index_col=0)  # unnamed삭제
print(data.head(3))

# 귀무(영가설, H0) : 집안일의 종류와 일하는 사람은 관계가 없다.(독립적이다)
# 대립(연구가설, 대안가설, H1) : 집안일의 종류와 일하는 사람은 관계가 있다.(독립적이지 않다)

chi2, pvalue, _, _ = scipy.stats.chi2_contingency(data)
print(f'chi2:{chi2}, pvale:{pvalue}')
# chi2:1364.5404438935336, pvale:1.8759478966116962e-273
# 해석 : 귀무가설 기각

# 동질성 검정(동질이나 독립이나 똑같음. 표현방법만 다르지). 빈도수냐 비율값이냐, 동질성은 비율로 따짐
# chi2_6에 만듦
