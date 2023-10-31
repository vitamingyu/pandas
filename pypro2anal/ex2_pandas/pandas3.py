# DataFrame : reshape, cut, merge, pivot
import numpy as np
import pandas as pd

df = pd.DataFrame(1000 + np.arange(6).reshape(2,3), index=['대전','서울'],columns=['2021', '2022',' 2023'])
print(df)
print()
df_row = df.stack()  # 재구조화 열 -> 행으로 변환
print(df_row)
print()
df_col = df_row.unstack()  # 행 -> 열로 변환 (스택을 원복)
print(df_col)
print()

print('범주화 : 연속형 자료를 범주형으로 변경')
price = [10.3, 5.5, 7.8, 3.6]
cut = [3, 7, 9, 11]  # 구간 기준값
result_cut = pd.cut(price, cut)
print(result_cut)
#  [(9, 11], ... 9초과 11이하, ...
print()
print(pd.value_counts(result_cut))
print()
datas = pd.Series(np.arange(1, 1001))
print(datas.head(3))
print(datas.tail(3))
print()
result_cut2 =pd.qcut(datas, 3)  # datas 값을 3개영역으로 범주화
print(result_cut2) 
print()
print(pd.value_counts(result_cut2))
print()
cut2 = [1, 500, 1000]
result_cut3 = pd.cut(datas,cut2)
print(result_cut3)
print()
print(pd.value_counts(result_cut3))  # 구간 나눠! qcut, 범위 지정 나눠! cut
print()

print('그룹별 함수 수행 : agg, apply')
group_col = datas.groupby(result_cut2)
print(group_col.agg(['count', 'mean', 'std', 'max']))
print()

# agg 대신 함수 직접 작성
def summary_func(gr):
    return{
        'count':gr.count(),
        'mean':gr.mean(),
        'std':gr.std(),
        'max':gr.max(),
    }
print(group_col.apply(summary_func))  # 스택 구조로 나오네
print('아래는 언 스택으로 바꿈')
print(group_col.apply(summary_func).unstack())















