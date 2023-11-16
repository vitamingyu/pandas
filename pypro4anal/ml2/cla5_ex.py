# [로지스틱 분류분석 문제3]
# Kaggle.com의 https://www.kaggle.com/truesight/advertisingcsv  file을 사용
# 참여 칼럼 :
#   Daily Time Spent on Site : 사이트 이용 시간 (분)
#   Age : 나이,
#   Area Income : 지역 소독,
#   Daily Internet Usage:일별 인터넷 사용량(분),
#   Clicked Ad : 광고 클릭 여부 ( 0 : 클릭x , 1 : 클릭o )
# 광고를 클릭('Clicked on Ad')할 가능성이 높은 사용자 분류.
# 데이터 간의 단위가 큰 경우 표준화 작업을 시도한다.
# ROC 커브와 AUC 출력

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('../testdata/advertisement.csv')
df = pd.DataFrame(df, columns=(['Daily Time Spent on Site', 'Age', 'Area Income', 'Daily Internet Usage', 'Clicked on Ad']))
print(df.head(5), df.shape)  #(1000, 5)


x = df.iloc[:, 0:4]
y = df['Clicked on Ad']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)  # (700, 4) (300, 4) (700,) (300,)


# scaling(크기를 고르게) - feature에 대한 표준화, 정규화 : 최적화 과정에서 안정성, 수련 속도를 향상, 오버피팅 or 언더피팅 방지 기능
# print(x_train[:4])
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

# sc.fit(x_train)
# sc.fit(x_test)
# x_train = sc.transform(x_train)
# x_test = sc.transform(x_test)  fit을 2번하고 있음 위처럼 2줄로 끝내도록

model = LogisticRegression(C=0.001, solver='lbfgs', multi_class='auto', random_state=0, verbose=0)
model.fit(x_train, y_train)

# 분류 예측
y_pred = model.predict(x_test)
# print('예측값 : ', y_pred)
# print('실제값 : ', y_test.values)
print('총 갯수 : %d, 오류 수 : %d'%(len(y_test), (y_test != y_pred).sum()))
# 총 갯수 : 300, 오류 수 : 18
print('분류 정확도 확인 1')
print('%.5f'%accuracy_score(y_test, y_pred))  # 0.94000

print('분류 정확도 확인 3')
print('test로 정확도는 ', model.score(x_test, y_test))  # 0.94
print('train으로 정확도는 ', model.score(x_train, y_train))  # 0.9471


from sklearn import metrics
fpr, tpr, thresholds = metrics.roc_curve(y_test, model.decision_function(x_test))
# confusion matrix의 값들 보기
cl_rep = metrics.classification_report(y_test, y_pred)
print(cl_rep)

import matplotlib.pyplot as plt
plt.plot(fpr, tpr, 'o-', label='LogisticRegression')
# 아래는 옵션으로 줌 (필수아님)
plt.plot([0, 1], [0,1], 'k--', label='Landom Classifier Line(AUC: 0.5지점)')
# plt.plot([fallout], [recall], 'ro', ms = 10)
# -----
plt.xlabel('fpr', fontdict={'fontsize':15})
plt.ylabel('tpr', fontdict={'fontsize':15})
plt.legend()
plt.show()

# AUC : 성능평가에 있어서 수치적인 기준이 될 수 없는 값으로,
# 1에 가까울수록 그래프가 좌상단에 근접하게 되므로 좋은 모델이라 할 수 있다
print('AUC : ', metrics.auc(fpr, tpr))  # 0.9896
print('테스트 정확도: ', model.score(x_test, y_test))  # 0.94
