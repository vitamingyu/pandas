import joblib
import pandas as pd
mymodel = joblib.load('linear6m.model')

# 예측2 : 새로운 tv, radio값으로 sales를 추정
x_new2 = pd.DataFrame({'Price':[200.0, 250.0, 170.0], 'Advertising':[37.8, 45.3, 55.0],
                       'Age':[33, 44, 39], 'Income':[35.0, 72.0, 70.0]})
new_pred2 = mymodel.predict(x_new2)
print('sales 추정값 : ', new_pred2.values)  # [ 7.05072271  4.95349361 10.92284472]