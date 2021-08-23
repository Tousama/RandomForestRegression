
#%%
import pandas as pd
import numpy as np
import math
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder


label_encoder=LabelEncoder()

df=pd.read_csv('train.csv')
del df['ADDRESS']

a=df['POSTED_BY'].dtype

for i in df:
    if df[i].dtype==a:
        df[i]=label_encoder.fit_transform(df[i])
        
        

df1=df.copy()
del df1['TARGET(PRICE_IN_LACS)']
df2=df['TARGET(PRICE_IN_LACS)']
def train_val_test(df):
    return df[:20600], df[20600:25025], df[25025:]

x_train, x_val, x_test=train_val_test(df1)
y_train, y_val, y_test=train_val_test(df2)



def rmse(x,y):
    return math.sqrt(((x-y)**2).mean())

def print_score(m):
    print(f"RMSE of train set {rmse(m.predict(x_train),y_train)}")
    print(f"R^2 of train set {m.score(x_train,y_train)}")
    print(f"RMSE of validation set {rmse(m.predict(x_val),y_val)}")
    print(f"R^2 of validation set {m.score(x_val,y_val)}")


scores=[]
for i in range(1,200,5):
    m=RandomForestRegressor(n_estimators=i, n_jobs=-1)
    %time m.fit(x_train, y_train)
    scores.append(m.score(x_val,y_val))

best_estimator=np.argmax(scores)*5+1
m=RandomForestRegressor(n_estimators=best_estimator, n_jobs=-1)
m.fit(x_train, y_train)
print_score(m)


print('RMSE of test set: ',rmse(m.predict(x_test),y_test))
print('R^2 of test set: ',m.score(x_test,y_test))



