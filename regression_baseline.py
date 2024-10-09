import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import lightgbm as lgb
import random
import time
import csv
import os
import stats

def mape(y_true, y_pred):
    return np.mean( np.abs(y_pred - y_true) / np.abs(y_true))

def linear(X_train, y_train, X_test):
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model.predict(X_test)

def lightgbm(X_train, y_train, X_test):
    train_data = lgb.Dataset(X_train, label=y_train)
    params = {
    'objective': 'regression',  # For regression tasks
    'metric': 'mape',           # Root Mean Squared Error
    'boosting_type': 'gbdt',    # Gradient Boosting Decision Tree
    'learning_rate': 0.1,       # Learning rate
    'num_leaves': 31,           # Number of leaves in one tree
    'verbose': -1               # Suppress warning messages
    }
    gbm = lgb.train(params, train_data, num_boost_round=100)
    return gbm.predict(X_test, num_iteration=gbm.best_iteration)

def export(df_y, results, time, dataset, algo):
    try:
        os.mkdir(dataset.replace('data', 'results')[:-4])
    except:
        pass

    with open(f'{dataset.replace('data', 'results')[:-4]}/{algo}-{dataset.split('/')[-1]}', 'w', newline='') as f:    
        write = csv.writer(f)
        df_y.append('Time')
        write.writerow([y for y in df_y])
        for r in results:
            write.writerow(r)
        ll = (len(df_y)-1) * ['']
        ll.append(time)
        write.writerow(ll)

def calc_baseline(dataset, repeats):
    st = time.time()
    results = []
    lrstat  = stats.SOME(txt='LR')
    lgbmstat  = stats.SOME(txt='LGBM')
    for _ in range(5):
        df = pd.read_csv(dataset)
        df = df.sample(frac=1)

        df_y = [c for c in df.columns if (c[-1] == '-' or c[-1] == '+')]
        y = df[df_y]
        X = df[[c for c in df.columns if (c[-1] != '-' and c[-1] != '+' and c[-1] != 'X')]]
        X = pd.get_dummies(df, columns=[c for c in df.columns if c[0].islower], drop_first=False)

        res = []
        #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=int(100*random.random()))
        
        for n1 in range(5):
            lo = len(X)//5 * n1
            hi = len(X)//5 * (n1+1)
            X_test = X.iloc[lo:hi]
            y_test = y.iloc[lo:hi]

            X_train = pd.concat([X.iloc[:lo], X.iloc[hi:]])
            y_train = pd.concat([y.iloc[:lo], y.iloc[hi:]])

            for target_column in df_y:
                y_pred_lr = linear( X_train, y_train[target_column], X_test)
                y_pred_lgbm = lightgbm( X_train, y_train[target_column], X_test)
                #res.append(round(mape(y_test[target_column], y_pred),3))
                sdv = y_train[target_column].std()
                for at in range(len(y_test)):
                    lrstat.add((y_test[target_column].iloc[at] - y_pred_lr[at])/sdv)
                    lgbmstat.add((y_test[target_column].iloc[at] - y_pred_lgbm[at])/sdv)
        
        #results.append(res)
    return [lrstat,lgbmstat]
    #export(df_y, results , round(time.time()-st,2), dataset, algo)


#print("started.")
#calc_baseline('linear', 'data/hpo/healthCloseIsses12mths0001-hard.csv')
#calc_baseline('lightgbm', 'data/hpo/healthCloseIsses12mths0001-hard.csv')