import h5py
import numpy as np 
import json
from collections import defaultdict
from catboost import CatBoostRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.metrics import mean_squared_error

np.random.seed(1)

def get_cindex(Y, P):
    summ = 0
    pair = 0
    
    for i in range(1, len(Y)):
        for j in range(0, i):
            if i != j:
                if Y[i] > Y[j]:
                    pair += 1
                    summ += 1 * (P[i] > P[j]) + 0.5 * (P[i] == P[j])

    if pair != 0:
        return summ / pair
    else:
        return 0

def read_data(train_Y_file, test_Y_file, feature_file):
    test_Y = json.load(open(test_Y_file))
    train_Y = json.load(open(train_Y_file))
    f = h5py.File(feature_file, 'r')   # Open h5 file                   
    train_feature = f['train_feature'][:]
    test_feature = f['test_feature'][:]
    f.close()
    return train_Y, test_Y, train_feature, test_feature

def experiment(regressor, regressor_args):
    with open(f'log0918_catboost_bagging_n800.txt', 'w') as f:
        clf = regressor(**regressor_args)
        print(f'Present regressor is {str(regressor)}', file=f)
        regr = BaggingRegressor(base_estimator=clf, n_estimators=10, random_state=1, n_jobs=1, verbose=1)
        regr.fit(train_feature, train_Y)

        print(f'params:{regr.get_params}', file=f)
        gbdt_pred = regr.predict(test_feature)

        mse = mean_squared_error(test_Y, gbdt_pred)
        ci = get_cindex(test_Y, gbdt_pred)

        print(f'MSE:{mse},CI:{ci}', file=f)

train_Y, test_Y, train_feature, test_feature = read_data(train_Y_file="train_Y.txt", test_Y_file="test_Y.txt", feature_file='DenseFeature.h5')
regressor = CatBoostRegressor
regressor_args = defaultdict(dict, {CatBoostRegressor: dict(random_state=1, iterations=100, depth=10, learning_rate=0.05)})
# You can adjust the parameters based on your problem and dataset

for reg in regressor:
    args = (reg, regressor_args[reg])
    experiment(*args)
