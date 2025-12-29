import time

import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, GradientBoostingClassifier, \
    RandomForestClassifier
from sklearn.metrics import mean_squared_error, f1_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from RuleTree.ensemble.GBoostedTreeClassifier import GBoostedTreeClassifier
from RuleTree.ensemble.GBoostedTreeRegressor import GBoostedTreeRegressor
from RuleTree.ensemble.XGBoostedTreeRegressor import XGBoostedTreeRegressor


def _fit_pred_times(model, X_train, y_train, X_test):
    start_train = time.time()
    model.fit(X_train, y_train)
    end_train = time.time()
    start_test = time.time()
    y_pred = model.predict(X_test)
    end_test = time.time()

    return round(end_train - start_train, 4), round(end_test - start_test, 4), y_pred

def test_reg():
    df = pd.read_csv('datasets/REG/students.csv')
    df['Extracurricular Activities'] = LabelEncoder().fit_transform(df['Extracurricular Activities'])
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


    rf = RandomForestRegressor(n_estimators=100)
    train_time, pred_time, y_pred = _fit_pred_times(rf, X_train, y_train, X_test)
    print('Random Forest:', mean_squared_error(y_test, y_pred))
    print(f'Training time: {train_time}s\tPrediction time: {pred_time}s\n\n')

    sk_gb = GradientBoostingRegressor(n_estimators=100)
    train_time, pred_time, y_pred = _fit_pred_times(sk_gb, X_train, y_train, X_test)
    print('sk-learn gboost:', mean_squared_error(y_test, y_pred))
    print(f'Training time: {train_time}s\tPrediction time: {pred_time}s\n\n')

    gbr = GBoostedTreeRegressor(n_estimators=100)
    train_time, pred_time, y_pred = _fit_pred_times(gbr, X_train, y_train, X_test)
    print('RuleTree gboost:', mean_squared_error(y_test, y_pred))
    print(f'Training time: {train_time}s\tPrediction time: {pred_time}s\n\n')

    xgbr = XGBoostedTreeRegressor(n_estimators=100)
    train_time, pred_time, y_pred = _fit_pred_times(xgbr, X_train, y_train, X_test)
    print('XGBoostedTreeRegressor:', mean_squared_error(y_test, y_pred))
    print(f'Training time: {train_time}s\tPrediction time: {pred_time}s\n\n')

def test_clf():
    df = pd.read_csv('datasets/CLF/glass.csv')
    X = df.iloc[:, :-1].values
    y = LabelEncoder().fit_transform(df.iloc[:, -1].values)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)

    rf = RandomForestClassifier(n_estimators=100)
    train_time, pred_time, y_pred = _fit_pred_times(rf, X_train, y_train, X_test)
    print('Random Forest:', classification_report(y_test, y_pred), end='\t')
    print(f'Training time: {train_time}s\tPrediction time: {pred_time}s\n\n')

    sk_gb = GradientBoostingClassifier(n_estimators=100, criterion='squared_error')
    train_time, pred_time, y_pred = _fit_pred_times(sk_gb, X_train, y_train, X_test)
    print('sk-learn gboost:', classification_report(y_test, y_pred), end='\t')
    print(f'Training time: {train_time}s\tPrediction time: {pred_time}s\n\n')

    gbr = GBoostedTreeClassifier(n_estimators=100)
    train_time, pred_time, y_pred = _fit_pred_times(gbr, X_train, y_train, X_test)
    print('RuleTree gboost:', classification_report(y_test, y_pred), end='\t')
    print(f'Training time: {train_time}s\tPrediction time: {pred_time}s\n\n')

if __name__ == '__main__':
    print('Regression')
    test_reg()

    [print() for _ in range(3)]

    print('Classification')
    test_clf()
