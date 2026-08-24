import time

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from RuleTree import RuleTreeRegressor
from RuleTree.stumps.regression.DecisionTreeStumpRegressorBase import DecisionTreeStumpRegressorBase
from RuleTree.stumps.regression.matrix.DecisionTreeStumpRegressorFlatMatrix import DecisionTreeStumpRegressorFlatMatrix
from RuleTree.stumps.regression.matrix.DecisionTreeStumpRegressorMatrixAB import DecisionTreeStumpRegressorMatrixAB

if __name__ == "__main__":
    df = pd.read_csv("datasets/REG/carprice.csv")
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    reg = DecisionTreeRegressor(max_depth=2, random_state=42)
    start = time.time()
    reg.fit(X_train, y_train)
    y_pred = reg.predict(X_test)
    print("Time taken to train model: ", time.time() - start)

    mse = mean_squared_error(y_test, y_pred)
    rmse = mse ** 0.5
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"MSE:  {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE:  {mae:.4f}")
    print(f"R²:   {r2:.4f}")

    reg = RuleTreeRegressor(max_depth=2,
                            base_stumps=DecisionTreeStumpRegressorBase(random_state=42),
                            random_state=42)
    start = time.time()
    reg.fit(X_train, y_train)
    y_pred = reg.predict(X_test)
    print("[Base] Time taken to train model: ", time.time() - start)

    mse = mean_squared_error(y_test, y_pred)
    rmse = mse ** 0.5
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"MSE:  {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE:  {mae:.4f}")
    print(f"R²:   {r2:.4f}")

    reg = RuleTreeRegressor(max_depth=2,
                             base_stumps=DecisionTreeStumpRegressorMatrixAB(random_state=42, batch_size=1),
                             random_state=42)
    start = time.time()
    reg.fit(X_train, y_train)
    y_pred = reg.predict(X_test)
    print("[Matrix AB] Time taken to train model: ", time.time() - start)

    mse = mean_squared_error(y_test, y_pred)
    rmse = mse ** 0.5
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"MSE:  {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE:  {mae:.4f}")
    print(f"R²:   {r2:.4f}")

    reg = RuleTreeRegressor(max_depth=2,
                             base_stumps=DecisionTreeStumpRegressorFlatMatrix(random_state=42, batch_size=1),
                             random_state=42)
    start = time.time()
    reg.fit(X_train, y_train)
    y_pred = reg.predict(X_test)
    print("[FlatMatrix] Time taken to train model: ", time.time() - start)

    mse = mean_squared_error(y_test, y_pred)
    rmse = mse ** 0.5
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"MSE:  {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE:  {mae:.4f}")
    print(f"R²:   {r2:.4f}")
