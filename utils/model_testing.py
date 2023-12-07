from sklearn.metrics import r2_score, mean_squared_error
import numpy as np
def test_score(model, X_test, y_test):
    y_pred = model.predict(X_test)
    mse = np.sqrt(mean_squared_error(y_test,y_pred))
    r2s = r2_score(y_test, y_pred)
    print(f"The mean square error is {mse} and the r2 score is {r2s}.")