from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import cross_val_predict
import numpy as np


def Performance_testing(model, X, y):
    train_pred = cross_val_predict(model, X, y, cv=3)
    matrix = confusion_matrix(y, train_pred)
    score = f1_score(y, train_pred, average="weighted")
    return matrix, score