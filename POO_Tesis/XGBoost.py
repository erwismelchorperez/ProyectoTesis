from audioop import cross
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import xgboost as xgb
from xgboost import XGBRegressor
from xgboost import XGBClassifier
from datetime import datetime
import numpy as np
class XGBoost_:
    def __init__(self, dataset):
        self.param_grid = {'max_depth':2, 'eta':1, 'objective':'binary:logistic' }
        self.dataset = dataset
        self.num_round = 2
        self.bts = None
        self.dtrain = xgb.DMatrix(self.dataset.get_xtrain(),self.dataset.get_ytrain())
        self.dvalidation = xgb.DMatrix(self.dataset.get_xvalidation(),self.dataset.get_yvalidation())
        self.f = open("logger/XGBoost"+str(datetime.now())+".txt",'w')

    def Regression(self):
        scores = cross_val_score(XGBRegressor(objective='reg:squarederror'), self.dataset.get_xtrain(), self.dataset.get_ytrain(), scoring='neg_mean_squared_error')
        print((-scores)**0.5)
        self.f.write("\nXGBoostRegression \n")
        self.f.write("scores:    " + str((-scores)**0.5))

    def Classification(self):
        scores = cross_val_score(XGBClassifier(), self.dataset.get_xtrain(), self.dataset.get_ytrain())
        print("Score Validación cruzada:     ",np.average(scores))
        self.f.write("\nXGBoostClassifier validation_cross\n")
        self.f.write("scores:   " + str(scores))
        self.f.write("Promeido de validación cruzada:   " + str(np.average(scores)))

        scores = cross_val_predict(XGBClassifier(), self.dataset.get_xtrain(), self.dataset.get_ytrain())
        self.f.write("\nXGBoostClassifier validation_predict\n")
        self.f.write("MatrixConfusion:   " + str(confusion_matrix(self.dataset.get_ytrain(), scores)))


    def ClassifierXGBClassifier(self):
        #xbg = xgb.XGBClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0)
        #xbg = xgb.XGBClassifier()#mayor rendimiento
        xbg = xgb.XGBClassifier(n_estimators=100, learning_rate=0.02)
        xbg.fit(self.dataset.get_xtrain(),self.dataset.get_ytrain())
        predictions = xbg.predict(self.dataset.get_xtest())
        scores = accuracy_score(self.dataset.get_ytest(),predictions)
        print("accuracy:     ", scores)
        self.f.write("\nXGBoostClassifier Predict \n")
        self.f.write("scores:   " + str(scores))

    def CloseArchivo(self):
        self.f.close()
