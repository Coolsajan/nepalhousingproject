from nepal_housing_project.logger import logging
from nepal_housing_project.exception import hosuingprojectException
import sys

from sklearn.model_selection import GridSearchCV

from sklearn.metrics import r2_score



def evaluate_model(X_train,y_train,X_test,y_test,models):
    try:
        report={}
        logging.info("Model Selection started")
        for i in range(len(models)):
            model=list(models.values())[i]

            model.fit(X_train,y_train)

            y_pred=model.predict(X_test)

            accuracy_score=r2_score(y_pred,y_test)

            report[list(models.keys())[i]]=accuracy_score
        
        return report
    except Exception as e:
        raise hosuingprojectException(e,sys)





        
