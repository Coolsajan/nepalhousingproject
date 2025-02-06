from nepal_housing_project.logger import logging
from nepal_housing_project.exception import hosuingprojectException
import os,sys
from typing import Tuple
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score,mean_squared_error,root_mean_squared_error
from nepal_housing_project.entity.model_selection import evaluate_model
from sklearn.tree import DecisionTreeRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import  RandomForestRegressor
from sklearn.svm import SVR

from nepal_housing_project.utils.main_utils import *
from nepal_housing_project.entity.config_entity import ModelTraningConfig
from nepal_housing_project.entity.artifact_entity import ModelTrainerArtifact,DataTransformationArtifact,RegressorMetricArtifact
from nepal_housing_project.entity.estimator import HousingModel


class ModelTrainer:
    def __init__(self,data_transformation_artifact:DataTransformationArtifact,
                 model_trainer_config:ModelTraningConfig):
    
        """
        :param data_ingestion_artifact: Output reference of data ingestion artifact stage
        :param data_transformation_config: Configuration for data transformation
        """
        self.data_transformation_artifact = data_transformation_artifact
        self.model_trainer_config = model_trainer_config

    def get_model_object_and_report(self,train:np.array,test:np.array)->Tuple[object,object]:
        """
        Method Name :   get_model_object_and_report
                
        Output      :   Returns metric artifact object and best model object
        On Failure  :   Write an exception log and then raise an exception
        """
        try:
            logging.info("Using helper function on evaluate model.")
            models={"DecisionTreeRegressor":DecisionTreeRegressor(),
                    "LIGHTboost":LGBMRegressor(random_state=44),
                    "RandomForest":RandomForestRegressor(random_state=44)
                    }
                        
            x_train, y_train, x_test, y_test = train[:, :-1], train[:, -1], test[:, :-1], test[:, -1]
            
            model_report:dict=evaluate_model(X_train=x_train, y_train= y_train,X_test=x_test,y_test=y_test,models=models)
            logging.info("Model evaluation started")

            best_model_score=max(sorted(model_report.values()))

            best_model=list(model_report.keys())[list(model_report.values()).index(best_model_score)]
            
            logging.info(f"The best model is: {best_model} with the r_2_score: {best_model_score}")
            
            metric_artifact=RegressorMetricArtifact(r_2_score=best_model_score)

            return best_model,metric_artifact
            


        except Exception as e:
            raise hosuingprojectException(e,sys) from e
        
    def initiate_model_trainer(self,)->ModelTrainerArtifact:
        logging.info("Entered initiate_model_trainer method of ModelTrainer class")
        """
        Method Name :   initiate_model_trainer
        Description :   This function initiates a model trainer steps
        
        Output      :   Returns model trainer artifact
        On Failure  :   Write an exception log and then raise an exception
        """

        try:
            train_arr=load_numpy_array_data(file_path=self.data_transformation_artifact.transformed_train_file_path)
            test_arr=load_numpy_array_data(file_path=self.data_transformation_artifact.transformed_test_file_path)

            best_model,metric_artifact=self.get_model_object_and_report(train=train_arr,test=test_arr)

            preprocessing_obj=load_object(file_path=self.data_transformation_artifact.transformed_object_file_path)
        
            housing_model=HousingModel(preprocessing_object=preprocessing_obj,
                                   trained_model_object=best_model)
        
            logging.info("Created housing model with object with preprocessor and model")
            logging.info("Created best model file path")

            save_object(self.model_trainer_config.trained_model_file_path,housing_model)

            model_trained_artifact=ModelTrainerArtifact(
                trained_model_file_path=self.model_trainer_config.trained_model_file_path,
                metric_artifact=metric_artifact
            )
            logging.info(f"Model traniner artifact: {model_trained_artifact}")
            return model_trained_artifact
        except Exception as e:
            raise hosuingprojectException(sys,e) from e