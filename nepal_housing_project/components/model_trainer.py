from nepal_housing_project.logger import logging
from nepal_housing_project.exception import hosuingprojectException

import os,sys

from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score,mean_squared_error,root_mean_squared_error
from neuro_mf import ModelFactory

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
        Description :   This function uses neuro_mf to get the best model object and report of the best model
        
        Output      :   Returns metric artifact object and best model object
        On Failure  :   Write an exception log and then raise an exception
        """
        try:
            logging.info("Using neuro_ml to get best model object and report")
            model_factory = ModelFactory(model_config_path=self.model_trainer_config.model_config_file_path)
            
            x_train, y_train, x_test, y_test = train[:, :-1], train[:, -1], test[:, :-1], test[:, -1]
            print(train)
            best_model_detail = model_factory.get_best_model(
                X=x_train,y=y_train,base_accuracy=self.model_trainer_config.expected_accuracy
            )
            model_obj = best_model_detail.best_model
            print(model_obj)
            y_pred = model_obj.predict(x_test)

            mean_sqr_error=mean_squared_error(y_test,y_pred)
            root_mean_sqr_error=root_mean_squared_error(y_test,y_pred)
            r_2_score=r2_score(y_test,y_pred)
            metric_artifact=RegressorMetricArtifact(mean_sqr_error=mean_sqr_error,root_mean_sqr_error=root_mean_sqr_error,r_2_score=r_2_score)

            return best_model_detail,metric_artifact
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

            best_model_detial,metric_artifact=self.get_model_object_and_report(train=train_arr,test=test_arr)

            preprocessing_obj=load_object(file_path=self.data_transformation_artifact.transformed_object_file_path)

            if best_model_detial.best_score < self.model_trainer_config.expected_accuracy:
                logging.info("No best model found with the score more the base score")
                raise Exception('No best model found with the score more then base score')
        
            housing_model=HousingModel(preprocessing_object=preprocessing_obj,
                                   trained_model_object=best_model_detial.best_model)
        
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