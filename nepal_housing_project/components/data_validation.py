import json
import sys

from nepal_housing_project.exception import hosuingprojectException
from nepal_housing_project.logger import logging

from pandas import DataFrame
import pandas as pd
from nepal_housing_project.utils.main_utils import read_yaml_file,write_yaml_file
from nepal_housing_project.entity.artifact_entity import DataingestionArtifact,DataValidationArtifact
from nepal_housing_project.entity.config_entity import DataValidationConfig
from nepal_housing_project.constants import SCHEMA_FILE_PATH

from evidently.model_profile import Profile
from evidently.model_profile.sections import DataDriftProfileSection


class DataValidation:
    def __init__(self,data_ingestion_artifact:DataingestionArtifact,data_validation_config:DataValidationConfig):
        """
        :param data_igenstion_artifact:Output refrence of data ingestion artifact stage
        :parm data_validation_config:configuration for data validation
        """
        try:
            self.data_ingestion_artifact=data_ingestion_artifact
            self.data_validation_config=data_validation_config
            self._schema_config=read_yaml_file(file_path=SCHEMA_FILE_PATH)
        except Exception as e:
            raise hosuingprojectException(e,sys)

    def validate_no_of_columns(self,df:DataFrame) ->bool:
        """
        Method Name : is_column_exist
        Description : This menthod validates the existence of a numerical and categorical columns

        Output : Return bool value on validation result
        On Falier : Write an exception log and the raise an exception
        """
        try:
            dataframe_columns=df.columns
            missing_numerical_columns=[]
            missing_categorical_columns=[]
            for column in self._schema_config['numerical_columns']:
                if column not in dataframe_columns:
                    missing_numerical_columns.append(column)

            if len(missing_categorical_columns)>0:
                logging.info(f"Missing numerical column: {missing_numerical_columns}")

            for column in self._schema_config['categorical_columns:']:
                if column not in dataframe_columns:
                    missing_categorical_columns.append(column)

            if len(missing_categorical_columns)>0:
                logging.info(f"Missing categorical column: {missing_categorical_columns}")

            return False if len(missing_categorical_columns)>0 or len(missing_numerical_columns)>0 else True
        except Exception as e:
            raise hosuingprojectException(e,sys) from e
            
    @staticmethod
    def read_data(file_path)->DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise hosuingprojectException(e,sys)
        
    def detect_dataset_drift(self,reference_df:DataFrame,current_df:DataFrame)-> bool:
        """
        Method Name : detect_dataset_drift
        Description : This method validates if drift is deteced

        Output : Return bool value based on validation results
        on failuers : Write an exception log and then raise an exception    
        """
        try: 
            data_drift_profile=Profile(selection=[DataDriftProfileSection()])
            
            data_drift_profile.calculate(reference_df,current_df)

            report=data_drift_profile.json()
            json_report=json.loads(report)

            write_yaml_file(file_path=self.data_validation_config.data_drift_flie_path)

            n_features=json_report['data_drift']['data']['metrics']['n_features']
            n_drifted_feature=json_report['data_drift']['data']['metrics']['n_drifted_features']

            logging.info(f"{n_drifted_feature}/{n_features} drift detected")
            drift_status=json_report['data_drift']['data']['metrics']['dataset_drift']
            return drift_status
        except Exception as e:
            raise hosuingprojectException(e,sys) from e
        
    def initiate_data_validation(self) -> DataValidationArtifact:
        """
        Method Name : initiate_data_validation
        Description : This menthod will initiate data valiation component from the pipeline

        Output : Return bool value based on validation results
        On failure : Write an exception log and then raise an exception        
        """
        try:
            validation_error_message=""
            logging.info("Starting of validation.")
            train_df,test_df=(DataValidation.read_data(file_path=self.data_ingestion_artifact.trained_file_path),
                              DataValidation.read_data(file_path=self.data_ingestion_artifact.test_file_path))
            
            status=self.validate_no_of_columns(df=train_df)
            logging.info(f"All reuired columns presents in tranning dataframe:{status}")
            if not status:
                validation_error_message += f"Columns are missing in tranning dataframe."
            status=self.validate_no_of_columns(df=test_df)

            logging.info(f"All required columns prosent in testing dataframe: {status}")
            if not status:
                validation_error_message+=f"Columns are missing inthe test dataframe"

            statuts=self.is_column_exist(df=train_df)

            if not status:
                validation_error_message+=f"Columns are missing in test dataframe."
                validation_status=len(validation_error_message)==0

            if validation_status:
                drift_status = self.detect_dataset_drift(train_df, test_df)
                if drift_status:
                    logging.info(f"Drift detected.")
                    validation_error_message = "Drift detected"
                else:
                    validation_error_message = "Drift not detected"
            else:
                logging.info(f"Validation_error: {validation_error_message}")

            data_validation_artifact = DataValidationArtifact(
                validation_status=validation_status,
                message=validation_error_message,
                drift_report_file_path=self.data_validation_config.drift_report_file_path
            )

            logging.info(f"Data validation artifact: {data_validation_artifact}")
            return data_validation_artifact
        except Exception as e:
            raise hosuingprojectException(e, sys) from e
            

            
            



