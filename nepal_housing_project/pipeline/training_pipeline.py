import sys
from nepal_housing_project.logger import logging
from nepal_housing_project.exception import hosuingprojectException
from nepal_housing_project.components.data_ingestion import DataIngestion
from nepal_housing_project.components.data_validation import DataValidation
from nepal_housing_project.components.data_transformation import DataTransformation
from nepal_housing_project.entity.config_entity import (DataIngestionConfig,
                                                        DataValidationConfig,
                                                        DataTransformationConfig)
from nepal_housing_project.entity.artifact_entity import (DataingestionArtifact,
                                                          DataValidationArtifact,
                                                          DataTransformationArtifact)


class TrainPipeline:
    def __init__(self):
            self.data_ingestion_config=DataIngestionConfig()
            self.data_validation_config=DataValidationConfig()
            self.data_transformation_config=DataTransformationConfig()


    def start_data_ingestion(self) -> DataingestionArtifact:
        """
        This method of TrainPipeline class is responsible for starting data ingestion component
        """
        try:
            logging.info("Entered the start_data_ingestion method of TrainPipeline class")
            logging.info("Getting the data from mongodb")
            data_ingestion = DataIngestion(data_ingestion_config=self.data_ingestion_config)
            data_ingestion_artifact = data_ingestion.initiate_data_ingestion()
            logging.info("Got the train_set and test_set from mongodb")
            logging.info(
                "Exited the start_data_ingestion method of TrainPipeline class"
            )
            return data_ingestion_artifact
        except Exception as e:
            raise hosuingprojectException(e, sys) from e
        
    def start_data_validation(self,data_ingestion_artifact:DataingestionArtifact)-> DataValidationArtifact:
        """
        This method of traning pipeline will help you to validate the data.
        """
        logging.info("Entred into data validation step of train pipleline.")
        try:
            data_validation=DataValidation(data_ingestion_artifact=data_ingestion_artifact,
                                           data_validation_config=self.data_validation_config)  

            data_validation_artifact=data_validation.initiate_data_validation()

            logging.info("Data validation Perfomed")
            
            logging.info("Exited the start_data_validation method of TrainPipeline class")

            return data_validation_artifact


        except Exception as e:
              raise hosuingprojectException(e,sys) from e

    def start_data_transformation(self,data_ingestion_artifact:DataingestionArtifact,data_validation_artifact:DataValidationArtifact)->DataTransformationArtifact:
        """
        This method of Trainpipeline is responsible for  starting data transformation
        """
        try:
            data_transformation=DataTransformation(data_ingestion_artifact=data_ingestion_artifact,
                                                    data_validation_artifact=data_validation_artifact,
                                                    data_transformation_config=self.data_transformation_config)

            data_transformation_artifact=data_transformation.initiate_data_transformation()
            return data_transformation_artifact

        except Exception as e:
            raise hosuingprojectException(e,sys)

            
    def run_pipeline(self, )->None:
            '''
            This method of TrainPipleline class is responsible for running complete pipeline
            '''
            try:
                    data_ingestion_artifact=self.start_data_ingestion()
                    data_validation_artifact=self.start_data_validation(data_ingestion_artifact=data_ingestion_artifact)
                    data_transformation_artifact = self.start_data_transformation(
                                                    data_ingestion_artifact=data_ingestion_artifact, 
                                                    data_validation_artifact=data_validation_artifact)
            except Exception as e:
                  raise hosuingprojectException(e,sys)


                
            
