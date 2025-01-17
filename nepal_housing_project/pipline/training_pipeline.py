import sys
from nepal_housing_project.logger import logging
from nepal_housing_project.exception import hosuingprojectException
from nepal_housing_project.components.data_ingestion import DataIngestion
from nepal_housing_project.entity.config_entity import DataIngestionConfig
from nepal_housing_project.entity.artifact_entity import DataingestionArtifact


class TrainPipleline:
    def __init__(self):
            self.data_ingestion_config=DataIngestionConfig()


    def start_data_ingestion(self) -> DataingestionArtifact:
        '''
        This method of TrainPipeline class is responsible for starting data ingestion component          
            '''
    
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
            
    def run_pipleline(self, )->None:
            '''
            This method of TrainPipleline class is responsible for running complete pipeline
            '''
            try:
                    data_ingestion_artifact=self.start_data_ingestion()
                
            except Exception as e:
                  raise hosuingprojectException(e,sys)


                
            
