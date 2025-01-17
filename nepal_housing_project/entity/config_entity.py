import os
from nepal_housing_project.constants import *
from dataclasses import  dataclass
from datetime import datetime

TIMESTAMP:str=datetime.now().strftime("%m_%d_%Y_%H_%M_%S")



@dataclass
class TraningPipelineConfig:
    pipeline_name:str=PIPELINE_NAME
    artifact_dir:str=os.path.join(ARTIFACT_DIR,TIMESTAMP)
    timestamp:str=TIMESTAMP


traning_pipeline_config:TraningPipelineConfig=TraningPipelineConfig()

@dataclass
class DataIngestionConfig:
    data_ingestion_dir:str=os.path.join(traning_pipeline_config.artifact_dir,DATA_INGESTION_DIR_NAME)
    feature_store_file_path:str=os.path.join(data_ingestion_dir,DATA_INGESTION_FEATURE_STORE_DIR,FILE_NAME)
    traning_file_path:str=os.path.join(feature_store_file_path,DATA_INGESTION_INGESTED_DIR,TRAIN_FILE_NAME)
    testing_file_path:str=os.path.join(DATA_INGESTION_INGESTED_DIR,TEST_FILE_NAME)
    train_test_split_ration:float=DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO
    collection_name:str=COLLECTION_NAME

@dataclass
class DataValidationConfig:
    data_validation_dir:str=os.path.join(traning_pipeline_config.artifact_dir,DATA_VALIDATION_DIR_NAME)
    data_drift_flie_path:str=os.path.join(data_validation_dir,DATA_VALIDATION_DRIFT_REPORT_DIR,DATA_VALIDATION_DRIFT_REPORT_FILE_NAME)

@dataclass
class DataTransformationConfig:
    data_transformation_dir_path:str=os.path.join(traning_pipeline_config.artifact_dir,DATA_TRANSFORMATION_DIR_NAME)
    