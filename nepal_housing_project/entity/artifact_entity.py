from dataclasses import dataclass



@dataclass
class DataingestionArtifact:
    trained_file_path:str
    test_file_path:str

@dataclass
class DataValidationArtifact:
    validation_status:bool
    message:str
    drift_report_drift_flie_path:str
    
@dataclass
class DataTransformationArtifact:
    transformed_object_file_path:str
    transformed_train_file_path:str
    transformed_test_file_path:str

@dataclass
class RegressorMetricArtifact:
    mean_sqr_error:float
    root_mean_sqr_error:float
    r_2_score:float

@dataclass
class ModelTrainerArtifact:
    trained_model_file_path:str
    metric_artifact:RegressorMetricArtifact



