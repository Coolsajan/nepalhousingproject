from dataclasses import dataclass



@dataclass
class DataingestionArtifact:
    trained_file_path:str
    test_file_path:str

@dataclass
class DataValidationArtifact:
    validation_status:bool
    message:str
    drift_report_flie_path:str
    
