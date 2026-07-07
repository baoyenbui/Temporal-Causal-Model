import pandas as pd
import os

def load_data(base_path):
    data = {}
    
    files = {
        "training_data": "checkins_lessons_checkouts_training.csv",
        "student_meta": "student_metadata.csv",
        "topic_pathway": "topic_pathway_metadata.csv",
        "construct_prerequisites_test": "construct_prerequisites_test.csv",
        "constructs_input_test": "constructs_input_test.csv",
        "construct_experiments_test": "construct_experiments_ates_test.csv",
        "checkin_to_checkout": "checkin_to_checkout.csv",
        "constructs_test": "construct_experiments_input_test.csv",
        "subject_meta": "subject_metadata.csv"
    }
    
    for key, file in files.items():
        path = os.path.join(base_path, file)
        if not os.path.exists(path):
            print(f"Warning: {file} not found at {path}")
            data[key] = pd.DataFrame()
        else:
            data[key] = pd.read_csv(path)
    
    return data