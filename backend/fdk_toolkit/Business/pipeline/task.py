# Connects all data pipeline scripts 
# Output of one script is check, and then used as input for the next
# Used as a RQ task in API routes

from .preprocessing import preprocessing
from .postprocessing import postprocessing
from .fdk_business_pipeline import run_pipeline
import re

def business_task(filepath, result_id):

    # 1) Preprocessing / Feature detection
    output = preprocessing(filepath)
    if output.get("Error") == None:
        # No error
        df_mapped = output
    else:
        return output
    
    # 2) Analysis / Calculate fairness metrics 
    audit_response = run_pipeline(df_mapped, save_to_disk=False)

    # 3) Postprocessing / Create analysis report and summary
    output = postprocessing(audit_response, result_id)
    return output