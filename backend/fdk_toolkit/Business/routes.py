import os
import secrets
from werkzeug.utils import secure_filename
import pandas as pd
import numpy as np
from flask import Blueprint, request, current_app, jsonify
from datetime import datetime
from extensions import task_queue

# Import tasks
from .pipeline.task import business_task

# Flask blueprint
business_bp = Blueprint('business', __name__)


def allowed_file(filename):
    """Checks if the file extension is in the allowed set."""
    return '.' in secure_filename(filename) and \
           filename.rsplit('.', 1)[1].lower() in current_app.config.get('ALLOWED_EXTENSIONS')

@business_bp.route('/upload', methods=['POST'])
def upload_file():
     # 1. Input Validation: Check for file existence
    if 'file' not in request.files:
        return jsonify({
            'error': 'Missing file part of request',
        }), 400 # 400 Bad Request
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({
            'error': 'No file selected',
        }), 400 # 400 Bad Request

    # 2. File Type Validation
    if not allowed_file(file.filename):
        return jsonify({
            'error': 'Invalid file type',
            'allowed': list(current_app.config.get('ALLOWED_EXTENSIONS'))
        }), 400 # 400 Bad Request
    
    # 3. Save File
    # Generate a unique id using timestamp and random string.
    result_id =  datetime.now().strftime('%Y%m%d%H%M%S') + secrets.token_urlsafe(8)
    filepath = os.path.join(os.environ.get('UPLOAD_FOLDER', './uploads'), result_id)
    
    file.save(filepath)

    # 4. Enqueue task for business data processing pipeline
    rq_job = task_queue.enqueue(business_task, filepath, result_id, job_id=result_id)
    
    # Return 202 ACCEPTED, indicating the job has been accepted for processing
    return jsonify({
        'message': 'File uploaded and analysis job started.',
        'result_id': result_id
    }), 202