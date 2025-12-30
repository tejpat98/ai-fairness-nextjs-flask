import datetime
import os
import secrets
import json
from werkzeug.utils import secure_filename
from werkzeug.exceptions import BadRequest
from flask import Flask, jsonify, request
from flask_cors import CORS
from extensions import task_queue, redis_conn
from rq.job import Job


# Import blueprints
from fdk_toolkit.Business.routes import business_bp

UPLOAD_FOLDER = os.environ.get('UPLOAD_FOLDER', './uploads')
REPORTS_FOLDER = os.environ.get('REPORTS_FOLDER', './reports')
ALLOWED_EXTENSIONS = set(os.environ.get('ALLOWED_EXTENSIONS', 'csv').split(','))

app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['REPORTS_FOLDER'] = REPORTS_FOLDER
app.config['ALLOWED_EXTENSIONS'] = ALLOWED_EXTENSIONS
CORS(app)  # Enable CORS for all routes

# Register blueprints
app.register_blueprint(business_bp, url_prefix='/api/business')

@app.route('/api/')
def index():
    return jsonify(message="Hello from Flask Backend!")

@app.route('/api/check-status/<result_id>', methods=['GET'])
def api_check_status(result_id):
    # 1. Sanitise before querying redis
    result_id = secure_filename(result_id)
    try:
    # 2. Fetch the actual job object from RQ
        job = Job.fetch(result_id, connection=redis_conn)
        status = job.get_status()

        # 3. Handle different job states
        if status == 'finished':
            return jsonify({
                'status': 'finished'
            }), 200

        elif status == 'failed':
            return jsonify({
                'status': 'failed',
                'message': 'The data processing script crashed.'
            }), 500

        elif status == 'started' or status == 'queued':
            # Status is likely 'queued' or 'started'
            return jsonify({
                'status': status,
                'message': f'Job is currently {status}. Please wait.'
            }), 202

    except Exception as e:
        # Job was deleted from Redis (likely expired based on result_ttl)
        return jsonify({
            'status': 'expired',
            'message': 'The job record no longer exists in the queue.'
        }), 410 # 410 Gone is appropriate for expired resources
    
@app.route('/api/report/<domain_name>/<result_id>', methods=['GET'])
def api_get_report(domain_name, result_id):
    # 1. Santise domain and result_id
    safe_result_id = secure_filename(result_id)
    safe_domain_name = secure_filename(domain_name)

    # 2. Construct path for domain reports folder
    reports_path = os.environ.get('REPORTS_FOLDER', './reports')
    domain_reports_path = os.path.join(reports_path, safe_domain_name)

    # 3. Ensure the directory exists
    if not os.path.exists(domain_reports_path):
        return jsonify({"error": f"Invalid domain: {safe_domain_name}"}), 500

    response_data = {}

    try:
        # 4. Iterate through files in the directory
        for filename in os.listdir(domain_reports_path):
            # Check if filename starts with result_id and ends with .json or .txt
            if filename.startswith(safe_result_id) and filename.lower().endswith(('.json', '.txt')):
                
                file_path = os.path.join(domain_reports_path, filename)
                # Key for the JSON object (filename without extension)
                key_name = os.path.splitext(filename)[0]

                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                    # 5. Parse content: Try to load as JSON, fallback to raw string
                    if filename.lower().endswith('.json'):
                        try:
                            response_data[key_name] = json.loads(content)
                        except json.JSONDecodeError:
                            response_data[key_name] = content # Return as string if JSON is malformed
                    else:
                        # For .txt files, check if it's valid JSON anyway, otherwise keep as string
                        try:
                            response_data[key_name] = json.loads(content)
                        except json.JSONDecodeError:
                            response_data[key_name] = content

        if not response_data:
            return jsonify({"message": f"No matching report with id '{safe_result_id}' found"}), 404

        return jsonify(response_data), 200

    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({"error": f"failed to find report file with id {safe_result_id}"}), 500


    


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)