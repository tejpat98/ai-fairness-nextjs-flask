from waitress import serve
from app import app

if __name__ == '__main__':
    print("Starting Waitress server on port 5000...")
    serve(app, host='0.0.0.0', port=5000)


# We are using waitress instead of "flask run", since flask run is not production grade
# ... unsafe and slower. Its only meant for development.