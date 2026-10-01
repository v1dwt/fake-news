import os
from app import app

if __name__ == "__main__":
    # Support both Gunicorn (Linux/Cloud) and Waitress (Windows/Local)
    port = int(os.environ.get("PORT", 5000))
    try:
        from waitress import serve
        print(f"Starting TruthLens production server with Waitress on port {port}...")
        serve(app, host="0.0.0.0", port=port)
    except ImportError:
        print(f"Starting TruthLens development server on port {port}...")
        app.run(host="0.0.0.0", port=port)
