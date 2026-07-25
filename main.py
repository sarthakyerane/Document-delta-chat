"""
delta-chat · main.py
Application entry point — creates the FastAPI app and exposes it as `app`.
"""
from src.api.app import create_app

app = create_app()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
