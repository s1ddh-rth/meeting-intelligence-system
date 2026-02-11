#!/bin/bash
set -e

# Start FastAPI in the background
uvicorn src.app:app --host 0.0.0.0 --port 8000 &

# Start Streamlit in the foreground
streamlit run ui/streamlit_app.py --server.port 8501 --server.address 0.0.0.0 --server.headless true
