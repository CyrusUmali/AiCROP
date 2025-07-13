import threading
import subprocess
import time

def run_streamlit():
    subprocess.run(["streamlit", "run", "app.py"])

def run_fastapi():
    subprocess.run(["uvicorn", "api.main:app", "--reload"])

if __name__ == "__main__":
    # Start Streamlit in a separate thread
    streamlit_thread = threading.Thread(target=run_streamlit)
    streamlit_thread.start()
    
    # Start FastAPI in the main thread
    run_fastapi()