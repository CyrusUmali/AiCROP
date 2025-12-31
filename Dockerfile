FROM python:3.10

WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all code
COPY . .

# Start FastAPI
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "7860"]
