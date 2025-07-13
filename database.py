import sqlite3
from datetime import datetime

# Function to initialize database
def initialize_database():
    conn = sqlite3.connect('logs.db')
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            model_name TEXT NOT NULL,
            input_data TEXT NOT NULL,
            result TEXT NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()

# Function to insert a prediction log into database
def log_prediction(model_name, input_data, result):
    conn = sqlite3.connect('logs.db')
    c = conn.cursor()
    c.execute('''
        INSERT INTO predictions (model_name, input_data, result)
        VALUES (?, ?, ?)
    ''', (model_name, str(input_data), result))
    conn.commit()
    conn.close()

# Function to retrieve all logs from database
def fetch_logs():
    conn = sqlite3.connect('logs.db')
    c = conn.cursor()
    c.execute('''
        SELECT model_name, input_data, result, timestamp FROM predictions ORDER BY timestamp DESC
    ''')
    logs = c.fetchall()
    conn.close()
    return logs

# Function to clear all logs from database
def clear_logs():
    conn = sqlite3.connect('logs.db')
    c = conn.cursor()
    c.execute('''
        DELETE FROM predictions
    ''')
    conn.commit()
    conn.close()

# Initialize the database when the script runs
initialize_database()
