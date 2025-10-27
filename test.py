import requests
import json
import time

def test_stream_real_time():
    print("🚀 Testing Real-time Streaming...")
    
    data = {
        "message": "How to grow tomatoes in small spaces?",
        "chat_history": []
    }
    
    response = requests.post(
        "http://localhost:8000/api/v1/chat/stream",
        json=data,
        stream=True,
        timeout=30
    )
    
    print("AI Response: ", end="", flush=True)
    
    for line in response.iter_lines(decode_unicode=True):
        if line and line.startswith('data: '):
            try:
                chunk_data = json.loads(line[6:])
                if chunk_data.get('chunk'):
                    print(chunk_data['chunk'], end='', flush=True)
                    time.sleep(0.1)  # Small delay to see streaming effect
                if chunk_data.get('is_complete'):
                    print(f"\n\n✅ Done! Disclaimer: {chunk_data.get('disclaimer', '')}")
            except:
                pass

if __name__ == "__main__":
    test_stream_real_time()