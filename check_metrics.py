import pickle

with open('precomputation/metrics.pkl', 'rb') as f:
    metrics = pickle.load(f)
    
print("Loaded metrics keys:", metrics.keys())
print("Full content:", metrics)