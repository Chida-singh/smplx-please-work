import torch
import pickle

# Pick any .pkl file from the folder, e.g., "00873.pkl"
with open("word-level-dataset/00873.pkl", "rb") as f:
    data = pickle.load(f)

print(type(data))
if hasattr(data, 'keys'):
    print(data.keys())
else:
    print(dir(data))

# If it's a dict, print a sample value
if isinstance(data, dict):
    for k in list(data.keys())[:3]:
        print(f"Key: {k}")
        print(f"Value type: {type(data[k])}")
        print(f"Value sample: {str(data[k])[:500]}")