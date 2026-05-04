import pickle
import joblib

for f in ['../knn_classifier.pkl', '../pca_model.pkl', '../scaler.pkl']:
    try:
        model = joblib.load(f)
        print(f"Successfully loaded {f}: {type(model)}")
    except Exception as e:
        print(f"Failed to load {f}: {e}")
