import os
import cv2
import numpy as np
import pickle
import base64
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from django.conf import settings

# Must define CustomPCA here so pickle can unpickle the model
class CustomPCA:
    def __init__(self, n_components=None):
        self.n_components = n_components
        self.components_ = None
        self.explained_variance_ = None
        self.explained_variance_ratio_ = None
        self.mean_ = None

    def fit_transform(self, X):
        self.mean_ = np.mean(X, axis=0)
        X_centered = X - self.mean_
        n_samples = X_centered.shape[0]
        cov_matrix = np.dot(X_centered.T, X_centered) / (n_samples - 1)
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues  = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        total_var = np.sum(eigenvalues)
        self.explained_variance_       = eigenvalues
        self.explained_variance_ratio_ = eigenvalues / total_var

        if isinstance(self.n_components, float) and self.n_components < 1:
            cumulative_variance = np.cumsum(self.explained_variance_ratio_)
            self.n_components = np.argmax(cumulative_variance >= self.n_components) + 1

        self.components_ = eigenvectors[:, :self.n_components].T
        return np.dot(X_centered, eigenvectors[:, :self.n_components])

    def transform(self, X):
        X_centered = X - self.mean_
        return np.dot(X_centered, self.components_.T)

import sys
sys.modules['__main__'].CustomPCA = CustomPCA

from sklearn.datasets import fetch_olivetti_faces

# Load models globally so they're only loaded once
base_dir = settings.BASE_DIR.parent
try:
    with open(os.path.join(base_dir, 'scaler.pkl'), 'rb') as f:
        scaler = pickle.load(f)
    with open(os.path.join(base_dir, 'pca_model.pkl'), 'rb') as f:
        pca_model = pickle.load(f)
    with open(os.path.join(base_dir, 'knn_classifier.pkl'), 'rb') as f:
        knn_model = pickle.load(f)
    
    # Load dataset for retrieving sample images of subjects
    dataset = fetch_olivetti_faces(shuffle=False)
    olivetti_faces = (dataset.data * 255).astype(np.uint8)
except Exception as e:
    print(f"Warning: Failed to load models or dataset: {e}")
    scaler, pca_model, knn_model, olivetti_faces = None, None, None, None


class DetectFaceView(APIView):
    def post(self, request, *args, **kwargs):
        file_obj = request.FILES.get('image')
        if not file_obj:
            return Response({"error": "No image provided"}, status=status.HTTP_400_BAD_REQUEST)
        
        try:
            # Read image
            file_bytes = np.asarray(bytearray(file_obj.read()), dtype=np.uint8)
            img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            if img is None:
                return Response({"error": "Invalid image file"}, status=status.HTTP_400_BAD_REQUEST)
            
            # Convert to grayscale for detection
            if len(img.shape) == 3:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            else:
                gray = img
                img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            
            # Load Haar Cascade
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            face_cascade = cv2.CascadeClassifier(cascade_path)
            
            # Detect faces
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            
            for (x, y, w, h) in faces:
                # Draw Bounding Box only
                cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
                
            # Encode image to base64
            _, buffer = cv2.imencode('.jpg', img)
            img_base64 = base64.b64encode(buffer).decode('utf-8')
            
            return Response({"result_image": f"data:image/jpeg;base64,{img_base64}"}, status=status.HTTP_200_OK)
        except Exception as e:
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


class PredictFaceView(APIView):
    def post(self, request, *args, **kwargs):
        if not all([scaler, pca_model, knn_model]):
            return Response({"error": "Models are not loaded on the server."}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
        
        file_obj = request.FILES.get('image')
        if not file_obj:
            return Response({"error": "No image provided"}, status=status.HTTP_400_BAD_REQUEST)
        
        try:
            # Read image
            file_bytes = np.asarray(bytearray(file_obj.read()), dtype=np.uint8)
            img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            if img is None:
                return Response({"error": "Invalid image file"}, status=status.HTTP_400_BAD_REQUEST)
            
            # Convert to grayscale for detection and prediction
            if len(img.shape) == 3:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            else:
                gray = img
                img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR) # ensure we have BGR to draw colored boxes
            
            # Load Haar Cascade
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            face_cascade = cv2.CascadeClassifier(cascade_path)
            
            # Detect faces
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            
            predicted_classes = []
            predicted_images = []
            for (x, y, w, h) in faces:
                # Crop and preprocess the face
                face_roi = gray[y:y+h, x:x+w]
                resized = cv2.resize(face_roi, (64, 64))
                flattened = resized.reshape(1, -1).astype(np.float64)
                
                # Apply Preprocessing (Scaler + PCA)
                scaled = scaler.transform(flattened)
                pca_transformed = pca_model.transform(scaled)
                
                # Predict
                pred = knn_model.predict(pca_transformed)
                
                class_id = int(pred[0])
                predicted_classes.append(f"Subj {class_id}")
                
                # Get a sample image of the predicted person if available
                if olivetti_faces is not None:
                    # Each subject has 10 images, index class_id * 10 is the first image of that subject
                    subject_face_flat = olivetti_faces[class_id * 10]
                    subject_face_img = subject_face_flat.reshape(64, 64)
                    _, face_buffer = cv2.imencode('.jpg', subject_face_img)
                    face_base64 = base64.b64encode(face_buffer).decode('utf-8')
                    predicted_images.append(f"data:image/jpeg;base64,{face_base64}")
                else:
                    predicted_images.append(None)
                
                # Draw Bounding Box and Text
                cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
                text = f"Subj {class_id}"
                cv2.putText(img, text, (x, y - 10 if y > 20 else y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                
            # Encode image to base64
            _, buffer = cv2.imencode('.jpg', img)
            img_base64 = base64.b64encode(buffer).decode('utf-8')
            
            return Response({
                "result_image": f"data:image/jpeg;base64,{img_base64}", 
                "predicted_classes": predicted_classes,
                "predicted_images": predicted_images
            }, status=status.HTTP_200_OK)
        except Exception as e:
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
