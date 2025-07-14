import pinecone
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.decomposition import PCA
import os

# Initialize Pinecone
from pinecone import Pinecone

pc = Pinecone(
    api_key="pcsk_47VHah_TgYbCoPPcfmUmXSR2zfRkThWd5kkgoWCAMggXTdon15UBrJkSN2ZUGLBxNXYH4w",
    environment="us-east-1"
)

index_name = "breastcancet"
host = "https://breastcancet-19ofzca.svc.aped-4627-b74a.pinecone.io"

index = pc.Index(
    # api_key="pcsk_47VHah_TgYbCoPPcfmUmXSR2zfRkThWd5kkgoWCAMggXTdon15UBrJkSN2ZUGLBxNXYH4w",
    name=index_name,
    host=host,
)

# Load model and remove last two layers
model = load_model("models/breast_cancer.keras")
model.pop()
model.pop()

# Load stored embeddings and fit PCA
embeddings = np.load("models/embeddings.npy")
pca = PCA(n_components=48)
pca.fit(embeddings)

UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def preprocess_image(image_path):
    """Preprocesses an image for feature extraction."""
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Image at path '{image_path}' could not be loaded.")
    
    image = cv2.resize(image, (124, 124))
    image = np.array(image, dtype=np.float32) / 255.0
    image = np.expand_dims(image, axis=-1)  # Add channel dimension
    image = np.expand_dims(image, axis=0)   # Add batch dimension
    
    return image

def extract_features(img_path):
    """Extracts features from an image and applies PCA."""
    img_array = preprocess_image(img_path)
    features = model.predict(img_array)
    return features.flatten()

def search(image_file):
    """Search for similar images in Pinecone given an image file."""
    
    if isinstance(image_file, str):  # If a string (file path) is passed
        img_path = image_file  
    else:  # If a file object is passed
        img_path = os.path.join(UPLOAD_FOLDER, image_file.filename)
        image_file.save(img_path)  

    feature_vector = extract_features(img_path)
    
    try:
        response = index.query(
            vector=feature_vector.tolist(),
            top_k=6,
            include_metadata=True
        )
        
        results = [
            {"image_id": match["id"], "similarity": match["score"]}
            for match in response["matches"]
        ]
        print(results)
        return results
    
    except Exception as e:
        print(f"Error querying Pinecone: {str(e)}")
        return {"error": "Error querying Pinecone"}
