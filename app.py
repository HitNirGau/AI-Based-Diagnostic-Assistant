from report import generate_augmented_report  # Adjust the function name as per report.py
from vectordb import search
from flask import Flask, render_template, request, url_for
import os,cv2
import requests
import numpy as np
import torch
import torch.nn.functional as F
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from tensorflow.keras.models import load_model
from torchvision import models,transforms
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd

# Load the models
model = load_model("models/breast_cancer.keras")
anomaly_model = load_model("models/anomaly_detector_breast_cancer.keras")

app = Flask(__name__)

UPLOAD_FOLDER = os.path.abspath('images')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['ANOMALY_FOLDER'] = 'static/anomaly/'
app.config['MALIGNANT_FOLDER'] = 'static/malignant/'
app.config['BENIGN_FOLDER'] = 'static/benign/'
app.config['GRADCAM_FOLDER'] = 'static/uploads/'

for folder in [app.config['UPLOAD_FOLDER'], app.config['GRADCAM_FOLDER'], 
               app.config['ANOMALY_FOLDER'], app.config['MALIGNANT_FOLDER'], app.config['BENIGN_FOLDER']]:
    os.makedirs(folder, exist_ok=True)



threshold = 0.2517586052417755

def preprocess_data(image_path, img_size=(124, 124)):
    datagen = ImageDataGenerator(rescale=1./255)

    img = load_img(image_path, target_size=img_size, color_mode='grayscale')
    
    img_array = img_to_array(img)

    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array


def calculate_reconstruction_loss(data, model):
    reconstructions = model.predict(data)
    data = np.clip(data, 0, 1)
    reconstructions = np.clip(reconstructions, 0, 1)
    
    reconstruction_errors = np.mean(np.square(data - reconstructions), axis=(1,2,3)) 
    return reconstruction_errors[0]  

class SmoothGradCAMPlusPlus:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.register_hooks()

    def register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]
        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_backward_hook(backward_hook)

    def generate_cam(self, inputs, target_class):
        outputs = self.model(inputs)
        if target_class is None:
            target_class = torch.argmax(outputs, dim=1).item()
        self.model.zero_grad()
        one_hot_output = torch.zeros_like(outputs)
        one_hot_output[0, target_class] = 1
        outputs.backward(gradient=one_hot_output)
        gradients = self.gradients.cpu().data.numpy()[0]
        activations = self.activations.cpu().data.numpy()[0]
        weights = np.mean(gradients, axis=(1, 2))
        grad_cam = np.zeros(activations.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            grad_cam += w * activations[i]
        grad_cam = np.maximum(grad_cam, 0)
        grad_cam -= np.min(grad_cam)
        if np.max(grad_cam) > 0:
            grad_cam /= np.max(grad_cam)
        grad_cam = cv2.resize(grad_cam, (inputs.size(-1), inputs.size(-2)))
        return grad_cam

    def apply_smoothing(self, image, target_class, num_samples=50, noise_std=0.2):
        smoothed_cam = np.zeros_like(image[0, 0].cpu().numpy())
        for _ in range(num_samples):
            noise = torch.normal(0, noise_std, size=image.size()).to(image.device)
            noisy_image = image + noise
            cam = self.generate_cam(noisy_image, target_class)
            smoothed_cam += cam
        smoothed_cam -= np.min(smoothed_cam)
        smoothed_cam /= np.max(smoothed_cam)
        return smoothed_cam

# Load pre-trained model for Grad-CAM
resnet_model = models.resnet50(pretrained=True)
resnet_model.eval()
target_layer = resnet_model.layer4[-1].conv3
cam_generator = SmoothGradCAMPlusPlus(resnet_model, target_layer)

@app.route('/')
def index():
    anomaly_images = os.listdir(os.path.join(app.static_folder, 'anomaly'))
    malignant_images = os.listdir(os.path.join(app.static_folder, 'malignant'))
    benign_images = os.listdir(os.path.join(app.static_folder, 'benign'))

    # Ensure report is defined (dummy data if needed)
    report = {"Findings Explanation": "No findings available"}  

    return render_template('upload_xray.html',
                           anomaly_images=anomaly_images,
                           malignant_images=malignant_images,
                           benign_images=benign_images,
                           report=report)  # Pass report to template



ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/predict', methods=['POST'])
def predict():
    if 'imagefile' not in request.files:
        return "No file part"
    
    imagefile = request.files["imagefile"]

    if imagefile.filename == '':
        return "No selected file"
    
    if imagefile:
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], imagefile.filename)
        imagefile.save(image_path)

        print(image_path)
        data = preprocess_data(image_path)

        # Get predictions from the model
        predictions = model.predict(data)
         
        probability = (predictions[0][0])  # Convert to a scalar float

        # Determine class
        if probability >= 0.5:
            result_text = "Malignant"
            save_path = os.path.join(app.config['MALIGNANT_FOLDER'], imagefile.filename)
        else:
            result_text = "Benign"
            save_path = os.path.join(app.config['BENIGN_FOLDER'], imagefile.filename)

        # Save to respective folder
       # os.rename(image_path, save_path)

        # Anomaly detection
        anomaly_loss = calculate_reconstruction_loss(data, anomaly_model)
        # threshold = np.mean(anomaly_loss) + np.std(anomaly_loss) * 1.5  # Adjust with standard deviation

        # anomaly_loss = np.abs(anomaly_loss)
        print(f"Average Reconstruction Loss for Normal Data: {np.mean(anomaly_loss)}")

        anomaly_result = "Normal" if anomaly_loss < threshold else "Anomalous"

        # Save anomaly image if detected
        if anomaly_result == "Anomalous":
            anomaly_save_path = os.path.join(app.config['ANOMALY_FOLDER'], imagefile.filename)
            # os.rename(save_path, anomaly_save_path)

        image = Image.open(save_path).convert("RGB")
        preprocess = transforms.Compose([
            transforms.Resize((124,124)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        input_image = preprocess(image).unsqueeze(0)
        smoothed_heatmap = cam_generator.apply_smoothing(input_image, target_class=None)
        heatmap = cv2.applyColorMap(np.uint8(255 * smoothed_heatmap), cv2.COLORMAP_TURBO)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

        original_image = np.array(image)
        heatmap = cv2.resize(heatmap, (original_image.shape[1], original_image.shape[0]))

        overlayed_image = cv2.addWeighted(original_image, 0.6, heatmap, 0.4, 0)
        gradcam_path = os.path.join(app.config['GRADCAM_FOLDER'], imagefile.filename)
        cv2.imwrite(gradcam_path, cv2.cvtColor(overlayed_image, cv2.COLOR_RGB2BGR))
        findings = f"Anomaly Status: {anomaly_result}, Reconstruction Loss: {anomaly_loss}"
        augmented_report = generate_augmented_report(result_text, findings)

        # Send Data to Sequelize Backend
        data_to_send = {
            # 'patient_id': patient_id,
            'prediction': result_text,
            'anomaly_result': anomaly_result,
            'grad_cam_image': gradcam_path,
            'report': augmented_report
        }

        searching_results = search(image_path)
        print(f"Searching Result: {searching_results}")
        csv_path = r'/data/patients.csv'
        try:
         df = pd.read_csv('data/patients.csv')
        except Exception as e:
         print(e)
         df = None  # Assign None if reading fails

        if df is not None:
         print(df.to_string(index=False))

        else:
         print("CSV could not be loaded.")


        # Assuming patient name is provided in the form
        patient_name = request.form.get('patientName')  # Add patient_name field in the form

        # Check if patient_name and df are valid

        # Ensure 'data' directory exists
    csv_dir = 'data'
    os.makedirs(csv_dir, exist_ok=True)
    csv_path = os.path.join(csv_dir, 'patients.csv')

    # Load existing CSV or create a new DataFrame
    try:
        df = pd.read_csv(csv_path)
        print("CSV Loaded Successfully.")
    except FileNotFoundError:
        print("CSV not found. Creating a new file.")
        df = pd.DataFrame(columns=["Doctor's Name", "Patient Name", "Age", "Gender", "Contact",
                                "Medical History", "Medications", "Allergies",
                                "Diagnosis", "Date", "Feedback"])

    # Patient Name Handling
    # Clean patient name from form
    # Ensure we handle None, empty strings, and whitespace properly
    patient_name = request.form.get('patientName', '').strip().lower()
    df['Patient Name'] = df['Patient Name'].astype(str).str.strip().str.lower()

    # Compare names
    if patient_name not in df['Patient Name'].values:
        patient_name = 'unknown'

    # Update or Insert Patient Data
    if patient_name in df['Patient Name'].values:
        df.loc[df['Patient Name'] == patient_name, 'Diagnosis'] = result_text
        df.loc[df['Patient Name'] == patient_name, 'Feedback'] = augmented_report
    else:
        new_data = {
            "Doctor's Name": "Unknown",
            "Patient Name": patient_name,
            "Age": "", "Gender": "", "Contact": "",
            "Medical History": "", "Medications": "", "Allergies": "",
            "Diagnosis": result_text,
            "Date": pd.Timestamp.now().strftime('%Y-%m-%d'),
            "Feedback": augmented_report
        }
        df = pd.concat([df, pd.DataFrame([new_data])], ignore_index=True)

    df.to_csv(csv_path, index=False)

    return render_template("upload_xray.html", 
                        output=f"Prediction: {result_text}", 
                        anomaly_result=anomaly_result,
                        gradcam_image=url_for('static', filename=f'uploads/{imagefile.filename}'),
                        report=augmented_report,
                        searching_results=searching_results)                       


    return "File upload failed"

if __name__ == '__main__':
    app.run(port=5000, debug=True)
