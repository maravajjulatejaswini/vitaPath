from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import joblib
from PIL import Image
import numpy as np
import io
import numpy as np
import joblib
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.utils import img_to_array
from tensorflow.keras.applications.vgg16 import preprocess_input
import os
print(os.path.abspath('frontend'))
app = Flask(__name__, template_folder='frontend')

# Initialize Flask app with custom template folder

# Load the saved Random Forest model
rf_classifier = joblib.load('random_forest_vitamin_bnew.pkl')

# Load the pre-trained VGG16 model for feature extraction
vgg_model = VGG16(weights='imagenet', include_top=False, input_shape=(150, 150, 3))
feature_extractor = Model(inputs=vgg_model.input, outputs=vgg_model.output)

for layer in feature_extractor.layers:
    layer.trainable = False

# Define class names
class_names = {
    0: 'Vitamin A Deficiency',
    1: 'Vitamin B Deficiency',
    2: 'Vitamin C Deficiency',
    3: 'Vitamin D Deficiency',
    4: 'Vitamin E Deficiency',
    5: 'No Deficiency'
}

def predict_deficiency(img):
    img = img.resize((150, 150))  # Resize to match model input
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    
    features = feature_extractor.predict(img_array)
    features_flattened = features.reshape(features.shape[0], -1)
    
    prediction = rf_classifier.predict(features_flattened)
    return class_names.get(prediction[0], 'Unknown Deficiency')

@app.route('/upload', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        file = request.files['image']
        if file:
            img = Image.open(file.stream)
            predicted_class_name = predict_deficiency(img)
            return render_template('upload.html', prediction=predicted_class_name)
    
    return render_template('upload.html', prediction=None)

@app.route('/')
def index():
    return render_template('index.html')
@app.route('/login')
def login():
    return render_template('login.html')

@app.route('/signup')
def signup():
    return render_template('signup.html')

@app.route('/home')
def home():
    return render_template('home.html')

@app.route('/get_started')
def get_started():
    return render_template('index(3).html') 
# 
# @app.route('/upload', methods=['GET', 'POST'])
# def predict():
#     if request.method == 'POST':
#         # Check if an image is provided
#         if 'image' not in request.files:
#             return render_template('img_upload.html', prediction="No file part.")
#         
#         file = request.files['image']
#         if file.filename == '':
#             return render_template('img_upload.html', prediction="No file selected.")
#         
#         if file and allowed_file(file.filename):
#             # Process the image without saving it
#             try:
#                 image = Image.open(file.stream).convert('RGB')
#                 image = image.resize((64, 64))  # Resize image to match model input
#                 image_array = np.array(image).flatten().reshape(1, -1)  # Flatten and reshape for model input
#                 
#                 # Predict using the loaded model
#                 prediction = model.predict(image_array)[0]
#                 return render_template('img_upload.html', prediction=f"Predicted Vitamin Deficiency: {prediction}")
#             except Exception as e:
#                 return render_template('img_upload.html', prediction=f"Error processing image: {str(e)}")
#     
#     return render_template('img_upload.html', prediction=None)

if __name__ == '__main__':
    app.run(debug=True)
