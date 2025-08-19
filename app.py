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
from flask import Flask, render_template, request
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
import os

from flask import Flask, render_template, request, redirect, url_for, flash
from flask_sqlalchemy import SQLAlchemy
import requests
import re




from flask import request, redirect, url_for, flash
from flask import session, redirect, url_for, render_template


from flask import render_template, request, redirect, url_for, flash, session
from flask_sqlalchemy import SQLAlchemy



from flask import Flask, render_template, request, redirect, url_for, flash, session
from flask_sqlalchemy import SQLAlchemy
import requests
import re
from werkzeug.security import generate_password_hash, check_password_hash  # For password security

app = Flask(__name__, template_folder='frontend')
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'  # Use SQLite for simplicity
app.config['SECRET_KEY'] = '9f8b8c4d5a6f7c8e9d0a1b2c3d4e5f6a'

db = SQLAlchemy(app)

# Create a User model
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(100), unique=True, nullable=False)
    password = db.Column(db.String(255), nullable=False)  
# Route for Signup Page
@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        name = request.form.get('name')
        email = request.form.get('email')
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')

        # Validate that all fields are filled
        if not name or not email or not password or not confirm_password:
            flash("All fields are required!", "danger")
            return redirect(url_for('signup'))

        # Check if password and confirm password match
        if password != confirm_password:
            flash("Passwords don't match!", "danger")
            return redirect(url_for('signup'))

        # Validate email format
        if not validate_email(email):
            flash("Invalid email format", "danger")
            return redirect(url_for('signup'))

        # Verify if email is already in the database
        if User.query.filter_by(email=email).first():
            flash("Email already exists!", "danger")
            return redirect(url_for('signup'))

        # Verify email existence using Hunter.io (optional)
        if not verify_email_exists(email):
            flash("Invalid email", "danger")
            return redirect(url_for('signup'))

        # Hash the password before storing it
        hashed_password = generate_password_hash(password, method='pbkdf2:sha256')

        # Create a new user
        new_user = User(name=name, email=email, password=hashed_password)
        db.session.add(new_user)
        db.session.commit()

        flash("Account created successfully!", "success")
        return redirect(url_for('login'))  # Redirect to login after successful signup

    return render_template('signup.html')

# Route for Login Page
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')

        # Validate that email and password are provided
        if not email or not password:
            flash("Email and password are required!", "danger")
            return redirect(url_for('login'))

        # Retrieve the user from the database
        user = User.query.filter_by(email=email).first()

        if user is None or not check_password_hash(user.password, password):
            flash("Invalid email or password.", "danger")
            return redirect(url_for('login'))

        session['user_id'] = user.id  # Store user ID in session
        flash("Login successful!", "success")
        return redirect(url_for('home'))  # Redirect to home page after successful login

    return render_template('login.html')

# Email validation function using Hunter.io
def verify_email_exists(email):
    hunter_api_key = "b4229aa30ab3a569894b51c8b05b007641ce636e" #  API key
    url = f"https://api.hunter.io/v2/email-verifier?email={email}&api_key={hunter_api_key}"
    
    try:
        response = requests.get(url)
        data = response.json()
        return data.get('data', {}).get('result') == 'deliverable'
    except Exception as e:
        print(f"Error verifying email: {e}")
        return False

# Email format validation using regex
def validate_email(email):
    email_regex = r'^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$'
    return re.match(email_regex, email) is not None







# Initialize Flask app with custom template folder

# Load the saved Random Forest model
''''rf_classifier = joblib.load('random_forest_vitamin_bnew.pkl')

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


@app.route('/upload', methods=['GET','POST'])
def upload():
    return render_template('upload.html', prediction=None)
@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        file = request.files['image']  # Use .get() to avoid errors if the key doesn't exist
        if file:
            try:
                # Open the image file
                img = Image.open(file.stream)
                img.show()  # This will display the image, helpful for debugging
                print(f"Image uploaded: {file.filename}")

                # Call your prediction function (replace with your actual function)
                predicted_class_name = predict_deficiency(img)

                print(f"Prediction: {predicted_class_name}")  # Ensure prediction is made

                # Return the result to the user
                return render_template('upload.html', prediction=predicted_class_name)
            except Exception as e:
                print(f"Error processing image: {e}")
                return render_template('upload.html', prediction="Error during prediction")
    
    return render_template('upload.html', prediction=None)'''

from flask import Flask, render_template, request, redirect, url_for
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import os


from flask import Flask, request, jsonify

# Load the trained model
'''MODEL_PATH = "deficiency_classifier.h5"
model = tf.keras.models.load_model(MODEL_PATH)
print("Model loaded successfully.")  # Debug

# Define class labels (update these with your dataset class names)
class_labels = ['Vitamin A and B12 deficiency', 'B1, B2 ,B3 ,Iron deficiency', 'B2, B3, B12 deficiency', 'B7, B9, C deficiency', 'C deficiency']

# Ensure the 'uploads' folder exists for storing uploaded images
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Route to render the upload page
@app.route('/upload')
def upload():
    return render_template('upload.html')

# Route to handle image upload and prediction
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file uploaded", 400

    file = request.files['file']
    if file.filename == '':
        return "No file selected", 400

    if file:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)

        # Preprocess the image
        img = image.load_img(file_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        img_array /= 255.0  # Normalize

        # Predict using the model
        predictions = model.predict(img_array)
        predicted_class = np.argmax(predictions)  # Get the highest probability class
        predicted_label = class_labels[predicted_class]

        return jsonify({"Prediction": predicted_label}) ---- it workss'''
from flask import Flask, render_template, request, jsonify
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import os



# Load the TFLite model
MODEL_PATH = r"D:\Desktop\essentials\feature wise major\feature wise major\f1 img prediction\additionals\my f1\deficiency_classifier_quantized.tflite"
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()


input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

class_labels = ['Vitamin A and B12 deficiency', 'B1, B2 ,B3 ,Iron deficiency', 
                'B2, B3, B12 deficiency', 'B7, B9, C deficiency', 'C deficiency']


UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Route to render the upload page
@app.route('/upload')
def upload():
    return render_template('upload.html')

# Route to handle image upload and prediction
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file uploaded", 400

    file = request.files['file']
    if file.filename == '':
        return "No file selected", 400

    if file:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)

        # Preprocess the image
        img = image.load_img(file_path, target_size=(224, 224))  # Adjust based on model input
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0).astype(np.float32)  # Ensure correct shape
        img_array /= 255.0  # Normalize if needed

        # Run inference
        interpreter.set_tensor(input_details[0]['index'], img_array)
        interpreter.invoke()
        predictions = interpreter.get_tensor(output_details[0]['index'])

        # Get the highest probability class
        predicted_class = np.argmax(predictions)
        predicted_label = class_labels[predicted_class]

        return jsonify({"Prediction": predicted_label})



@app.route('/')
def index():
    return render_template('index.html')
'''@app.route('/login')
def login():
    return render_template('home.html')

@app.route('/signup')
def signup():
    return render_template('home.html')'''

@app.route('/home')
def home():
    return render_template('home.html')

@app.route('/get_started')
def get_started():
    return render_template('index(3).html') 
@app.route('/symptombased')
def symptombased():
    return render_template('symptombased.html')
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

data = pd.read_csv('filtered_dataset.csv')

rda_data ={
  "Vitamin A - RAE": {

    "Infants (0-6 months)": {"RDA": 400, "UL": 600},
    "Infants (7-12 months)": {"RDA": 500, "UL": 600},
    "Children (1-3 years)": {"RDA": 300, "UL": 600},
    "Children (4-8 years)": {"RDA": 400, "UL": 900},
    "Children (9-13 years)": {"RDA": 600, "UL": 1700},
    "Teens (14-18 years)": {"RDA": 700, "UL": 2800},  # Add missing category
    "Adults (19+ years)": {"RDA": 900, "UL": 3000},
    
    "Pregnant Women": {"RDA": 770, "UL": 3000},
    "Lactating Women": {"RDA": 1300, "UL": 3000}
  
  },
  "Vitamin B12": {
    "Infants (0-6 months)": {"RDA": 0.4, "UL": "No UL"},
    "Infants (7-12 months)": {"RDA": 0.5, "UL": "No UL"},
    "Children (1-3 years)": {"RDA": 0.9, "UL": "No UL"},
    "Children (4-8 years)": {"RDA": 1.2, "UL": "No UL"},
    "Children (9-13 years)": {"RDA": 1.8, "UL": "No UL"},
    "Teens (14-18 years)": {"RDA": 2.4, "UL": "No UL"},
    "Adults (19+ years)": {"RDA": 2.4, "UL": "No UL"},
    "Pregnant Women": {"RDA": 2.6, "UL": "No UL"},
    "Lactating Women": {"RDA": 2.8, "UL": "No UL"}
  },
  "Vitamin B6": {
    "Infants (0-6 months)": {"RDA": 0.1, "UL": "No UL"},
    "Infants (7-12 months)": {"RDA": 0.3, "UL": "No UL"},
    "Children (1-3 years)": {"RDA": 0.5, "UL": 30},
    "Children (4-8 years)": {"RDA": 0.6, "UL": 40},
    "Children (9-13 years)": {"RDA": 1.0, "UL": 60},
    "Teens (14-18 years)": {"RDA": 1.2, "UL": 80},
    "Adults (19+ years)": {"RDA": 1.3, "UL": 100},
   
    "Pregnant Women": {"RDA": 1.9, "UL": 100},
    "Lactating Women": {"RDA": 2.0, "UL": 100}
  },
  "Vitamin C": {
    "Infants (0-6 months)": {"RDA": 40, "UL": "No UL"},
    "Infants (7-12 months)": {"RDA": 50, "UL": "No UL"},
    "Children (1-3 years)": {"RDA": 15, "UL": 400},
    "Children (4-8 years)": {"RDA": 25, "UL": 650},
    "Children (9-13 years)": {"RDA": 45, "UL": 1200},
    "Teens (14-18 years)": {"RDA": 65, "UL": 1800},
    "Adults (19+ years)": {"RDA": 75, "UL": 2000},
    "Pregnant Women": {"RDA": 85, "UL": 2000},
    "Lactating Women": {"RDA": 120, "UL": 2000}
  },
  "Vitamin E": {
    "Infants (0-6 months)": {"RDA": 4, "UL": "No UL"},
    "Infants (7-12 months)": {"RDA": 5, "UL": "No UL"},
    "Children (1-3 years)": {"RDA": 6, "UL": 200},
    "Children (4-8 years)": {"RDA": 7, "UL": 300},
    "Children (9-13 years)": {"RDA": 11, "UL": 600},
    "Teens (14-18 years)": {"RDA": 15, "UL": 800},
    "Adults (19+ years)": {"RDA": 15, "UL": 1000},
    "Pregnant Women": {"RDA": 15, "UL": 1000},
    "Lactating Women": {"RDA": 19, "UL": 1000}
  },
  "Vitamin K": {
    "Infants (0-6 months)": {"RDA": 2.0, "UL": "No UL"},
    "Infants (7-12 months)": {"RDA": 2.5, "UL": "No UL"},
    "Children (1-3 years)": {"RDA": 30, "UL": "No UL"},
    "Children (4-8 years)": {"RDA": 55, "UL": "No UL"},
    "Children (9-13 years)": {"RDA": 60, "UL": "No UL"},
    "Teens (14-18 years)": {"RDA": 75, "UL": "No UL"},
    "Adults (19+ years)": {"RDA": 90, "UL": "No UL"},
    "Pregnant Women": {"RDA": 90, "UL": "No UL"},
    "Lactating Women": {"RDA": 90, "UL": "No UL"}
  }
}



import requests
from bs4 import BeautifulSoup
from urllib.parse import quote

import requests
from bs4 import BeautifulSoup

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options


def fetch_image_url(query):
    search_url = f"https://www.google.com/search?q={quote(query)}+food&tbm=isch"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }

    try:
        # Send GET request to Google
        response = requests.get(search_url, headers=headers, timeout=10)
        response.raise_for_status()  # Raise exception for HTTP errors

        # Parse HTML response with BeautifulSoup
        soup = BeautifulSoup(response.text, 'html.parser')

        # Look for multiple image attributes (e.g., 'src', 'data-src', 'srcset')
        img_tags = soup.find_all('img')
        for img_tag in img_tags:
            img_url = img_tag.get('src') or img_tag.get('data-src') or img_tag.get('srcset')
            if img_url:
                return img_url

        # If no valid image found
        print(f"No valid image found for query: {query}")
        return "https://via.placeholder.com/150"  # Fallback image

    except requests.exceptions.RequestException as e:
        print(f"Error fetching image for query '{query}': {e}")
        return "https://via.placeholder.com/150"  # Fallback image




@app.route('/recommend', methods=['POST', 'GET'])
def recommend():
    if request.method == 'POST':
        # Get inputs from form
        deficient_vitamin = request.form['vitamin']
        user_category = request.form['category']
        N = int(request.form['num_items'])

        # Get RDA value
        rda_value = rda_data[deficient_vitamin][user_category]['RDA']

        # Extract relevant vitamin column
        vitamin_column = f"Data.Vitamins.{deficient_vitamin}"
        X = data[[vitamin_column]]

        # Standardize the data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        rda_scaled = scaler.transform([[rda_value]])

        # Apply KNN
        knn = NearestNeighbors(n_neighbors=N, metric='euclidean')
        knn.fit(X_scaled)
        distances, indices = knn.kneighbors(rda_scaled)

        # Generate recommendations
        recommendations = []
        for idx in indices[0]:
            item = data.iloc[idx]
            food_name = item['Description']
            image_url = fetch_image_url(item['Category'])
            recommendations.append({
                "category": item['Category'],
                "description": food_name,
                "vitamin_value": item[vitamin_column],
                "image_url": image_url
            })

        return render_template('recommend.html', recommendations=recommendations)

    # If it's a GET request, just render the recommendation form
    return render_template('recommend.html')


if __name__ == '__main__':
    with app.app_context():
        db.create_all()  # Ensure the database is created
    app.run(debug=True)
else:
    gunicorn_app = app


