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

app = Flask(__name__, template_folder='frontend')
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'  
app.config['SECRET_KEY'] = '9f8b8c4d5a6f7c8e9d0a1b2c3d4e5f6a'

db = SQLAlchemy(app)
from flask import request, redirect, url_for, flash
from flask import session, redirect, url_for, render_template

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(100), unique=True, nullable=False)
    password = db.Column(db.String(100), nullable=False)

from flask import render_template, request, redirect, url_for, flash, session
from flask_sqlalchemy import SQLAlchemy




@app.route('/update_profile', methods=['GET', 'POST'])
def update_profile():
    user_email = session.get('user_email')
    
  
    
    user = User.query.filter_by(email=user_email).first()
    
    if not user:
        flash('User not found!', 'danger')
        return redirect(url_for('home'))
    
    if request.method == 'POST':
        new_name = request.form['name']
        new_password = request.form['password']
        
        # Update the user's details in the database
        user.name = new_name
        user.password = new_password
        db.session.commit()
        
        flash('Profile updated successfully!', 'success')
        return redirect(url_for('home'))  # Redirect after successful update
    
    return render_template('update_profile.html', user=user)

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        name = request.form.get('name')
        email = request.form.get('email')
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')

        if not name or not email or not password or not confirm_password:
            flash("All fields are required!", "danger")
            return redirect(url_for('signup'))

        if password != confirm_password:
            flash("Passwords don't match!", "danger")
            return redirect(url_for('signup'))

        if not validate_email(email):
            flash("Invalid email format", "danger")
            return redirect(url_for('signup'))

        # Verify if email is already in the database
        if User.query.filter_by(email=email).first():
            flash("Email already exists!", "danger")
            return redirect(url_for('signup'))
        if not verify_email_exists(email):
             flash("Invalid email ", "danger")
             return redirect(url_for('signup'))


        # Create a new user
        new_user = User(name=name, email=email, password=password)
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

        if user is None:
            flash("Invalid email or password.", "danger")
            return redirect(url_for('login'))

        # Check if the password matches
        if user.password != password:
            flash("Invalid email or password.", "danger")
            return redirect(url_for('login'))

        flash("Login successful!", "success")
        return redirect(url_for('home'))  # Redirect to home page after successful login

    return render_template('login.html')




# Email validation function using Hunter.io
'''def verify_email_exists(email):
    hunter_api_key = "1bee4c88f13bd20373a8fa40d933f556319f92da"  # Replace with your actual API key
    url = f"https://api.hunter.io/v2/email-verifier?email={email}&api_key={hunter_api_key}"
    
    try:
        response = requests.get(url)
        data = response.json()
        if data['data']['result'] == 'deliverable':
            return True
        else:
            return False
    except Exception as e:
        print(f"Error verifying email: {e}")
        return False'''
import requests
def verify_email_exists(email):
    hunter_api_key = "b4229aa30ab3a569894b51c8b05b007641ce636e"  # Replace with your actual API key
    url = f"https://api.hunter.io/v2/email-verifier?email={email}&api_key={hunter_api_key}"
    
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        
        # Debugging: Print the full response
        print("Hunter.io API Response:", data)
        
        email_data = data.get('data', {})
        
        # Check the status field (preferred over deprecated 'result')
        status = email_data.get('status')
        
        if status == 'valid':
            return True  # Email is valid
        elif status == 'accept_all' and email_data.get('smtp_check') and email_data.get('mx_records'):
            return True  # Email might be valid for "accept_all" domains
        else:
            return False  # Risky or undeliverable email
    except requests.exceptions.RequestException as e:
        print(f"Request error: {e}")
        return False
    except Exception as e:
        print(f"Error verifying email: {e}")
        return False



# Email format validation using regex
def validate_email(email):
    email_regex = r'^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$'
    if re.match(email_regex, email):
        return True
    else:
        return False


from flask import Flask, render_template, request, jsonify
import os
from werkzeug.utils import secure_filename
import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_preprocess
from tensorflow.keras.applications.efficientnet import preprocess_input as efficientnet_preprocess

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'jfif'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load models
nail_model = tf.keras.models.load_model('nail_model.h5')  # EfficientNetB0 (224x224)
lip_model = tf.keras.models.load_model('angular_cheilitis_model.h5')  # MobileNetV2 (224x224)
tongue_model = tf.keras.models.load_model('tongue_condition_model.h5')  # Custom CNN (150x150)
eye_model = tf.keras.models.load_model('eye_condition_model.h5')  # EfficientNetB0 (224x224)
skin_model = tf.keras.models.load_model('skin_condition_model.h5')  # EfficientNetB0 (224x224)

# Define categories
nail_categories = ['beau lines', 'leukonychia', 'spooned nails']
lip_categories = ['angular cheilitis']
tongue_categories = ['glossitis', 'mouth ulcers', 'red color', 'smooth texture']
eye_categories = ['redness', 'glaucoma']
skin_categories = ['purpura']

deficiency_mapping = {
    'angular cheilitis': 'B1, B2, B3, Iron',
    'glossitis': 'B2, B3, B12',
    'red color': 'B12, Iron',
    'mouth ulcers': 'B12',
    'smooth texture': 'B12, Iron',
    'beau lines': 'B7, B9, Zinc',
    'leukonychia': 'Calcium, B7, B9',
    'spooned nails': 'C, B7, B9',
    'redness': 'Vitamin A Deficiency',
    'glaucoma': 'Deficiencies in folate, vitamin B12',
    'purpura': 'Vitamin C Deficiency'
}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def predict_category(image_path, model, categories, target_size, preprocess_fn):
    """Function to preprocess image, run model, and return predictions."""
    try:
        print(f"Processing image: {image_path}")

        img = Image.open(image_path).convert('RGB')  
        img = img.resize(target_size)
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        img_array = preprocess_fn(img_array)

        print("Image shape after preprocessing:", img_array.shape)

        # Check if the model requires flattened input
        if len(model.input_shape) == 2:
            img_array = img_array.reshape(1, -1)  # Flatten if needed

        predictions = model.predict(img_array)

        if predictions.shape[1] == 1:  # Binary classification (sigmoid output)
            scores = {categories[0]: predictions[0][0] * 100}
        else:  # Multi-class classification (softmax output)
            scores = {categories[i]: predictions[0][i] * 100 for i in range(len(categories))}

        print("Predictions:", scores)
        return scores

    except Exception as e:
        print(f"Error processing image: {e}")
        return None

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files or 'category' not in request.form:
        return jsonify({'error': 'No file or category selected'}), 400

    file = request.files['file']
    category = request.form.get('category')

    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        print(f"File saved at: {filepath}")

        # Model configurations based on category
        category_models = {
            'nails': (nail_model, nail_categories, (224, 224), efficientnet_preprocess),
            'lips': (lip_model, lip_categories, (224, 224), mobilenet_preprocess),
            'tongue': (tongue_model, tongue_categories, (150, 150), lambda x: x),  # Custom CNN (No special preprocessing)
            'eyes': (eye_model, eye_categories, (224, 224), efficientnet_preprocess),
            'skin': (skin_model, skin_categories, (224, 224), efficientnet_preprocess)
        }

        if category not in category_models:
            return jsonify({'error': 'Invalid category selected'}), 400

        model, categories, target_size, preprocess_fn = category_models[category]

        scores = predict_category(filepath, model, categories, target_size, preprocess_fn)

        if scores is None:
            return jsonify({'error': 'Error processing image'}), 500

        best_category = max(scores, key=scores.get)
        deficiency = deficiency_mapping.get(best_category, 'No deficiency detected')

        return jsonify({
            'predicted_category': best_category,
            'confidence': scores[best_category],
            'deficiency': deficiency
        })

    return jsonify({'error': 'Invalid file type'}), 400

if __name__ == '__main__':
    app.run(debug=True)

@app.route('/')
def index():
    return render_template('index.html')
'''@app.route('/login')
def login():
    return render_template('login.html')'''

@app.route('/symtom')
def symtom():
    return render_template('symptombased.html')

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
    app.run(debug=True)

