'''import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import os
import shutil

# Define dataset paths
dataset_path = "finaldata" 
train_path = os.path.join(dataset_path, "train")
val_path = os.path.join(dataset_path, "val")

# Automatically split data if validation set doesn't exist
if not os.path.exists(val_path):
    os.makedirs(train_path, exist_ok=True)
    os.makedirs(val_path, exist_ok=True)
    for class_name in os.listdir(dataset_path):
        class_dir = os.path.join(dataset_path, class_name)
        if os.path.isdir(class_dir) and class_name not in ["train", "val"]:
            images = os.listdir(class_dir)
            split_idx = int(len(images) * 0.8)  # 80% train, 20% validation
            train_images = images[:split_idx]
            val_images = images[split_idx:]
            os.makedirs(os.path.join(train_path, class_name), exist_ok=True)
            os.makedirs(os.path.join(val_path, class_name), exist_ok=True)
            for img in train_images:
                shutil.copy(os.path.join(class_dir, img), os.path.join(train_path, class_name, img))
            for img in val_images:
                shutil.copy(os.path.join(class_dir, img), os.path.join(val_path, class_name, img))

# Data Augmentation and Generators
train_datagen = ImageDataGenerator(rescale=1./255, rotation_range=30, zoom_range=0.2, horizontal_flip=True)
val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(train_path, target_size=(224, 224), batch_size=32, class_mode='categorical')
val_generator = val_datagen.flow_from_directory(val_path, target_size=(224, 224), batch_size=32, class_mode='categorical')

# CNN Model
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(224, 224, 3)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(train_generator.num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train Model
epochs = 20
history = model.fit(train_generator, validation_data=val_generator, epochs=epochs)

# Save Model
model.save("deficiency_classifier.h5")

print("Model training complete and saved as deficiency_classifier.h5")'''

'''compressing original h5 file into tflite file to reduc size'''
import tensorflow as tf

# Load the trained model from .h5
model = tf.keras.models.load_model("deficiency_classifier.h5")

#  Save Model in TensorFlow SavedModel format (Corrected)
saved_model_dir = "saved_model/deficiency_classifier"
model.export(saved_model_dir)
print("Model saved in SavedModel format.")

# Convert to TensorFlow Lite (TFLite) format
converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
tflite_model = converter.convert()

# Save the TFLite model
with open("deficiency_classifier.tflite", "wb") as f:
    f.write(tflite_model)
print("TFLite model saved.")

# Apply Post-Training Quantization for Further Size Reduction
converter.optimizations = [tf.lite.Optimize.DEFAULT]  # Enable quantization
tflite_quantized_model = converter.convert()

# Save the quantized model
with open("deficiency_classifier_quantized.tflite", "wb") as f:
    f.write(tflite_quantized_model)
print("Quantized TFLite model saved.")











