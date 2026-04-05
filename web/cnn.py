from flask import Flask, request, render_template, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
app = Flask(__name__)
model = load_model("/home/sns/pythonworkspace2/PROJET_DATAMINING/brain_tumor_model.h5")
classes = ['glioma', 'meningioma', 'notumor', 'pituitary']
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
@app.route("/")
def index():
    return render_template("index.html")
@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400
    
    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)
    
    img = image.load_img(filepath, target_size=(64, 64))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    
    pred = model.predict(img_array)
    predicted_class = classes[np.argmax(pred)]
    
    return jsonify({"prediction": predicted_class})

if __name__ == "__main__":
    app.run(debug=True)