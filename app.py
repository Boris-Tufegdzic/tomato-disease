from flask import Flask, request, jsonify, render_template
from PIL import Image
import torch
import torchvision
from torchvision import transforms
import io
from model.utils.predict import pred

# Initialize the app
app = Flask(__name__)

# Loading the model
MODEL_PATH = "model/models/efficientnet_b3.pth"
device = "cuda" if torch.cuda.is_available() else "cpu"
if torch.cuda.is_available():
    weights = torch.load(MODEL_PATH)
else:
    weights = torch.load(MODEL_PATH, map_location=torch.device('cpu'))
model = torchvision.models.efficientnet_b3(weights=weights).to(device)
model.classifier = torch.nn.Sequential(
        torch.nn.Dropout(p=0.2, inplace=True), 
        torch.nn.Linear(in_features=1536, 
                        out_features=11, # same number of output units as our number of classes
                        bias=True)).to(device)

# Define routes
@app.route('/')
def index():
    return render_template('index.html')  # simple HTML form for uploading images

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # Read the image
    try:
        img_path = "static/" + file.filename	
        file.save(img_path)
        predicted_class = pred(model, img_path)
        class_mapping = {0: 'Bacterial_spot', 1: 'Early_blight', 
                         2: 'Late_blight', 3: 'Leaf_Mold', 
                         4: 'Septoria_leaf_spot', 5: 'Spider_mites Two-spotted_spider_mite', 
                         6: 'Target_Spot', 7: 'Tomato_Yellow_Leaf_Curl_Virus', 
                         8: 'Tomato_mosaic_virus', 9: 'healthy', 10: 'powdery_mildew'}
        if predicted_class.item() in class_mapping:
            result = class_mapping[predicted_class.item()]
        else:
            print(f"Predicted class {predicted_class} is not in class_mapping")
            return jsonify({'error': f'Invalid predicted class: {predicted_class}'}), 500
        return render_template('result.html', prediction=result)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Run the app
if __name__ == "__main__":
    app.run(debug=True)
