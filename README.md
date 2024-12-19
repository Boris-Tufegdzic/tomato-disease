# Tomato Disease Detection App

This project is a web-based application that predicts whether a tomato is healthy or diseased based on an uploaded image. The app uses a pre-trained EfficientNet model to classify tomato plant diseases.


### File Descriptions

- **`train.py`:** This script handles training the machine learning model using the data in `data/train` and `data/valid`.
  
- **`models/efficientnet_b3.pth`:** This file contains the pre-trained model weights (EfficientNet-B3) for tomato disease classification.

- **`utils/predict.py`:** Contains functions for loading the model and making predictions based on the input image.

- **`utils/setup_data.py`:** Defines functions for creating PyTorch `DataLoader` objects to easily load and preprocess the image datasets.

- **`utils/training_utils.py`:** Contains helper functions for model training and validation, such as performing a full training step

- **`app.py`:** The Flask web server that provides an interface for users to upload images and get disease predictions.

- **`static/`:** Contains static assets used in the web app, such as stylesheets, JavaScript files, and images.

- **`templates/`:** Stores the HTML templates that define the front-end of the web app.

- **`requirements.txt`:** Specifies the Python packages required to run the app, making it easy to set up a virtual environment.

## Installation

1. **Clone the repository:**

   git clone 
   cd tomato-disease

2. Set up a virtual environment:

    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`

3. Install the required dependencies:

pip install -r requirements.txt

4. Run the Flask app:

python app.py

### Usage

1. Training the Model:

If you want to train the model from scratch, use the train.py script: python model/train.py

You also have to download the data from : https://www.kaggle.com/datasets/cookiefinder/tomato-disease-multiple-sources
And place it in a "data" folder

This will use the data from the data/train and data/valid directories.

2. Prediction:

Upload an image through the web interface, and the model will predict whether the tomato is healthy or has a disease. Supported diseases include:

    Bacterial Spot
    Early Blight
    Late Blight
    Leaf Mold
    Septoria Leaf Spot
    Spider Mites
    Target Spot
    Tomato Yellow Leaf Curl Virus
    Tomato Mosaic Virus
    Healthy

### Model

The model is a pre-trained EfficientNet-B3 model, fine-tuned for tomato disease classification. The model is saved in the models/ directory as efficientnet_b3.pth.

### Dependencies

The dependencies are listed in requirements.txt
