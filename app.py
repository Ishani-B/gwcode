from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
import os
from PIL import Image
import json
from werkzeug.utils import secure_filename

app = Flask(__name__)
model = load_model('models/clothing_sorter.h5')

# Mapping of class indices to clothing categories
categories = ['dress', 'longsleeve', 'outwear', 'pants', 'shirt', 'shoes', 'shorts', 'skirt', 't-shirt']

# Ensure the upload folder exists
upload_folder = 'static/uploads/'
if not os.path.exists(upload_folder):
    os.makedirs(upload_folder)

# Path to the JSON file
closet_file = 'closet.json'

def load_closet():
    with open(closet_file, 'r') as file:
        return json.load(file)

def save_closet(closet_data):
    with open(closet_file, 'w') as file:
        json.dump(closet_data, file, indent=4)

def get_total_items(closet):
    return sum(closet.values())

def get_class_percentage(closet, clothing_class):
    total_items = get_total_items(closet)
    if total_items == 0:
        return 0
    return (closet[clothing_class] / total_items) * 100

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files.get('file')
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(upload_folder, filename)
            file.save(filepath)
            
            try:
                # Load and preprocess the image
                image = Image.open(filepath)
                image = image.resize((256, 256))
                image = img_to_array(image) / 255.0
                image = np.expand_dims(image, axis=0)

                # Make prediction
                prediction = model.predict(image)
                predicted_class = np.argmax(prediction, axis=1)[0]
                result = categories[predicted_class]

                # Load the current closet data
                closet = load_closet()

                # Check the total number of items
                total_items = get_total_items(closet)
                if total_items >= 50:
                    return jsonify({
                        'message': "Too many clothes, don't buy",
                        'prediction': result,
                        'confidence': float(np.max(prediction)),
                        'closet_counts': closet
                    })

                # Check if any class exceeds 25% of the total items
                class_percentage = get_class_percentage(closet, result)
                if class_percentage > 25:
                    return jsonify({
                        'message': f"More than 25% of your closet is {result}, perhaps focus on building other parts.",
                        'prediction': result,
                        'confidence': float(np.max(prediction)),
                        'closet_counts': closet
                    })

                # Check if the user indicated they are buying the item
                if 'buy_item' in request.form and request.form['buy_item'] == 'yes':
                    # Update the count for the predicted class
                    closet[result] += 1
                    save_closet(closet)

                # Get the updated count for all items
                return jsonify({ 
                    'prediction': result,
                    'confidence': float(np.max(prediction)),
                    'closet_counts': closet
                })
            except Exception as e:
                return jsonify({'error': str(e)})
        else:
            return jsonify({'error': 'Invalid file format'})
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
