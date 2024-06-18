import os
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from PIL import Image
import numpy as np
import requests
from bs4 import BeautifulSoup
from keras.src.saving import load_model
from keras.src.utils import load_img, img_to_array


app = Flask(__name__, template_folder='templates')


# Load model and define labels
model = load_model('FV.h5')
labels = {0: 'apple', 1: 'banana', 2: 'beetroot', 3: 'bell pepper', 4: 'cabbage', 5: 'capsicum', 6: 'carrot',
          7: 'cauliflower', 8: 'chilli pepper', 9: 'corn', 10: 'cucumber', 11: 'eggplant', 12: 'garlic', 13: 'ginger',
          14: 'grapes', 15: 'jalepeno', 16: 'kiwi', 17: 'lemon', 18: 'lettuce', 19: 'mango', 20: 'onion', 21: 'orange',
          22: 'paprika', 23: 'pear', 24: 'peas', 25: 'pineapple', 26: 'pomegranate', 27: 'potato', 28: 'raddish',
          29: 'soy beans', 30: 'spinach', 31: 'sweetcorn', 32: 'sweetpotato', 33: 'tomato', 34: 'turnip',
          35: 'watermelon'}

fruits = ['Apple', 'Banana', 'Bell Pepper', 'Chilli Pepper', 'Grapes', 'Jalepeno', 'Kiwi', 'Lemon', 'Mango', 'Orange',
          'Paprika', 'Pear', 'Pineapple', 'Pomegranate', 'Watermelon']
vegetables = ['Beetroot', 'Cabbage', 'Capsicum', 'Carrot', 'Cauliflower', 'Corn', 'Cucumber', 'Eggplant', 'Ginger',
              'Lettuce', 'Onion', 'Peas', 'Potato', 'Raddish', 'Soy Beans', 'Spinach', 'Sweetcorn', 'Sweetpotato',
              'Tomato', 'Turnip']




def fetch_calories(prediction):
    try:
        url = 'https://www.google.com/search?q=calories+in+' + prediction + '&hl=en'
        req = requests.get(url).text
        scrap = BeautifulSoup(req, 'html.parser')
        calories = scrap.select('div.BNeawe.iBp4i.AP7Wnd')[0].text
        return calories
    except Exception as e:
        return "Can't fetch the calories"




# Process image function
def processed_img(img_path):
    img = load_img(img_path, target_size=(224, 224, 3))
    img = img_to_array(img)
    img = img / 255
    img = np.expand_dims(img, [0])
    answer = model.predict(img)
    y_class = answer.argmax(axis=-1)
    y = " ".join(str(x) for x in y_class)
    y = int(y)
    res = labels[y]
    return res.capitalize()


# Fetch recipes function
def fetch_recipes(prediction):
    try:
        url = 'https://www.google.com/search?q=' + prediction + '+recipes'
        req = requests.get(url).text
        scrap = BeautifulSoup(req, 'html.parser')
        recipes = scrap.find_all("div", class_="BNeawe vvjwJb AP7Wnd")
        recipes_list = [recipe.text for recipe in recipes[:5]]  # Get top 5 recipes
        return recipes_list
    except Exception as e:
        return []


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/about')
def about():
    return render_template('about.html')


# Ensure the 'static' directory exists
if not os.path.exists('static'):
    os.makedirs('static')


@app.route('/prediction', methods=['POST'])
def prediction():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join('static', filename)
        file.save(filepath)

        # Process the image file
        result = processed_img(filepath)
        category = 'Vegetables' if result in vegetables else 'Fruit'
        calories = fetch_calories(result)
        recipes = fetch_recipes(result)

        # Return the result as a JSON response
        return jsonify({
            'result': result,
            'category': category,
            'calories': calories,
            'recipes': recipes
        })
    


if __name__ == '__main__':
    app.run(debug=True, port=int(os.environ.get('PORT', 5000)), host='0.0.0.0')
