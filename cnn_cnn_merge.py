from flask import Flask, request, jsonify
from PIL import Image
from keras.preprocessing.sequence import pad_sequences
import numpy as np
import requests
from io import BytesIO
from keras.models import load_model
from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.preprocessing.image import img_to_array
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

app = Flask(__name__)

# Load the Keras model
model = load_model('my_modelcnn_cnn_merge1.h5')

# Load ResNet50 model for feature extraction
resnet_model = ResNet50()
resnet_model = Model(inputs=resnet_model.inputs, outputs=resnet_model.layers[-2].output)

# Function to extract features from an image
def extract_features(image, model):
    image = image.resize((224, 224))
    image = img_to_array(image)
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    image = preprocess_input(image)
    feature = model.predict(image, verbose=0)
    return feature.flatten()  # Flatten the feature array

# Function to generate captions for the image
def generate_caption(model, photo, max_length=18, reference_sentence=None):  
    in_text = '<start>'
    for _ in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)
        yhat = model.predict([photo.reshape(1, -1), sequence], verbose=0)
        yhat = np.argmax(yhat)
        word = tokenizer.index_word[yhat]
        if word is None:
            break
        in_text += ' ' + word
        if word == '<end>':
            break
    final = in_text.split()
    final = final[1:-1]
    final = ' '.join(final)
    
    # Calculate BLEU score
    if reference_sentence:
        reference = reference_sentence.split()
        candidate = final.split()
        smoother = SmoothingFunction().method1
        bleu_score = sentence_bleu([reference], candidate, smoothing_function=smoother)
    else:
        bleu_score = None
    
    return final, bleu_score

# API endpoint for generating caption
@app.route('/generate_caption', methods=['POST'])
def get_caption():
    img_url = request.json.get('img_url')
    reference_sentence = request.json.get('reference_sentence')
    response = requests.get(img_url)
    img = Image.open(BytesIO(response.content))
    photo = extract_features(img, resnet_model)
    caption, bleu_score = generate_caption(model, photo, reference_sentence=reference_sentence)
    return jsonify({'caption': caption, 'bleu_score': bleu_score})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
