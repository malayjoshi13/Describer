#We will first set all the imports we need, only import referenced libraries that will help us keep the code light.
import os
from PIL import Image
import re
import gc
import urllib
from flask import Flask, render_template,  url_for, request

import numpy as np
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from pickle import load
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences


#This specifies the name of our Flask application, __name__ is actually the name of our file i.e. app.py
app = Flask(__name__)

#parameter values we used from our ML code and we will use it as static value as our model is trained on these values.
max_length = 34
vocab_size = 1798

directory = os.getcwd()

filename = 'word_to_index.pkl'
word_to_index = load(open(filename, 'rb'))

filename1 = 'index_to_word.pkl'
index_to_word = load(open(filename1, 'rb'))

filename2 = 'embedding_matrix.pkl'
embedding_matrix = load(open(filename2, 'rb'))

# here we are loading our model which is saved under the name ‘model.h5’
trained_model = load_model('modelzz.h5')

# as we already saw that we made use of Inception V3 model to vectorize our images, 
# so for new images also we need to vectorize them using the same, so the config should match with our trained ML model.
pre_trained_model = InceptionV3(weights='imagenet')
encoder = Model(pre_trained_model.input, pre_trained_model.layers[-2].output) 

# This specifies our home page, Input will be taken from this page and will be passed on to our server where the code is residing for execution. Whenever someone will hit our server they will be navigated to this page by default, because of ‘/’.
# @app.route is an annotation of flask, it helps us to route pages which is mentioned in the code. render_template helps us to send the specified html pages to client’s browser.
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():
 if request.method == 'POST':    
    image_list = os.listdir("static/")
    for i in image_list:
        os.remove("static/"+i)
    
    url = request.files['userfile']
    imageName = "static/{}".format(url.filename)
    url.save(imageName)
    
    img = Image.open(imageName)
    img = img.resize((299,299), Image.ANTIALIAS)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)    
    test_img_encoding = encoder.predict(img)
    
    caption = 'startseq'
    for i in range(max_length):
        sequence = [word_to_index[w] for w in caption.split() if w in word_to_index]
        sequence = pad_sequences([sequence], maxlen=max_length)
        inputs = [test_img_encoding,sequence]
        yhat = trained_model.predict(x=inputs, verbose=0)
        yhat = np.argmax(yhat)
        word = index_to_word[yhat]
        caption += ' ' + word
        if word == 'endseq':
            break
    final = caption.split()
    final = final[1:-1]
    final_caption = ' '.join(final)
    predict = re.sub(r'\b(\w+)( \1\b)+', r'\1', final_caption)    
    return render_template('result.html', prediction = predict, urlImg = url)
   


if __name__ == '__main__':
    app.run(debug=True)