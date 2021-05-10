# -*- coding: utf-8 -*-
"""**STEP 1) CONNECTING GOOGLE COLAB TO GOOGLE DRIVE**"""

from google.colab import drive
drive.mount("/content/drive")

"""**STEP 2) IMPORTING IMPORTANT LIBRARIES**"""

import numpy as np

from tensorflow.keras.preprocessing.image import load_img, img_to_array

from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input

from tensorflow.pickle import load

import matplotlib.pyplot as plt

from tensorflow.keras.models import Model,load_model

from tensorflow.keras.preprocessing.sequence import pad_sequences

"""**STEP 3) SOME VALUES TO BE USED HERE**"""

max_length = 34 
vocab_size = 1798

"""**STEP 4) LOADING PICKELED FILES OF "word_to_index", "index_to_word" and "embedding_matrix" WHICH WE HAVE SAVED DURING TRAINING PERIOD**"""

filename = '/content/drive/My Drive/Describe/word_to_index.pkl'
word_to_index = load(open(filename, 'rb'))

filename1 = '/content/drive/My Drive/Describe/index_to_word.pkl'
index_to_word = load(open(filename1, 'rb'))

filename2 = '/content/drive/My Drive/Describe/embedding_matrix.pkl'
embedding_matrix = load(open(filename2, 'rb'))

"""**STEP 5) PUT HERE PATH OF TEST IMAGE (FOR WHICH YOU WANT TO GENERATE CAPTION)**"""

filename =  '/content/drive/My Drive/Describe/aa.jpg'

"""**STEP 6) WILL GENERATE ENCODING OF TEST IMAGE BY InceptionV3**"""

pre_trained_model = InceptionV3(weights='imagenet')
encoder = Model(pre_trained_model.input, pre_trained_model.layers[-2].output) 

img = load_img(filename, target_size=(299, 299))
img = img_to_array(img)
img = np.expand_dims(img, axis=0)
img = preprocess_input(img)
test_img_encoding = encoder.predict(img)

"""**STEP 7) WILL KICK-START CAPTION GENERATION PROCESS**"""

trained_model = load_model('/content/drive/My Drive/Describe/modelzz.h5')

caption = 'startseq'

for i in range(max_length):
    sequence = [word_to_index[w] for w in caption.split() if w in word_to_index]
    sequence = pad_sequences([sequence], maxlen=max_length)
    yhat = trained_model.predict([test_img_encoding,sequence], verbose=0)
    yhat = np.argmax(yhat)
    word = index_to_word[yhat]
    caption += ' ' + word
    if word == 'endseq':
        break
final = caption.split()
final = final[1:-1]
final_caption = ' '.join(final)

"""**STEP 8) WILL SHOW TEST-IMAGE AND IT's CAPTION**"""

x=plt.imread(filename)
plt.imshow(x)
plt.show()
print(final_caption)