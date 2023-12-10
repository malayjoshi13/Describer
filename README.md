# Describer

Describer is an **image captioning system** which generates textual captions describing the images fed to it. It uses the InceptionV3 model to generate image embeddings and GloVe-6 Billion-200 dim model to generate textual-captions embeddings. These two embeddings then go to the scratch-made CNN model and long short-term memory model aka LSTM model (RNN). It is fine-tuned on ***Flickr8k data***

Got the following BLEU scores during model evaluation:<br>

BLEU-1: 0.468255,<br>
BLEU-2: 0.289801,<br>
BLEU-3: 0.207424,<br>
BLEU-4: 0.102605

![image](https://user-images.githubusercontent.com/71775151/192083201-035fc4c6-f1eb-42b0-ab68-1bc7942ad90a.png)
 
## 1) Bring your own image and generate the most suited caption describing it (aka Inference)
Check out this easy-to-use [inference.ipynb](https://github.com/malayjoshi13/Describer/blob/main/inference.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/malayjoshi13/Describer/blob/main/inference.ipynb). Upload your image, add its path in this notebook and get your captions!

## 2) Re-training on your own dataset and hyperparameters to improve it further
Check out this easy-to-train and evaluation [training.ipynb](https://github.com/malayjoshi13/Describer/blob/main/inference.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/malayjoshi13/Describer/blob/main/inference.ipynb). Run each code block of this script to train on default hyperparameter settings and the Flickr8K dataset. Otherwise, you can also choose your own dataset and value of hyperparameters within this script.

To re-train again on the Flick8K dataset with experimented hyperparameters, then:
- Create a directory in your Google Drive with the name --- `Describer` (or if you have already created one during inference, then skip this step and move down to the next one).
- Request the Flickr8k dataset from this link https://illinois.edu/fb/sec/1713398. Download and place it inside `Describer/dataset` folder in your Google Drive. 
- Now rename few files in `./dataset` folder to following filenames:-<br>
  - Flickr8k.token.txt to `captions.txt`. <br> 
  - Flickr_8k.trainImages.txt to `TrainImagesName.txt`. <br>
  - Flickr_8k.devImages.txt to `DevImagesName.txt`. <br>
  - Flickr_8k.testImages.txt to `TestImagesName.txt`. <br>
  - Flickr8k_Dataset contains to `All_images`.

## 3) Evaluating default trained weights or your own trained weights
Check out this easy-to-evaluate [evaluating.ipynb](https://github.com/malayjoshi13/Describer/blob/main/evaluating.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/malayjoshi13/Describer/blob/main/evaluating.ipynb). Run each code block of this script to evaluate default trained weights. Otherwise, you can also evaluate model weights trained by yourself.
