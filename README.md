# Describer

Describer is an **image captioning system** comprising of ***InceptionV3 model*** (CNN) and ***long short term memory model aka LSTM*** (RNN) fine-tuned on ***Flickr8k data***, which generates textual captions describing the images feed to it.

Got the following BLEU scores during model evaluation:<br>

BLEU-1: 0.468255,<br>
BLEU-2: 0.289801,<br>
BLEU-3: 0.207424,<br>
BLEU-4: 0.102605

![image](https://user-images.githubusercontent.com/71775151/192083201-035fc4c6-f1eb-42b0-ab68-1bc7942ad90a.png)

## 1) Getting started 

### 1.1) Setting up for evaluation and inference
- Create a directory in your Google Drive with name --- `Describer`.
- In this directory, create a shortcut of the `pre_trained_model` folder from following [link](https://drive.google.com/drive/folders/1Ve9oPapUVvnLTVH2z96ubXPSswyuU3-C?usp=sharing). This folder has needed files and pre-trained model weights, which you will need during inference for generating captions for your images as well as to evaluate the trained model.
  
### 1.2) Setting up for training
- Create a directory in your Google Drive with name --- `Describer` (or if you have already created one during inference, then skip this step and move down to the next one).
- Request the Flickr8k dataset from this link https://illinois.edu/fb/sec/1713398. Download and place it inside `Describer/dataset` folder in your Google Drive. 
- Now rename few files in `./dataset` folder to following filenames:-<br>
  - Flickr8k.token.txt to `captions.txt`. <br> 
  - Flickr_8k.trainImages.txt to `TrainImagesName.txt`. <br>
  - Flickr_8k.devImages.txt to `DevImagesName.txt`. <br>
  - Flickr_8k.testImages.txt to `TestImagesName.txt`. <br>
  - Flickr8k_Dataset contains to `All_images`.
    
## 2) Bring your own image and generate the most suited caption describing it (aka Inference)
Check out this easy-to-use [Google Colab Notebook](https://colab.research.google.com/drive/1HIpLysJeD401qB8bayn7sKXehEQUzl8L?usp=sharing). Upload your image, add its path in this notebook and get your captions!

## 3) Training on your own dataset and hyperparameters to improve it further
Check out this easy-to-train and evaluation [Google Colab Notebook](https://colab.research.google.com/drive/1uE59v-rfCzwTqnG2T7kGYPAdYLEwGiqM?usp=sharing). Run each code block of this script to train on default hyperparameter settings and the Flickr8K dataset. Otherwise, you can also choose your own dataset and value of hyperparameters within this script.

## 4) Evaluating default trained weights or your own trained weights
Check out this easy-to-evaluate [Google Colab Notebook](https://colab.research.google.com/drive/1qm6776oQAgWK-pjFvhw3HB4uBPxUEaNA?usp=sharing). Run each code block of this script to evaluate default trained weights. Otherwise, you can also evaluate model weights trained by yourself.
