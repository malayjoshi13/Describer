# Describer

Describer is an **image captioning system** which generates textual captions describing the images fed to it. It uses the InceptionV3 model to generate image embeddings and GloVe-6 Billion-200 dim model to generate textual-captions embeddings. These two embeddings then go to the scratch-made CNN model and long short-term memory model aka LSTM model (RNN). It is fine-tuned on ***Flickr8k data***

Got the following BLEU scores during model evaluation, via Greedy Search approach:<br>

BLEU-1: 0.795413<br>
BLEU-2: 0.666974<br>
BLEU-3: 0.581795<br>
BLEU-4: 0.399388

![image](https://user-images.githubusercontent.com/71775151/192083201-035fc4c6-f1eb-42b0-ab68-1bc7942ad90a.png)
 
## 1) Bring your own image and generate the most suited caption describing it (aka Inference)
Check out this easy-to-use script [inference.ipynb](https://github.com/malayjoshi13/Describer/blob/main/scripts/inference.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/malayjoshi13/Describer/blob/main/scripts/inference.ipynb). Upload your image, add its path in this notebook and get your captions!

## 2) Re-training on your own dataset and hyperparameters to improve it further
Check out this easy-to-train script [training.ipynb](https://github.com/malayjoshi13/Describer/blob/main/scripts/training.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/malayjoshi13/Describer/blob/main/scripts/training.ipynb). Run each code block of this script to train on default hyperparameter settings and the Flickr8K dataset. Otherwise, you can also choose your own dataset and value of hyperparameters within this script.

<br>

To re-train again on the Flick8K dataset with your own set of hyperparameter values, then:
- Create a directory in your Google Drive with the name `Describer` (skip this step if you have already cloned this Github repo in your Google Drive during the inference/default training/evaluating process).
- Request the Flickr8k dataset from this link https://illinois.edu/fb/sec/1713398. Download and place it inside the `Describer/dataset` folder in your Google Drive. 
- Now rename few files in `./dataset` folder to following filenames:-<br>
  - Flickr8k.token.txt to `captions.txt`. <br> 
  - Flickr_8k.trainImages.txt to `TrainImagesName.txt`. <br>
  - Flickr_8k.devImages.txt to `DevImagesName.txt`. <br>
  - Flickr_8k.testImages.txt to `TestImagesName.txt`. <br>
  - Flickr8k_Dataset contains to `All_images`.
 
A quick hack: as the Flickr8K dataset can't be distributed you have to individually get access to it. In my case, once got access and downloaded it in my GDrive, then whenever I need the dataset, I simply copy Gdrive link of `./dataset` folder from my main folder (in GDrive) and then use it in any my other GDrive account. To use in other GDrive account, I paste the copied GDrive link to my another account and ```create a copy/shortcut``` of `./dataset` folder in the directory of my cloned GitHub repo. This way I don't need to upload the dataset every time I work in another GDrive account.

## 3) Evaluating default trained weights or your own trained weights
Check out this easy-to-evaluate script [evaluating.ipynb](https://github.com/malayjoshi13/Describer/blob/main/scripts/evaluating.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/malayjoshi13/Describer/blob/main/scripts/evaluating.ipynb). Using this you can either evaluate default trained model or model re-trained by you. In both cases, you must have the `./dataset` folder within your working directory (if evaluating on Flickr8K, follow above steps).
