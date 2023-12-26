# Describer

Describer is an **image captioning system** which generates textual captions describing the images fed to it. The system is trained on ***Flickr8k data***.

It uses ***Merge Architecture*** where the pre-trained ***InceptionV3*** convolutional neural network generates image embeddings and the ***GloVe*** (having 6 Billion pairs of words and their corresponding 200 dim representational vectors) initialized Embedding layer generates caption's word embeddings. Then, image embeddings go to a ***dense layer*** which compresses image embeddings into 256 dim and word embeddings go to ***LSTM*** recurrent neural network which outputs 256 dim representation. These two embeddings/representations then get added together and fed to the ***Feed-forward network*** which outputs the next word of the caption. 

Got the following BLEU scores during model evaluation, via Greedy Search approach:<br>

| BLEU-1 score | 0.79 |<br>
| BLEU-2 score | 0.66 |<br>
| BLEU-3 score | 0.58 |<br>
| BLEU-4 score | 0.39 |<br>

![image](https://user-images.githubusercontent.com/71775151/192083201-035fc4c6-f1eb-42b0-ab68-1bc7942ad90a.png)
 
## Get your image captioned!! (aka Inference)
Check out this easy-to-use script [inference.ipynb](https://github.com/malayjoshi13/Describer/blob/main/scripts/inference.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/malayjoshi13/Describer/blob/main/scripts/inference.ipynb). Upload your image, add its path in this notebook and get your captions!

## Re-training - on your own dataset or set of hyperparameters or both 
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

## Evaluating - default trained model or your own trained model
Check out this easy-to-evaluate script [evaluating.ipynb](https://github.com/malayjoshi13/Describer/blob/main/scripts/evaluating.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/malayjoshi13/Describer/blob/main/scripts/evaluating.ipynb). Using this you can either evaluate default trained model or model re-trained by you. In both cases, you must have the `./dataset` folder within your working directory (if evaluating on Flickr8K, follow above steps).

## End-note
Thank you for patiently reading till here. I am pretty sure just like me, you would have also learnt something new about integrating the capabilities of CNN and RNN models to build a real-world application to help visually impaired people to understand visual data around them. Using these learnt concepts I will push myself to scale this project further to improve captioning capabilities. I encourage you also to do the same!!

## Contributing
You are welcome to contribute to the repository with your PRs. In case of query or feedback, please write to me at 13.malayjoshi@gmail.com or https://www.linkedin.com/in/malayjoshi13/.

## Licence

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/malayjoshi13/Describer/blob/main/LICENSE)
