This file contains all the topics that I learned while working on this project.

# Flickr8k

Request the dataset from this link https://illinois.edu/fb/sec/1713398. The dataset consists of 8,000 images that are each paired with five different captions. This whole dataset comes into following files and folders:

Flickr8k.token.txt contains the raw captions of the Flickr8k Dataset. For our project we renamed it as `captions.txt`. <br> 
Flickr_8k.trainImages.txt contains the names of training images. For our project we renamed it as `TrainImagesName.txt`. <br>
Flickr_8k.devImages.txt contains the names of validation images. For our project we renamed it as `DevImagesName.txt`. <br>
Flickr_8k.testImages.txt contains the names of test images. For our project we renamed it as `TestImagesName.txt`. <br>
Flickr8k_Dataset contains all the images (train+val+test). For our project we renamed it as `All_images`.

![image](https://user-images.githubusercontent.com/71775151/192083276-df0a8530-3966-49fd-ad5b-7e0dc19990ff.png)

# Glove
Since the number of unique words can be very large, thus doing one hot encoding of the words is not a good idea. Therefore, a pre-trained embedding model called ```GloVe``` is used that takes every word of every training caption and outputs the corresponding word embedding vector. 

