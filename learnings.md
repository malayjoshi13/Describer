This file contains all the topics that I learned while working on this project.

# Flickr8k

Request the dataset from this link https://illinois.edu/fb/sec/1713398. The dataset consists of 8,000 images (out of which 6000 are training images, 1000 are validation images and 1000 are testing images) that are each paired with five different captions. This whole dataset comes into following files and folders:

Flickr8k.token.txt contains the raw captions of the Flickr8k Dataset. <br> 
Flickr_8k.trainImages.txt contains the names of training images. <br>
Flickr_8k.devImages.txt contains the names of validation images. <br>
Flickr_8k.testImages.txt contains the names of test images. <br>
Flickr8k_Dataset contains all the images (train+val+test). 

![image](https://user-images.githubusercontent.com/71775151/192083276-df0a8530-3966-49fd-ad5b-7e0dc19990ff.png)

# Glove
Since the number of unique words can be very large, thus doing one hot encoding of the words is not a good idea. Therefore, a pre-trained embedding model called ```GloVe``` is used that takes every word of every training caption and outputs the corresponding word embedding vector. 

# Verbose in model.fit
![image](https://user-images.githubusercontent.com/71775151/192151776-6162f1c8-46b3-4794-8c13-6f67aa8e688d.png)

# Training loss and Validation loss
In machine learning and deep learning there are basically three cases

1) Underfitting

When training loss > validation_loss, as this means model is not even able to learn patterns in training data.

2) Overfitting

When loss << validation_loss, as this means that your model is fitting very nicely to training data but not at all generalizing to the validation data (which is unseen data)

3) Perfect fitting

loss == validation_loss + both the values are converging/decreasing (plot the loss over time) then chances are very high that you are doing it right

