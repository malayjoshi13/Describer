This file contains all the topics that I learned while working on this project.

# 3 possible structures of image-captioning models:-

1) Generate the Whole Sequence

![image](https://user-images.githubusercontent.com/71775151/194770885-f68100eb-b263-4074-aa36-ef0e1658f680.png)

This is a one-to-many sequence prediction model that generates the entire output in a one-shot manner given the input photograph. This model puts a heavy burden on the language model to generate the right words in the right order. Sometimes a same word is repeated in the whole output sentence.

2) Generate Word from Word

![image](https://user-images.githubusercontent.com/71775151/194771051-e16c63fc-9ffa-46c8-a90d-faccb623e0af.png)

This is a one-to-one sequence prediction model where the LSTM generates a prediction of one word given a photograph and one word as input. The recursive word generation process is repeated until an end of sequence token is generated.

Input 1: Photograph <br>
Input 2: One word of sequence. This one word input is either a token to indicate the start of the sequence in the case of the first time the model is called, or is the word generated from the previous time the model was called <br>
Output: Next word in sequence <br>

The model does generate some good n-gram sequences, but gets caught in a loop repeating the same sequences of words for long descriptions. There is insufficient memory in the model to remember what has been generated previously.

3) Generate Word from Sequence

![image](https://user-images.githubusercontent.com/71775151/194771085-4a1871b1-a875-412a-b2ce-61f6601547a3.png)

This is a many-to-one sequence prediction model where given a photograph and a sequence of words already generated for the photograph as input, predict the next word in the description.

Input 1: Photograph <br>
Input 2: Previously generated sequences of words, or start of sequence token <br>
Output: Next word in sequence <br>

It is a generalization of the above Model 2 where the input sequence of words gives the model a context for generating the next word in the sequence. The model does readily generate readable descriptions, the quality of which is often refined by larger models trained for longer. We have used this kind of model in Describe.

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

# Saving weights or whole model?
Models saved in .hdf5 format are great because the whole model is one place and can be loaded somewhere else, such as in deployment. However the files can get large, and saving your model at every epoch can get storage intensive fast. One option available in the ModelCheckpoint callback constructor is save_weights_only=True. This will save space, but will not save the entire model architecture. In order to recover it, you would rebuild the model and then assign the saved weights, rather than just loading it all in one step.

https://machinelearningmastery.com/save-load-keras-deep-learning-models/

If you only save weights, then do as below:-
![image](https://user-images.githubusercontent.com/71775151/192160237-2ad1f5f1-bac8-4936-b59a-02555666d311.png)

# Model checkpoint
While using model checkpoint, I used minimum val loss as paramater to save weights(if save_weights_only=True) or whole model(if save_weights_only=False), coz such models are more tend to overfitting, which can be prevented if both val and train losses are nearly equal and minimum. So once we save all min val losses, we will see out of all the min val losses, at which min val loss, train loss is also min and nearly equal to the min val loss. 

I have not used min train loss as parameter, coz this model get easily overfit that means train loss keeps dec but val loss after a point starts inc. So point where val loss will start inc and will not come down even after 10 iterations, there will stop training via `earlystopping`. then we see out of all min val loss, at which val loss, train loss is also min and nearly equal to val loss. Weight corresponding to that moment is final answer.

Now in order to save weights at every iteration I use `weights-improvement-{epoch:02d}-{val_accuracy:.2f}.hdf5` as file name of saved weights as being unique, the newer weights will not replace already saved prev weights. But if I would wanted to not save weights in every iterations but just to save the best one then I would use `weights.best.hdf5` as filename, coz being unique filename of weights, new weights will replace prev weights thus saving only one best weight not all best ones.

# Use EarlyStopping Together with Checkpoint
In the examples above, an attempt was made to fit your model with 150 epochs. In reality, it is not easy to tell how many epochs you need to train your model. One way to address this problem is to overestimate the number of epochs. But this may take significant time. After all, if you are checkpointing the best model only, you may find that over the several thousand epochs run, you already achieved the best model in the first hundred epochs, and no more checkpoints are made afterward.

It is quite common to use the ModelCheckpoint callback together with EarlyStopping. It helps to stop the training once no metric improvement is seen for several epochs.

# Data generator

# BLEU score
Having high BLEU score always dont tells model is very good. Coz high BLEU score can also happen due to overfitting. In such case model will catch very minute details of image but caption generated will not be good. **Thus our aim should be to achieve high BLEU without overfitting model**. 

# Validation helps to see if overfitting dont happens. 
Actually validation loss is increasing after 5th epoch but training loss keeps dec. It shows that overfitting is happening.

# Future 
gridsearchCV,<br>
cross-validation instead of validation (https://www.quora.com/What-is-the-difference-between-validation-and-cross-validation , https://www.google.com/search?q=k+fold+cross+validation+vs+shuffle&client=ms-android-samsung-ga-rev1&sxsrf=ALiCzsblth3GLcpe7H38ZuL6XrCHITE6Mw%3A1665321266679&ei=MslCY-79KKuB3LUPw7-kqAc&oq=k+fold+cross+validation+vs+shuffl&gs_lcp=ChNtb2JpbGUtZ3dzLXdpei1zZXJwEAEYADIHCCEQoAEQCjIHCCEQoAEQCjIHCCEQoAEQCjoKCAAQRxDWBBCwAzoECCMQJzoGCAAQHhAWOgUIABCGAzoFCAAQgAQ6BwgAEIAEEAo6BQghEKABOggIIRAeEBYQHUoECEEYAFCrFViJQ2DDVGgBcAB4AIABpQOIAcwhkgEIMi0xNS4xLjGYAQCgAQHIAQjAAQE&sclient=mobile-gws-wiz-serp),<br>
use larger dataset,<br>
attention model

