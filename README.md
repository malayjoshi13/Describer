# Describe
Describe is a machine learning based image captioning system which by help of a pre-trained ***convolutional neural network model (CNN)*** of ***"InceptionV3" model*** (fine tuned on data from Flickr8k link:-https://forms.illinois.edu/sec/1713398 provided by the University of Illinois at Urbana-Champaign) and a ***recurrent neural network (RNN)*** of ***long short term memory model (LSTM)***  generates text based captions corresponding to images feeded to it. 

Users can further extend accuracy of this model by re-training it on new annotated dataset starting from the weights of existing model.

This system paves the path towards development of "automated guiding system" which will guide people with visual disability to cross busy roads and to perform their other day to day based tasks. It will do so by taking video inputs of user's surroundings from camera fitted in their headwear and then converting these inputs into audio based captions/descriptions.
<br><br>

## 1) What makes Decribe work?

Describe uses both Natural Language Processing and Computer Vision to generate the captions/textual description of an image. 

![explain_2](https://user-images.githubusercontent.com/71775151/114168565-0f697380-994e-11eb-98d3-7db106d58718.jpg)

It comprises of three main components:

**1.1) Convolutional Neural Network (CNN) acting as Encoder:** A pre-trained CNN known as ```InceptionV3``` is used as an encoder to encode an input image to its features vector known as image feature vector. These image features vectors of dimention (4096,) are obtained from the second last layer by removing the last softmax layer of InceptionV3.<br>

![Schematic-diagram-of-InceptionV3-model-compressed-view](https://user-images.githubusercontent.com/71775151/114168447-e47f1f80-994d-11eb-830b-aa212eadebc2.jpg)

**1.2) Word embedding model:** Since the number of unique words can be large, a one hot encoding of the words is not a good idea. Thus, a pre-trained embedding model ```GloVe``` is used that takes every word of every training caption and outputs an word embedding vector of dimension (1, 128). 

These outputs of ```image feature vectors``` and ```word embedding vectors``` go further into a function called as ```data_generator``` (will discuss in later sections) which converts both of them into a format suitable for training.

**1.3) Recurrent Neural Network acting as Decoder:** One of the types of RNN known as ```LSTM network``` (Long short-term memory) has been employed as decoder for the task of generating captions. It takes the image features vector and word embeddings of partial captions from ```data_generator``` function at the current timestep as input and decodes them to generate the next most probable word as output.<br>

![0127](https://user-images.githubusercontent.com/71775151/114167969-42f7ce00-994d-11eb-962e-638cbe27bd28.jpg)

<br><br>

## 2) Usage

### 2.1) Prediction using Web-application

### 2.1) Prediction
**2.1.1)** Install Miniconda in your system from link:- https://docs.conda.io/en/latest/miniconda.html. (Tip: Keep agreeing and allowing blindly to what all prompts and buttons come in the process of installing miniconda with an exception case where you have to also tick the option to add miniconda to environment variable, i.e.:

Before

![Inked11aGz_LI (1)](https://user-images.githubusercontent.com/71775151/118517428-d1cde680-b754-11eb-88ec-edb6388063c3.jpg)

After

![install_python_path](https://user-images.githubusercontent.com/71775151/118516836-4f452700-b754-11eb-998e-6d96f56b9aed.png)

Also install Git Bash from link:- https://git-scm.com/downloads.

**2.1.2)** Open Git Bash and type following code in it to clone this GitHub repository at any location of your wish.

**Note:** Copy address of location where you want to clone this repository and paste it in format- "cd path_of_location"
```
cd /c/Users/thispc/Desktop
```
then,
```
git clone https://github.com/malayjoshi13/Describe.git
```
 
**2.1.3)** Now type following code in your command prompt (CMD) to change you current working directory/location to location of "Describe" folder (name of the cloned repository on your local system)

**Note:** Copy address of "Describe" folder and paste it in format- "cd path_of_location"
```
cd C:\Users\thispc\Desktop\Describe
```
then,
```
conda env create -f environment.yml
```
above command creates virtual environment named "Describe" (as present at top of "environment.yml") and also install packages mentioned in the yml file, then execute,
```
conda activate Describe
```
then,
```
python app.py
```
as you execute above code you will some write-up of this kind on your command prompt
![Inkedimage-72-1024x576_LI](https://user-images.githubusercontent.com/71775151/118514812-66831500-b752-11eb-975a-db1aa5860e03.jpg)

just copy the address looking like the red marked one, and paste it on your web browser. On doing this you will be directed to your web application running on local server. There upload image of your choice and get the corresponding description/caption displayed.

**2.1.4)** Captions generated by Describe

![96965447-cb162280-1529-11eb-8d54-4b1932950972](https://user-images.githubusercontent.com/71775151/114160431-8ac62780-9944-11eb-9ea6-ba2ac1b2a7a9.png)

![96965388-b0dc4480-1529-11eb-89d9-5c3dbb037e0d](https://user-images.githubusercontent.com/71775151/114160437-8d288180-9944-11eb-960c-3acd586e70c6.png)

![96965333-97d39380-1529-11eb-8f0e-1ac51c337c4d](https://user-images.githubusercontent.com/71775151/114160441-8e59ae80-9944-11eb-8568-876859131246.png)

![96964503-30691400-1528-11eb-9b5a-821e9984aa33](https://user-images.githubusercontent.com/71775151/114160445-8f8adb80-9944-11eb-823e-68a71ecaa72a.png)
<br>
<br>
<br>

### 2.2) Training 

Note:- line wise code explaination, thus you are requested to keep refering and side-by-side executing steps in ```training.ipynb``` file located at https://drive.google.com/file/d/1ZPuK15FFpQt4kPeWRqZpz7qm2EJcmDwh/view?usp=sharing (just click on link and then open it using Google Colaboratory) for better understanding.
<br>

**2.2.1) Setting up working environment in "Describe" named folder at Google Drive**

Go to the link https://forms.illinois.edu/sec/1713398 and fill the form present there. You will receive Flickr8K data in the gmail id you have provided in the form. Click on both links one by one to download ```Flickr 8k Images``` and ```Flickr 8k Text Data```.

![bandicam 2021-05-07 00-30-55-840](https://user-images.githubusercontent.com/71775151/117351590-881b0b80-aecb-11eb-8e53-d0cdcfc3a7a9.jpg)

Now as whole training process will happen by help of Google Drive and Google Colab thus upload the downloaded Flickr8k zipped data as well as ```training.py``` file (present in this GitHub repository, so just download it from here!) to your Google Drive in a new folder named ```Describe```. 

After uploading both folders, unzip them both (i.e ```Flickr 8k Images``` and ```Flickr 8k Text Data``` folders). After this, take out ```Flickr_8k.devImages.txt```, ```Flickr_8k.testImages.txt```, ```Flickr_8k.trainImages.txt``` and ```Flickr8k.token.txt``` files out of unzipped ```Flickr 8k Text Data``` folder and after taking out these files, delete ```Flickr 8k Text Data``` folder. Then last step will be to rename ```Flickr_8k.devImages.txt``` → ```DevImages.txt```, ```Flickr_8k.testImages.txt``` → ```TestImages.txt```, ```Flickr_8k.trainImages.txt``` → ```TrainImages.txt``` and ```Flickr8k.token.txt``` → ```captions.txt```. At end your ```Describe``` folder must be looking something like this:

![bandicam 2021-05-07 01-08-45-604](https://user-images.githubusercontent.com/71775151/117355934-cc5cda80-aed0-11eb-98b9-e2ae954446af.jpg)
<br>
<br>

**2.2.2) Initiating training.py file**

Now double-click on ```training.py``` file to open it. First of all activate GPU in Colab file by navigating to ```Edit``` → ```Notebook Settings``` → ```Hardware Accelerator```drop-down → ```GPU```.

Once the GPU accelerator is selected, run the ```STEP (1.1)``` cell which will mount/link Google Drive to ```training.py``` Google Colab file. This will help us to directly gain access of Flickr8k data stored in Google Drive. 

After that by executing ```STEP (1.2)``` cell we will import necessary modules and libraries in our ```training.py``` notebook. 

And using ```STEP (1.3)``` cell we will merge content of ```TrainImages.txt``` and ```DevImages.txt``` into a new file ```TrainImagesNames.txt``` so that we can use dev images along with training images in the training process.
<br>
<br>

**2.2.3) Pre-processing images and captions dataset** 

Then by executing ```(STEP 2.1)``` we will modify pre-trained ```InceptionV3``` model and use it as encoder to get encodings (i.e. numerical representation of images) of images stored in ```Flicker8k_Dataset``` folder. Then we save these image encodings as a Pickle file ```all_images_encodings.pkl```. 

Once image encoding work is finished we will execute ```(STEP 2.2)``` which will clean captions (i.e. remove special characters, alphanumeric characters, punctuation marks, etc) and then convert all the cleaned captions into a particular format and saving them in a list called ```modified_captions```. 
<br>
<br>

**2.2.4) Seperating training captions and names of training images** 

After this our next job will be to extract image-encodings and captions corresponding to names of training images. So by executing ```STEP 3.1``` we seperate captions corresponding to training images and save them as a ```value``` in a pair with names of training images acting as a ```key``` of dictionary named ```training_captions```. ```(Step 3.2)``` is a step in continuation to ```(STEP 3.1)``` which will simply find maximum length that any training caption can have. This will help us later in padding all captions to a same length. Next thing that ```(STEP 3.2)``` will do is that it will form a list called ```most_occuring``` which will have those words which have occured atleast 10 times in whole dataset of training captions. Such filteration will help us to remove outliers from our training data i.e. words which occur very less and thus are least important to training process. Words in list ```most_occuring``` are the words which we want for training process. So ```(STEP 3.2)``` do the next job of label-encoding words present in list ```most_occuring``` and saving to dictionary ```word_to_index``` i.e. it will refer these words by means of numeric characters (like 1,2,3,....) starting from index of "1". Label-encoding is important is important beacsue during training process we willl intearct with computer which only understand numeric characters. Also, parallely of creating ```word_to_index``` dictionary, we also create another dictionary called as ```index_to_word``` which will be used to do the reverse job of converting numeric encoding back to their correponding textual words. 

After seperating out training captions, storing them in dictionary ```training_captions```, keeping most occuring words in ```most_occuring``` list and then label encoding these words and saving them in two dictionaries ```index_to_word``` and ```word_to_index```, our next job is to seperate image-encodings of training images out of ```all_images_encodings.pkl``` pickel file. This we will do by executing ```(STEP 3.3)``` which will seperate image encodings of training images and will store them in dictionary called as ```training_images_encodings```.

Therefore, documents used during taining process will be:-

``` dictionaries "training_images_encodings" (key->training imagename & value->corresponding image encoding) and "training_captions" (key->training imagename & value->corresponding caption)```

Now although we have label encoded training captions in ```(STEP 3.2)```, but as label encoding is not so efficient manner thus by executing ```(STEP 3.4)``` we use Glove embedding. But before executing this step we have to download:
```glove.6B.zip``` from line ```Wikipedia 2014 + Gigaword 5 (6B tokens, 400K vocab, uncased, 50d, 100d, 200d, & 300d vectors, 822 MB download): glove.6B.zip``` of following link:
https://nlp.stanford.edu/projects/glove/. 
After downloading, unzip folder, take out ```glove.6B.200d.txt``` file and place it to ```Describe``` folder (your current working directory) in Google Drive.

What Glove embedding do is that correponding to each word it gives a list of coordinates of that word being placed somewhere in a 200-D plane. What is good point of this is that by coordintaes it can be decided how far or how close any two or more than two words are placed, thus telling a good relationship between words which helps neurons to understand relation between words sequencing during training time.

So this step creates a dictionary named ```embedding_matrix``` having key->index/numerical encoding of every word in training caption and value->Glove embedding correponding to a particular index/numerical encoding. This dictionary is later saved as ```embedding_matrix.pkl``` file.
<br>
<br>

**2.2.5) Scripting function called "data_generator"**

The need of this function will be to make training data (i.e. image encodings+captions) in format suitable for training. This function will also create multiple batches of size 36 each so that at a single time during training phase there will be no need to upload whole training data. This function will be executed by ```(STEP 4)```.

So, this function at a particular time will take 36 training captions and corresponding 36 image-encodings in a single batch like:

1st data --> Encoding_of_pic1 & startseq Ram is boy endseq
2nd data --> Encoding_of_pic2 & startseq dog is barking endseq
.
.
18th data --> Encoding_of_pic18 & startseq it is book endseq
.
.
36th data --> Encoding_of_pic36 & startseq snow is falling endseq

After that it will split each of 36 captions in following format:

![bandicam 2021-05-09 23-41-16-619](https://user-images.githubusercontent.com/71775151/117582876-cd387b00-b121-11eb-8ab4-9e1f87115ba2.jpg)

And once this function will collect dataset corresponding to 36 training image-encodings and captions, it will push this whole single batch to flow into "training_model" for training process.
<br>
<br>

**2.2.6) Creating structure of training model and setting arguments of "compile" method** 

![bandicam 2021-05-09 01-55-49-417](https://user-images.githubusercontent.com/71775151/117552662-6c9a3700-b06a-11eb-9add-e0e0ec47fbca.jpg)

On executing ```(STEP 5)```, the structure of our model called as "training_model" will be created. This structure will start from "inputs" layer which will further split into "inputs1" and "inputs2" layers. These two layers will take input data from "data_generator" function (refer STEP 3.2.5) by help of Keras's "Input" layer. 

Input data of type partial/incomplete captions enters "inputs2" via "Input" layer of Keras, gets stored in it and then when it is about to be output from "inputs1" then "Embedding" layer of Keras acts further on output given out by "inputs1" so as to convert these partial captions flowing out of "inputs1" into their embeddings and save them in "layerA" (i.e. in numerical form for computer to understand). Then further "Dropout" layer of Keras acts and in every iteration keeps only 50% of neurons (in alternating manner) in active mode to remember the patterns in words and save these observations in "layerB". After this, "layerB" outputs these observations and LSTM model remembers sequencing of words in training captions. Once done, LSTM model save these rememberings in "layerC".

Now parallely to this process of building structure for textul data, another process of building structure of model for training on image-encodings is also going hand-in-hand. For this first image-encodings through "Inputs" layer of Keras enters "inputs2". After that a "Dropout" layer acts upon and keeps only 50% of neurons active (in alternating manner) in every iteration to learn patterns in image-encodings of training images and save these observations in "layer1". The output from this layer is then worked upon by "dense" layer of Keras which changes dimension of image encodings and save them to "layer2".

After this the outputs from "layerC" and "layer2" (which are memorization of LSTM and modified dimensioned image-encodings respectively) merge together via "add" layer of "Keras.merge" and save them into "merging_point". Then "relu" activation is applied over output given out by "merging_point" so as to make the output graph non-linear to be fit well around the training data. This non-linear graph is then saved in "activator". Then at last with help of "softmax" layer our model predicts one word out of all words posssible to be the next word of the input partial/incomplete caption feeded to the model for training purpose and save it in "outputs". The number of possibilities of predicting such word = total words in "most_occuring_words" list = "vocab_size" = 1798.

Once the structure for our model to be trained called as "training_model" is been constructed, the next job set the arguments of "compile" method which will configure the model for training time and based upon their actions the weights of training model will be updated during backpropagation process. Now let us understand what happens in backpropagation process. So during this process the output predicted by model (which is actually the word next to the incomplete/partial input training caption) is compared with the word which is goaled/aimed to be present next to the incomplete training caption. And the mission during whole backpropagation is to just minimise this gap between what was aimed and what is actually predicted by model.
<br>
<br>

**2.2.7) Starting the training process** 

NOTE: Before starting this process check that in "Describe" folder located at Google Drive you have following files:- Flickr8k_Dataset, captions.txt, DevImages.txt, TrainImages.txt, TestImages.txt, training.ipynb, glove.6B.200d.txt

Now we will initiate training process so that using the structure we designed in ```(STEP 5)``` of model called "training_model", our input data will flow, get observed and memorized by neurons and also updating process of memorized patterns during backpropagation process will happen. We will set some parameters to direct in what way the training process will take place. Therefore, we have asked model to see whole dataset from end-to-end whole 30 times (by setting ```epoch = 30```). In each of these 30 epochs, our model will see whole dataset in piece wise manner, i.e. model has to train itself on 200 batches (this is known as "steps_per_epoch") having 35 elements each.

After all of this process we will finally get our "weights" or the memorization of how each word is related to other word and also to a particular image. We will then save these weights using "training_model.save_weights". 
<br>
<br>

**2.2.8) Evaluating the trained model**

Using ```(STEP 7.1)``` and ```(STEP 7.2)``` we will extract captions and image-encodings for validation images respectively. 

Using ```(STEP 7.3)``` we will load the saved weights of trained model and then using it in ```(STEP 7.4)``` we predict whole validation captions iteration-by-iteration corresponding to feeded image-encodings and one starting word "startseq". We save these in list "predicted".

On other hand we save each word of each validation captions in list "actual".

Then in ```(STEP 7.5)``` we compare how many words predicted by model actually matches with actual words. On basis of this comparison we calculate BLEU score. BLEU score lies between 0 to 1 and score more towards 1 show higher accuracy of model's pediction.
