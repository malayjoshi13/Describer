# Describe
Describe is a machine learning based **image captioning system** which by help of ***InceptionV3 model*** (a type of convolutional neural network, [read more](https://github.com/malayjoshi13/Describe/blob/main/learnings.md)) and ***long short term memory model (LSTM)*** (a type of recurrent neural network, [read more](https://github.com/malayjoshi13/Describe/blob/main/learnings.md)) fine tuned on ***Flickr8k data*** ([read more](https://github.com/malayjoshi13/Describe/blob/main/learnings.md/#flickr8k)), generates textual captions describing about images feed to it. 

Got following BLEU scores ([read more](https://github.com/malayjoshi13/Describe/blob/main/learnings.md/#flickr8k)) during model evaluation:<br>
BLEU-1: 0.475314,<br>
BLEU-2: 0.288632,<br>
BLEU-3: 0.202204,<br>
BLEU-4: 0.095817

https://user-images.githubusercontent.com/71775151/120343096-0b3a5080-c316-11eb-832c-b21f190bfe6f.mp4

![image](https://user-images.githubusercontent.com/71775151/192083201-035fc4c6-f1eb-42b0-ab68-1bc7942ad90a.png)

## 1) How to use Describe?

### 1.1) For training:-

- After downloading and placing dataset files (by following descriptions in [link](https://github.com/malayjoshi13/Describe/blob/main/learnings.md/#flickr8k)), 
https://drive.google.com/file/d/1ZPuK15FFpQt4kPeWRqZpz7qm2EJcmDwh/view?usp=sharing

### 1.2) For prediction:- 
https://colab.research.google.com/drive/1HIpLysJeD401qB8bayn7sKXehEQUzl8L?usp=sharing

## 2) How is Describe working?

### 2.1) While training:-

#### Step 1) Encoding all the images using InceptionV3
All training and testing images of Flickr8k dataset of size (299, 299, 3) is firstly encoded into feature vector of length 4096 using InceptionV3 model whose last output layer is removed (as last layer of every CNN layer being softmax is used for classification task, which we are not doing here, so no need of last layer).

#### Step 2) Cleaning all the captions
Then all training and texting captions are cleaned and pre-processed by tokenizing, converting to lower case, removing punctuation and single letter words (“s” and “a”). Then to each caption, we add the start and end signs, “startseq” and “endseq” respectively.
Describe uses both Natural Language Processing and Computer Vision to generate the captions that gives description of an input image. 

#### Step 3) Seperating training captions and image-encodings from overall data
#### Step 4) For training captions, creating vocabulary of most occuring words and creating the word <--> index mappers
#### Step 5) Using GloVe to generate embeddings for each word in the vocabulary. This vocabulary has most occuring words present in training captions
#### Step 6) Scripting `data_generator` function
The need of this function will be to convert total training data (i.e. image encodings+captions) into multiple batches comprising of 36 training captions and corresponding 36 image-encodings. This is done so that at a single time during training phase, there will be no need to upload whole training data.

For example, `data_generator` function takes input of a single batch comprising of following 36 training captions and corresponding 36 image-encodings:

1st data --> Encoding_of_pic1 & startseq Ram is boy endseq <br>
2nd data --> Encoding_of_pic2 & startseq dog is barking endseq <br>
. <br>
. <br>
18th data --> Encoding_of_pic18 & startseq it is book endseq <br>
. <br>
. <br>
36th data --> Encoding_of_pic36 & startseq snow is falling endseq 

And outputs data converted into following format:

![bandicam 2021-05-09 23-41-16-619](https://user-images.githubusercontent.com/71775151/117582876-cd387b00-b121-11eb-8ab4-9e1f87115ba2.jpg)

Then `data_generator` function will push this whole single batch to flow into `training_model` to train it.

### Step 7) Training the image-captioning model

As the training process starts, first pair of image-encoding and its corresponding 5 captions belonging to first batch flows into the image-captioning model. 

Image-captioning model comprises of two input pipelines. As the training process starts, first input pipeline, aka `image-encoding pipeline` accepts first image's encoding present in `X1` container (coming from `data_generator` function) via `input1` entrance. This encoding is then passed through dropout and dense layer (which compressing encodings into dimension of 256 as we will be using LSTM of 256 cells). At last, this encoding will wait for first caption (corresponding to first image) at `layer2` exit point of `image-encoding pipeline`. 

Second pipeline, aka `caption pipeline` accepts first partial caption (corresponding to first training image) present in `X2` container (coming from `data_generator` function) via `input2` entrance. This partial caption is then passed through embedding layer (which will encode this partial caption using weights of Glove), dropout layer and then LSTM layer (having 256 cells which understands sequencing of words in first partial caption). At last, the partial caption comes to `layerC` exit point of `caption pipeline`.


![image](https://user-images.githubusercontent.com/71775151/192044962-ebe4a6f3-f8b7-4003-9b33-0bb3594191f8.png)

It comprises of three main components:

**1.1) Convolutional Neural Network (CNN) acting as Encoder:** A pre-trained CNN known as ```InceptionV3``` is used as an encoder to obtain feature vector of every image from its (i.e. InceptionV3) second last layer (beacuse last layer of every CNN is a softmax layer which gives prediction probability).<br>

![Schematic-diagram-of-InceptionV3-model-compressed-view](https://user-images.githubusercontent.com/71775151/114168447-e47f1f80-994d-11eb-830b-aa212eadebc2.jpg) 

**1.3) Recurrent Neural Network acting as Decoder:** A type of RNN known as ```LSTM network``` (Long short-term memory) generates caption of every given image. It do so by taking input of feature vector of the given image and word embedding vector of the starting part of that image's caption and then keep on generating the next most probable words for this input partial caption (by considering the feature vector of image it is related to and the word embeddings of previous words).<br>

![0127](https://user-images.githubusercontent.com/71775151/114167969-42f7ce00-994d-11eb-962e-638cbe27bd28.jpg)


**Understanding some code parts of training file**

**b) Creating structure of training model and setting arguments of "compile" method** 

![bandicam 2021-05-09 01-55-49-417](https://user-images.githubusercontent.com/71775151/117552662-6c9a3700-b06a-11eb-9add-e0e0ec47fbca.jpg)

i) On executing ```(STEP 5)```, the structure of our model called as "training_model" will be created. This structure will start from "inputs" layer which will further split into "inputs1" and "inputs2" layers. These two layers will take input data from "data_generator" function (refer STEP 3.2.5) by help of Keras's "Input" layer. 

ii) Input data of type partial/incomplete captions enters "inputs2" via "Input" layer of Keras, gets stored in it and then when it is about to be output from "inputs1" then "Embedding" layer of Keras acts further on output given out by "inputs1" so as to convert these partial captions flowing out of "inputs1" into their embeddings and save them in "layerA" (i.e. in numerical form for computer to understand). Then further "Dropout" layer of Keras acts and in every iteration keeps only 50% of neurons (in alternating manner) in active mode to remember the patterns in words and save these observations in "layerB". After this, "layerB" outputs these observations and LSTM model remembers sequencing of words in training captions. Once done, LSTM model save these rememberings in "layerC".

iii) Now in parallel to this process of building structure for textul data, another process of building structure of model for training on image-encodings is also going hand-in-hand. For this first image-encodings through "Inputs" layer of Keras enters "inputs2". After that a "Dropout" layer acts upon and keeps only 50% of neurons active (in alternating manner) in every iteration to learn patterns in image-encodings of training images and save these observations in "layer1". The output from this layer is then worked upon by "dense" layer of Keras which changes dimension of image encodings and save them to "layer2".

iv) After this the outputs from "layerC" and "layer2" (which are memorization of LSTM and modified dimensioned image-encodings respectively) merge together via "add" layer of "Keras.merge" and save them into "merging_point". Then "relu" activation is applied over output given out by "merging_point" so as to make the output graph non-linear to be fit well around the training data. This non-linear graph is then saved in "activator". Then at last with help of "softmax" layer our model predicts one word out of all words posssible to be the next word of the input partial/incomplete caption feeded to the model for training purpose and save it in "outputs". The number of possibilities of predicting such word = total words in "most_occuring_words" list = "vocab_size" = 1798.

v) Once the structure for our model to be trained called as "training_model" is been constructed, the next job set the arguments of "compile" method which will configure the model for training time and based upon their actions the weights of training model will be updated during backpropagation process. Now let us understand what happens in backpropagation process. So during this process the output predicted by model (which is actually the word next to the incomplete/partial input training caption) is compared with the word which is goaled/aimed to be present next to the incomplete training caption. And the mission during whole backpropagation is to just minimise this gap between what was aimed and what is actually predicted by model.

## 3) Things learnt
a) different layers and activation fns and types of CNN - https://pyimagesearch.com/2017/03/20/imagenet-vggnet-resnet-inception-xception-keras/, b) RNN, LSTM, c) embedding model like Glove and word2vec, d) loss and optimizers, e) sequential model vs functional model, f) attention with captioning - https://medium.com/swlh/image-captioning-using-attention-mechanism-f3d7fc96eb0e, g) BLEU score, h) marge model (we used) vs inject model

## 4) Read
a) https://towardsdatascience.com/a-guide-to-image-captioning-e9fd5517f350, b) https://arxiv.org/abs/1612.01887, c) https://machinelearningmastery.com/caption-generation-inject-merge-architectures-encoder-decoder-model/, d) https://arxiv.org/abs/1411.4555, e) https://arxiv.org/abs/1502.03044, f) https://arxiv.org/abs/1703.09137, g) https://arxiv.org/abs/1708.02043, h) https://arxiv.org/abs/1601.03896
