# Describer

Describer is an **image captioning system** comprising of ***InceptionV3 model*** (CNN) and ***long short term memory model aka LSTM*** (RNN) fine-tuned on ***Flickr8k data***, which generates textual captions describing the images feed to it.

Got the following BLEU scores during model evaluation:<br>
BLEU-1: 0.475314,<br>
BLEU-2: 0.288632,<br>
BLEU-3: 0.202204,<br>
BLEU-4: 0.095817

![image](https://user-images.githubusercontent.com/71775151/192083201-035fc4c6-f1eb-42b0-ab68-1bc7942ad90a.png)

## 1) Bring your own image and generate the most suited caption describing it (aka Inference)
Check out this easy-to-use [Google Colab Notebook](https://colab.research.google.com/drive/1HIpLysJeD401qB8bayn7sKXehEQUzl8L?usp=sharing). Upload your image and get your captions!

## 2) Training and Evaluating on your own dataset and hyperparameters to improve it further

- Create a directory in your Google Drive of name `Describe`.
- Request the Flickr8k dataset from this link https://illinois.edu/fb/sec/1713398. Download and place it inside `Describe/dataset` folder in your Google Drive. 
- Now rename few files in `/dataset` folder to following filenames:-<br>
  - Flickr8k.token.txt to `captions.txt`. <br> 
  - Flickr_8k.trainImages.txt to `TrainImagesName.txt`. <br>
  - Flickr_8k.devImages.txt to `DevImagesName.txt`. <br>
  - Flickr_8k.testImages.txt to `TestImagesName.txt`. <br>
  - Flickr8k_Dataset contains to `All_images`.
- Copy following Colab script in the `Describe` directory in your Google Drive. Now according to where you created `Describe` directory in your Google Drive, give that location within the following script. 

Colab script for training: https://drive.google.com/file/d/1yYFVjrQAGs9gDLpZAf4dx4QIU93X9Inw/view?usp=sharing .

- Run each code cell of the above Colab script to train your model and evaluate it based on BLEU score.

## 3) Working of Describer

### 2.1) While training and evaluation:-

#### Step 1) Encoding all the images using InceptionV3
All training and testing images of Flickr8k dataset of size (299, 299, 3) is firstly encoded into feature vector of length 4096 using InceptionV3 model whose last output layer is removed (as last layer of every CNN layer being softmax is used for classification task, which we are not doing here, so no need of last layer).

For this task of extracting the features out of images, I tried out VGG-16, Resnet-50 and InceptionV3. Human top-5 error on Imagenet is 5.1%. VGG16 has almost 134 million parameters thhus it took almost 1hr for extracting the features but as its top-5 error on Imagenet is 7.3%, thus it better represents ann image. On other hand, InceptionV3 and Resnet-50 has 21 million parameters, thus in just 20mins extracted features but as InceptionV3's top-5 error on Imagenet is 3.46% and that of Resnet-50 is much lesser, thus both models represents images in less depth.

#### Step 2) Cleaning all the captions
Then all training and texting captions are cleaned and pre-processed by tokenizing, converting to lower case, removing punctuation and single letter words (“s” and “a”). Then to each caption, we add the start and end signs, “startseq” and “endseq” respectively.
Describe uses both Natural Language Processing and Computer Vision to generate the captions that gives description of an input image. 

#### Step 3) Seperating training captions and image-encodings from overall data
#### Step 4) For training captions, creating vocabulary of most occuring words and creating the word <--> index mappers
#### Step 5) Using GloVe to generate embeddings for each word in the vocabulary. This vocabulary has most occuring words present in training captions
#### Step 6) Scripting `data_generator` function
This function is needed to convert total training data (i.e. image encodings+captions) into multiple batches comprising of 35 training image-encodings and there corresponding captions, so that during training phase, there will be no need to upload whole training data all at once.

After splitting whole dataset into multiple batches, each batch look likes:

1st data --> Encoding_of_pic1 & startseq Ram is boy endseq <br>
2nd data --> Encoding_of_pic2 & startseq dog is barking endseq <br>
. <br>
. <br>
18th data --> Encoding_of_pic18 & startseq it is book endseq <br>
. <br>
. <br>
36th data --> Encoding_of_pic36 & startseq snow is falling endseq 

The data of each batch is then converted into following format:

![bandicam 2021-05-09 23-41-16-619](https://user-images.githubusercontent.com/71775151/117582876-cd387b00-b121-11eb-8ab4-9e1f87115ba2.jpg)

Then these single-single batches will be pushed by `data_generator` function to flow into image-captioning model to train it.

#### Step 7) Training the image-captioning model

As the training process starts, first pair of image-encoding and its corresponding 5 captions belonging to first batch flows into the image-captioning model. 

Image-captioning model comprises of two input pipelines. As the training process starts, first input pipeline, aka `image-encoding pipeline` accepts first image's encoding present in `X1` container (coming from `data_generator` function) via `input1` entrance. This encoding is then passed through dropout (50% dropout to reduce overfitting the training dataset, as this model configuration learns very fast) and dense layer (which compressing encodings into dimension of 256 as we will be using LSTM of 256 cells). At last, this encoding will wait for first caption (corresponding to first image) at `layer2` exit point of `image-encoding pipeline`. 

Second pipeline, aka `caption pipeline` accepts first partial caption (corresponding to first training image) present in `X2` container (coming from `data_generator` function) via `input2` entrance. This partial caption is then passed through embedding layer (which will encode this partial caption using weights of Glove), dropout layer and then LSTM layer (having 256 cells which understands sequencing of words in first partial caption). At last, the partial caption comes to `layerC` exit point of `caption pipeline`.

Now outputs from the two input pipelines get merge at `merging_point` layer. We do this so that we get training input in form "image encoding + corresponding partial caption". This merged input is then passed to dropout layer and softmax layer which will output probability distribution that across 1798 words (present in vocabulary made using most occuring words), which word could be possible next word in continuation to "X2" (partial caption feeded to model as input, during training time).

Now during backpropagation, the output predicted by model (which is actually the word next to the incomplete/partial training caption feeded to the model as input during training phase) is compared with the actual word (which should be actually present next to the incomplete training caption feeded as input to model during training phase). And the mission during whole backpropagation is to just minimise this gap between what word was aimed and what word is predicted by model. 

![image](https://user-images.githubusercontent.com/71775151/192115701-accc9822-6aae-4a30-af51-d8c23b28c473.png)

![image](https://user-images.githubusercontent.com/71775151/192117491-599ec8f0-9102-4837-ab45-35551ea39d90.png)


#### Step 8) Evaluating trained model

To evaluate the trained image-captioning model, first evaluation image is input to the model alongwith starting word "startseq". Model predicts word next to the starting word "startseq". Then again first word and second word alongwith the same first evaluation image is input to the trained model. Trained model now predicts third word. In same way, trained model predicts whole caption corresponding to first evaluation image. And gradually predicts all captions for all given evalauation images.

Then these predicted captions are matched with actual captions of the evaluation images. How closely the two sentences matches to each other is highlighted by BLEU score.

### 2.2) While inferencing:-

The process happening for multiple evaluation images happens in same way for single inference image. 
