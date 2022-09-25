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

### 1.1) For training and evaluation of trained model:-

- Create a directory in your Google Drive of name `Describe`.
- Request the Flickr8k dataset from this link https://illinois.edu/fb/sec/1713398. Download and place it inside `Describe/dataset` folder in your Google Drive. 
- Now rename few files in `/dataset` folder to following filenames:-<br>
  - Flickr8k.token.txt to `captions.txt`. <br> 
  - Flickr_8k.trainImages.txt to `TrainImagesName.txt`. <br>
  - Flickr_8k.devImages.txt to `DevImagesName.txt`. <br>
  - Flickr_8k.testImages.txt to `TestImagesName.txt`. <br>
  - Flickr8k_Dataset contains to `All_images`.
- Copy following Colab script in the `Describe` directory in your Google Drive. Now according to where you created `Describe` directory in your Google Drive, give that location within the following script. Colab script for training: https://drive.google.com/file/d/1ZPuK15FFpQt4kPeWRqZpz7qm2EJcmDwh/view?usp=sharing .
- Run each code cell of the above Colab script to train your model and evaluate it based on BLEU score.

### 1.2) For inference:- 
https://colab.research.google.com/drive/1HIpLysJeD401qB8bayn7sKXehEQUzl8L?usp=sharing

## 2) Working of Describe

### 2.1) While training and evaluation:-

#### Step 1) Encoding all the images using InceptionV3
All training and testing images of Flickr8k dataset of size (299, 299, 3) is firstly encoded into feature vector of length 4096 using InceptionV3 model ([read more](https://github.com/malayjoshi13/Describe/blob/main/learnings.md)) whose last output layer is removed (as last layer of every CNN layer being softmax is used for classification task, which we are not doing here, so no need of last layer).

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

Second pipeline, aka `caption pipeline` accepts first partial caption (corresponding to first training image) present in `X2` container (coming from `data_generator` function) via `input2` entrance. This partial caption is then passed through embedding layer (which will encode this partial caption using weights of Glove), dropout layer and then LSTM layer (having 256 cells which understands sequencing of words in first partial caption, [read more](https://github.com/malayjoshi13/Describe/blob/main/learnings.md)). At last, the partial caption comes to `layerC` exit point of `caption pipeline`.

Now outputs from the two input pipelines get merge at `merging_point` layer. We do this so that we get training input in form "image encoding + corresponding partial caption". This merged input is then passed to dropout layer and softmax layer which will output probability distribution that across 1798 words (present in vocabulary made using most occuring words), which word could be possible next word in continuation to "X2" (partial caption feeded to model as input, during training time).

Now during backpropagation, the output predicted by model (which is actually the word next to the incomplete/partial training caption feeded to the model as input during training phase) is compared with the actual word (which should be actually present next to the incomplete training caption feeded as input to model during training phase). And the mission during whole backpropagation is to just minimise this gap between what word was aimed and what word is predicted by model. 

![image](https://user-images.githubusercontent.com/71775151/192115701-accc9822-6aae-4a30-af51-d8c23b28c473.png)

![image](https://user-images.githubusercontent.com/71775151/192117491-599ec8f0-9102-4837-ab45-35551ea39d90.png)


#### Step 8) Evaluating trained model

To evaluate the trained image-captioning model, first evaluation image is input to the model alongwith starting word "startseq". Model predicts word next to the starting word "startseq". Then again first word and second word alongwith the same first evaluation image is input to the trained model. Trained model now predicts third word. In same way, trained model predicts whole caption corresponding to first evaluation image. And gradually predicts all captions for all given evalauation images.

Then these predicted captions are matched with actual captions of the evaluation images. How closely the two sentences matches to each other is highlighted by BLEU score ([read more](https://github.com/malayjoshi13/Describe/blob/main/learnings.md)).

### 2.2) While inferencing:-

The process happening for multiple evaluation imsges happens in same way for single inference image.

## 3) Things learnt
a) different layers and activation fns and adam is best - https://arxiv.org/pdf/1708.02043.pdf and cross-entropy loss satisfies the above criteria and is therefore a decent loss function for classification problems. Although our problem's a sequence output problem, at every time-step, it is spitting out a probability distribution of words from which we are picking the one having max probability. So, fundamentally at a step level, it is still a classification problem. and Number of Epochs: There's no way to know for sure how many epochs will be needed for training a deep learning model. We only know that as long as both train loss and validation loss are on a decline, we need to continue training the model as it has scope to learn more and improve. In the above case, we can observe that after around the 10th epoch, our model went haywire and started overfitting. So, this is a good cue to stop training.

Although we trained our model further and tried to see if it could improve which didn't happen, we saved the model weights at each and every step of the way. We can therefore say that the model which is trained for 10 epochs is the best and must be deployed to predict on test images. and types of CNN - https://pyimagesearch.com/2017/03/20/imagenet-vggnet-resnet-inception-xception-keras/, b) RNN, LSTM-https://colah.github.io/posts/2015-08-Understanding-LSTMs/, c) embedding model like Glove and word2vec, d) loss and optimizers, e) sequential model vs functional model, f) attention with captioning - https://medium.com/swlh/image-captioning-using-attention-mechanism-f3d7fc96eb0e, g) BLEU score, h) marge model (we used as better https://arxiv.org/pdf/1708.02043.pdf, https://arxiv.org/abs/1703.09137, https://machinelearningmastery.com/caption-generation-inject-merge-architectures-encoder-decoder-model/) vs inject model

## 4) Read
a) https://towardsdatascience.com/a-guide-to-image-captioning-e9fd5517f350, b) https://arxiv.org/abs/1612.01887, c) https://machinelearningmastery.com/caption-generation-inject-merge-architectures-encoder-decoder-model/, d) https://arxiv.org/abs/1411.4555, e) https://arxiv.org/abs/1502.03044, f) https://arxiv.org/abs/1703.09137, g) https://arxiv.org/abs/1601.03896, h) https://cs.stanford.edu/people/karpathy/cvpr2015.pdf
