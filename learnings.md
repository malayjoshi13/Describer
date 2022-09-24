This file contains all the topics that I learned while working on this project.

Link for prediction file:- https://colab.research.google.com/drive/1HIpLysJeD401qB8bayn7sKXehEQUzl8L?usp=sharing

Link for training file:- https://drive.google.com/file/d/1ZPuK15FFpQt4kPeWRqZpz7qm2EJcmDwh/view?usp=sharing

Understanding some code parts of training file

a) Scripting function called "data_generator"

The need of this function will be to make training data (i.e. image encodings+captions) in format suitable for training. This function will also create multiple batches of size 36 each so that at a single time during training phase there will be no need to upload whole training data.

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
Link for prediction file:- https://colab.research.google.com/drive/1HIpLysJeD401qB8bayn7sKXehEQUzl8L?usp=sharing

Link for training file:- https://drive.google.com/file/d/1ZPuK15FFpQt4kPeWRqZpz7qm2EJcmDwh/view?usp=sharing

Understanding some code parts of training file

a) Scripting function called "data_generator"

The need of this function will be to make training data (i.e. image encodings+captions) in format suitable for training. This function will also create multiple batches of size 36 each so that at a single time during training phase there will be no need to upload whole training data.

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
Link for prediction file:- https://colab.research.google.com/drive/1HIpLysJeD401qB8bayn7sKXehEQUzl8L?usp=sharing

Link for training file:- https://drive.google.com/file/d/1ZPuK15FFpQt4kPeWRqZpz7qm2EJcmDwh/view?usp=sharing

Understanding some code parts of training file

a) Scripting function called "data_generator"

The need of this function will be to make training data (i.e. image encodings+captions) in format suitable for training. This function will also create multiple batches of size 36 each so that at a single time during training phase there will be no need to upload whole training data.

So, this function at a particular time will take 36 training captions and corresponding 36 image-encodings in a single batch like:

1st data --> Encoding_of_pic1 & startseq Ram is boy endseq

2nd data --> Encoding_of_pic2 & startseq dog is barking endseq

.

.

18th data --> Encoding_of_pic18 & startseq it is book endseq

.

.

36th data --> Encoding_of_pic36 & startseq snow is falling endseq

After that it will split each of 36 captions in following format:Link for prediction file:- https://colab.research.google.com/drive/1HIpLysJeD401qB8bayn7sKXehEQUzl8L?usp=sharing

Link for training file:- https://drive.google.com/file/d/1ZPuK15FFpQt4kPeWRqZpz7qm2EJcmDwh/view?usp=sharing

Understanding some code parts of training file

a) Scripting function called "data_generator"

The need of this function will be to make training data (i.e. image encodings+captions) in format suitable for training. This function will also create multiple batches of size 36 each so that at a single time during training phase there will be no need to upload whole training data.

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

# Flickr8k
