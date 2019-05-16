# Source files for the chapter 3 of the Bsc Thesis

_This branch contains the files which were used to create the models which are discussed in the third chapter of the bachelor thesis._

### Following file are of particular importance:


The `models/` directory contains pickle dumps of all the models generated.

The file `cnn_train.py` contains the code which was used to train and evaluate the convolutional neural network approach.

The file `svm.py` contains the code which was used to train and evaluate the SVM based approach.

The file `svm_cnn.py` modifies the CNN model created by `cnn_train.py` to use the model as a feature extractor. These features are then used with the SVM classifier defined in `svm.py`
