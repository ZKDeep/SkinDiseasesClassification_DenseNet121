# SkinDiseasesClassification_DenseNet121
This is the classification model used for 7 type of skin diseases of human.
The code is simple to use......

# configurations
Open file "input_config.py" file and check all your requirements.
Give proper path to training, validation and test sets.
Otherwise you will see bunch of errors.

# Train the Model
To train model on skin diseases you have train, valid and test directory of images.
There is configuration file name "input_config.py" made changes in it according to your use.
Run "train.py" file using any tool spyder or cmd
It will start training and saving best weights in "wieghts.h5" based on minium loss of validation dataset.
Try to Run the training model for 40 to 50 epochs.
You can change epochs in "input_config.py"

# Test the Model
In order to to check the accuracy of model on Test set just
Run "evalute.py" and you will get test loss and accuracy accordingly on given saved weights.


It's all About it.
Dataset is avialble in the link given below

https://drive.google.com/drive/folders/1wX5mjP7gAj3wW8OrTQAwXo328_OFHvwm?usp=sharing
