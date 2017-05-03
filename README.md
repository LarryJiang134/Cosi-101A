# Cosi-101A

Projects did on cosi101A class

Before running the code to perform test function, make sure there are following files at source directory:
	model files
	annotation.txt

The final accurary on test example I receive is 0.8308.
	
Train.py :

The CNN frame is from tensorflow tutorial. I did little modification on that since somehow most of my modifications lower the overall test accuracy.

The only thing I did is to pre process the test image by enhance the exposure and padding rate. These raise the accuracy from 0.6 to 0.8308.

I added the save and restore feature to the file so that every time we train, we have a "model" file generated at the same folder of the source file.

If the python code is called without a path specified, it will train the model by first download the MNIST data (or simply use it if it already exists).

if the python code is called followed by a path which is supposed to have a few image files in it, then it will start the test algorithm by calling the model it generated and try to predict what digit every image should be (yes, to do it, a model file is necessary).

At the end of the test algorithm, it will generates a prediction.txt file which contains all the image name followed with its predicted digits.

After that, there is a section of the code that compare prediction.txt with annotation.txt which should be the correct answer to the test and then print out the overall test accuracy.
