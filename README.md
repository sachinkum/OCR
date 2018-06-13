# OCR
Optical character recognition using Python 3(Pandas/Scikit-Learn/Numpy/Scikit-image)

Just pass the number of images to the prompt. The images should be stored in numerical order in the base folder.

The training is done on 75000 images with different font styles for every character, divided into 2 groups for training and testing.
The code needs a few changes after the pickle model generation to accomodate the generated model. Else it would train every time.

Pass the path of the pickle file in the 'config.py' and 'text-config.py' files.

The trained dataset is not included due to github space contraints and development hardware constraints (training takes about 1 minute for each image)

Uses HOG and SVM for training and testing.

Should yield an accuracy of ~97% on ANY font style (theoretically).
Can be modified to gather text from images with non-white backgrounds (motivation pics,etc), similar to Google Lens (not on the same scale).
