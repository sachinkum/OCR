import numpy as np
from skimage.io import imread
from skimage.filters import threshold_otsu
from skimage.transform import resize
import pickle
from matplotlib import pyplot as plt
from skimage.morphology import closing, square
from skimage.measure import regionprops
from skimage import restoration
from skimage import measure
from skimage.color import label2rgb
import matplotlib.patches as mpatches


class UserData():
    """
    class in charge of dealing with User Image input.
    the methods provided are finalized to process the image and return
    the text contained in it.
    """

    def __init__(self, image_file):
        """
        reads the image provided by the user as grey scale and preprocesses it.
        """
        self.image = imread(image_file, as_grey=True)
        self.preprocess_image()

    def preprocess_image(self):
        """
        Denoises and increases contrast.
        """
        image = restoration.denoise_tv_chambolle(self.image, weight=0.1)
        thresh = threshold_otsu(image)
        self.bw = closing(image > thresh, square(2))
        self.cleared = self.bw.copy()
        return self.cleared

    def get_text_candidates(self):
        """
        identifies objects in the image. Gets contours, draws rectangles around them
        and saves the rectangles as individual images.
        """
        label_image = measure.label(self.cleared)
        borders = np.logical_xor(self.bw, self.cleared)
        label_image[borders] = -1

        coordinates = []
        i = 0

        for region in regionprops(label_image):
            if region.area > 10:
                minr, minc, maxr, maxc = region.bbox
                margin = 3
                minr, minc, maxr, maxc = minr - margin, minc - margin, maxr + margin, maxc + margin
                roi = self.image[minr:maxr, minc:maxc]
                if roi.shape[0] * roi.shape[1] == 0:
                    continue
                else:
                    if i == 0:
                        samples = resize(roi, (20, 20))
                        coordinates.append(region.bbox)
                        i += 1
                    elif i == 1:
                        roismall = resize(roi, (20, 20))
                        samples = np.concatenate((samples[None, :, :], roismall[None, :, :]), axis=0)
                        coordinates.append(region.bbox)
                        i += 1
                    else:
                        roismall = resize(roi, (20, 20))
                        samples = np.concatenate((samples[:, :, :], roismall[None, :, :]), axis=0)
                        coordinates.append(region.bbox)

        self.candidates = {
            'fullscale': samples,
            'flattened': samples.reshape((samples.shape[0], -1)),
            'coordinates': np.array(coordinates)
        }

        print('Images After Contour Detection')
        print('Fullscale: ', self.candidates['fullscale'].shape)
        print('Flattened: ', self.candidates['flattened'].shape)
        print('Contour Coordinates: ', self.candidates['coordinates'].shape)
        print('============================================================')

        return self.candidates

    def select_text_among_candidates(self, model_filename2):
        """
        it takes as argument a pickle model and predicts whether the detected objects
        contain text or not.
        """
        with open(model_filename2, 'rb') as fin:
            model = pickle.load(fin)

        is_text = model.predict(self.candidates['flattened'])

        self.to_be_classified = {
            'fullscale': self.candidates['fullscale'][is_text == '1'],
            'flattened': self.candidates['flattened'][is_text == '1'],
            'coordinates': self.candidates['coordinates'][is_text == '1']
        }

        print('Images After Text Detection')
        print('Fullscale: ', self.to_be_classified['fullscale'].shape)
        print('Flattened: ', self.to_be_classified['flattened'].shape)
        print('Contour Coordinates: ', self.to_be_classified['coordinates'].shape)
        print('Rectangles Identified as NOT containing Text ' + str(
            self.candidates['coordinates'].shape[0] - self.to_be_classified['coordinates'].shape[0]) + ' out of ' + str(
            self.candidates['coordinates'].shape[0]))
        print('============================================================')

        return self.to_be_classified

    def classify_text(self, model_filename36):
        """
        it takes as argument a pickle model and predicts character
        """
        with open(model_filename36, 'rb') as fin:
            model = pickle.load(fin)

        which_text = model.predict(self.to_be_classified['flattened'])

        self.which_text = {
            'fullscale': self.to_be_classified['fullscale'],
            'flattened': self.to_be_classified['flattened'],
            'coordinates': self.to_be_classified['coordinates'],
            'predicted_char': which_text
        }

        return self.which_text

    def realign_text(self):
        """
        processes the classified characters and prints them
        """
        max_maxrow = max(self.which_text['coordinates'][:, 2])
        min_mincol = min(self.which_text['coordinates'][:, 1])
        subtract_max = np.array([max_maxrow, min_mincol, max_maxrow, min_mincol])
        flip_coord = np.array([-1, 1, -1, 1])

        coordinates = (self.which_text['coordinates'] - subtract_max) * flip_coord

        ymax = max(coordinates[:, 0])
        xmax = max(coordinates[:, 3])

        coordinates = [list(coordinate) for coordinate in coordinates]
        predicted = [list(letter) for letter in self.which_text['predicted_char']]
        print(predicted)

