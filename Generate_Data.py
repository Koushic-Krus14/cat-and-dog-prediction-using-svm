import os
import glob
import numpy
import cv2

imagePaths = []
# input images
for img in glob.glob("Data/*.jpg"):  
    imagePaths = list(glob.glob("Data/*.jpg"))


def image_vector(image, size=(128, 128)):
    return cv2.resize(image, size).flatten()


imagematrix = []
imagelabels = []
pixels = None

for (i, path) in enumerate(imagePaths):
    
    image = cv2.imread(path)
    label = path.split(os.path.sep)[-1].split(".")[0]
    pixels = image_vector(image)

    
    imagematrix.append(pixels)
    imagelabels.append(label)

imagematrix = numpy.array(imagematrix)
imagelabels = numpy.array(imagelabels)

# save numpy arrays for future use
numpy.save("matrix.npy", imagematrix)
numpy.save("labels.npy", imagelabels)