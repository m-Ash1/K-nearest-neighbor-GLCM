import os
import cv2
import skimage
from sklearn.metrics import accuracy_score
import random
from sklearn.neighbors import KNeighborsClassifier
from skimage.feature import greycomatrix, greycoprops

DATADIR = "C:/" # put your directory here

CATEGORIES = ["fabric", "foliage", "glass", "leather", "metal", "paper", "plastic", "stone", "water", "wood"]

training_data = []


def create_training_data():
    for category in CATEGORIES:
        path = os.path.join(DATADIR, category)
        class_num = CATEGORIES.index(category)
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path, img), 0)
                IMG_SIZE = 100
                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                training_data.append([new_array, class_num])

            except Exception as e:
                print(str(e))


create_training_data()
random.shuffle(training_data)


x_training = []  # training features 700
y_training = []  # training labels

x_test = []  # testing features 300
y_test = []  # testing labels

for features, label in training_data[:700]:
    x_training.append(features)
    y_training.append(label)

for features, label in training_data[-300:]:
    x_test.append(features)
    y_test.append(label)


def featureExtraction(glcm):
    energy = skimage.feature.graycoprops(glcm, 'energy')
    contrast = skimage.feature.graycoprops(glcm, 'contrast')
    homogeneity = skimage.feature.graycoprops(glcm, 'homogeneity')
    correlation = skimage.feature.graycoprops(glcm, 'correlation')
    dissimilarity = skimage.feature.graycoprops(glcm, 'dissimilarity')
    ASM = skimage.feature.graycoprops(glcm, 'ASM')
    extractedFeatures = [energy[0][0], contrast[0][0], homogeneity[0][0], correlation[0][0], dissimilarity[0][0], ASM[0][0]]
    return extractedFeatures



final_training = []
for img in x_training:
    glcm = skimage.feature.graycomatrix(img, distances=[1], angles=[0], symmetric=True)
    features = featureExtraction(glcm)
    final_training.append(features)

final_test = []
for img in x_test:
    glcm = skimage.feature.graycomatrix(img, distances=[1], angles=[0], symmetric=True)
    features = featureExtraction(glcm)
    final_test.append(features)

classifier = KNeighborsClassifier(n_neighbors=1)
classifier.fit(final_training, y_training)

y_pred = classifier.predict(final_test)
print("Accuracy: ")
print(accuracy_score(y_test, y_pred))
