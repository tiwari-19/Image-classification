import os
import cv2
import numpy as np
import _pickle

np.random.seed(7)


def image_to_feature(image, size=(32, 32)):
    if image is not None:
        return cv2.resize(image, size).flatten()
    else:
        return None

pwd = os.getcwd()
DATASET_PATH = os.path.join(pwd, 'dataset')
CHECKPOINT_PATH = os.path.join(pwd, 'checkpoint_dir')

if not os.path.isdir(CHECKPOINT_PATH):
    os.makedirs(CHECKPOINT_PATH)

data = []
labels = []

print("Loading Data folder")
classes = os.listdir(DATASET_PATH)
for c in classes:
    print("PROCESSING", c, "images")
    class_folder = os.path.join(DATASET_PATH, c)
    images_in_class = os.listdir(class_folder)
    for img in images_in_class:
        image_path = os.path.join(class_folder, img)
        image = cv2.imread(image_path)
        if image is None:
            print("Failed to process ", image_path)
        else:
            features = image_to_feature(image)
            data.append(features)
            labels.append(c)
            # print("Successfully processed", image_path)


print("\n CONVERSION OF IMAGES TO FEATURE VECTORS COMPLETED!!")


data = np.array(data)/255.0
labels = np.array(labels)
labels = labels.reshape(len(labels), )


# saving the features and labels to the disk.
_pickle.dump(data, open(os.path.join(CHECKPOINT_PATH, 'dataset_feats.pkl'), 'wb'))
_pickle.dump(labels, open(os.path.join(CHECKPOINT_PATH, 'labels.pkl'), 'wb'))