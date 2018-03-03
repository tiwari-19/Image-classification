import os
import cv2
import numpy as np
import _pickle
from sklearn.preprocessing import LabelEncoder


def image_to_feature(image, size=(128, 128)):
    if image is not None:
        #image = np.array(image)/255.0
        return cv2.resize(image, size)
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


# converting the list of labels into proper format
le = LabelEncoder()
num_labels = len(classes)
Labels = le.fit_transform(labels)
Labels += 1

labels_dict = {}
for i in range(len(labels)):
    labels_dict[Labels[i]] = labels[i]


# label_dict will be used to generate the string output ex. 1-> car, 2->motorcycle etc.
print(labels_dict)
data = np.array(data)/255.0
Labels = (np.arange(1, num_labels+1) == Labels[:, None]).astype(np.float32)


# saving the features and labels to the disk.
_pickle.dump(data, open(os.path.join(CHECKPOINT_PATH, 'dataset_feats.pkl'), 'wb'))
_pickle.dump(Labels, open(os.path.join(CHECKPOINT_PATH, 'labels.pkl'), 'wb'))
_pickle.dump(labels_dict, open(os.path.join(CHECKPOINT_PATH, 'labels_dict.pkl'), 'wb'))