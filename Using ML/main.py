from time import sleep
import numpy as np
import pyautogui
import _pickle
import cv2
import os

CHECKPOINT_PATH = os.path.join(os.getcwd(), 'checkpoint_dir')


def image_to_feature(image, size=(32, 32)):
    if image is not None:
        image = np.array(image) / 255.0
        return cv2.resize(image, size).flatten()
    else:
        return None


clf = _pickle.load(open(os.path.join(CHECKPOINT_PATH, 'classifier.pkl'), 'rb'))


while True:
    t = int(input("\nEnter time(in sec) to take the screenshot or \'-1\' to exit:"))
    if t == -1:
        exit()

# taking screenshot
    sleep(t)
    pyautogui.screenshot("screenshot.png")
    print("SCREENSHOT TAKEN AND SAVED as \'screenshot.png\'")

# selecting the roi
    # _ = input("Press Enter to select the Region of Interest from the screenshot.")
    screenshot = cv2.imread('screenshot.png')
    roi = cv2.selectROI(screenshot)
    roi_img = screenshot[int(roi[1]): int(roi[1]+roi[3]), int(roi[0]): int(roi[0]+roi[2])]
    cv2.imwrite('region_of_interest.png', roi_img)

# process the region of interest
    features = image_to_feature(roi_img)
    features = features.reshape(1, len(features))

# predict the class using trained NN
    label = clf.predict(features)
    print(label)

# destroy the windows opened by opencv
    cv2.destroyAllWindows()