from CNN_model import *
from time import sleep
import pyautogui
import cv2


def image_to_feature(image, size=(128, 128)):
    if image is not None:
        #image = np.array(image) / 255.0
        return cv2.resize(image, size)
    else:
        return None


# load the trained neural network
saver = tf.train.Saver()
sess = tf.InteractiveSession()
saver.restore(sess, 'checkpoint_dir/trained_cnn')

while True:
    t = int(input("\nEnter time(in sec) to take the screenshot or \'-1\' to exit:"))
    if t == -1:
        exit()


# taking screenshot
    sleep(t)
    pyautogui.screenshot("screenshot.png")
    print("SCREENSHOT TAKEN AND SAVED as \'screenshot.png\'")


# selecting the roi
    _ = input("Press Enter to select the Region of Interest from the screenshot.")
    screenshot = cv2.imread('screenshot.png')
    roi = cv2.selectROI(screenshot)
    roi_img = screenshot[int(roi[1]): int(roi[1]+roi[3]), int(roi[0]): int(roi[0]+roi[2])]
    cv2.imwrite('region_of_interest.png', roi_img)


# process the region of interest
    features = image_to_feature(roi_img)
    features = features.reshape((1, 128, 128, 3))


# predict the class using trained NN
    prediction = predict.eval(session=sess, feed_dict={X: features})
    label = label_dict[prediction[0]]
    print("Predicted class:", prediction)
    print(label_dict)


# destroy the windows opened by opencv
    cv2.destroyAllWindows()
