# Image Classification
	Objective - To identify the object on the screen if the object boundary is selected by user.


# Image Preprocessing
	First, of all a dataset of four different objects (which can be extended further) is built
	by downloading around ~100 images per objects. However this is very small dataset as it is collected
	manually, but a larger dataset will always be preferred.
	
	The images are rescaled to 128x128x3, (3 denotes the color images with RGB values)

	The features are saved to 'dataset_feats.pkl' and labels are saved to 'labels.pkl' using the 
	Pickle library.

# CNN Model & Training
	A Convolutional Neural Network is used to classify the images.

	The architure of the network is - 3 convolutional layers, 2 fully connected layers followed by the 
	softmax output.
	
	AdamOptimizer is used for fast convergence.

	Learning Rate = 0.001. As it is a very small dataset so using the decaying learning rate won't make
	a big difference.

	The network is trained for 30 epoches and achieved the accuracy of 65.3 %.
	
	The trained model is saved to 'trained_cnn' using tensorflow saver function.

# Main.py
	This combines all the parts of the solution.
	A user could simply run this to test the model.
	
	First of all, the user will be prompt to enter the time to take screenshot. This implementation could
	be changed as per the requirement.
	
	After the screenshot is taken and saved, the user will be asked to select the boundary of the object
	by draging the mouse. This selected area will be our Region of Interest.
	
	We feed this region of interest selected by the user to our trained CNN model for prediction.
	
	The process continues until the user presses '-1'.
	

#Note: The virtual environment must be activated before running the script.
