# Image-classification
Image classification int o 4 categories - car, ship, motorcycle, plane
Assignment
 To identify the object on the screen if the object boundary is selected by user.


# Image Preprocessing
	First, of all a dataset of four different objects (which can be extended further) is built
	by downloading around ~100 images per objects. However this is very small dataset as it is collected
	manually, but a larger dataset will always be preferred.
	
	Each image is rescaled to 32 x 32 x 3, and flatten into 1-D vector

	The features are saved to 'dataset_feats.pkl' and labels are saved to 'labels.pkl' using the 
	Pickle library.


# Classification

	  RandomForestClassifier is used to classify the images.
	  n_estimators = 200
	  Accuracy achieved on validation data = 71.7 %.
	  The fitted model is saved as classifier.pkl using the pickle library.


# Main.py
	
 	This combines all the parts of the solution.
	A user could simply run this to test the model.
	
	First of all, the user will be prompt to enter the time to take screenshot. This implementation could
	be changed as per the requirement.
	
	After the screenshot is taken and saved, the user will be asked to select the boundary of the object
	by draging the mouse. This selected area will be our Region of Interest.
	
	We feed this region of interest selected by the user to our fitted RandomForest Classifier.
	
	The process continues until the user presses '-1'.
	

#Note: The virtual environment must be activated before running the script.
