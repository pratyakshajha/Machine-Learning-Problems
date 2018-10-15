## Dataset Description 
Name: Optical Recognition of Handwritten Digits Data Set
Data Set Characteristics:
+	Number of Instances: 5620
+	Number of Attributes: 64
+	Attribute Information: 8x8 image of integer pixels in the range 0..16.
+	Missing Attribute Values: None
+	Creator: E. Alpaydin (alpaydin '@' boun.edu.tr)
+	Date: July; 1998

This is a copy of the test set of the UCI ML hand-written digits datasets
[link](http://archive.ics.uci.edu/ml/datasets/Optical+Recognition+of+Handwritten+Digits). The data set contains images of hand-written digits: 10 classes where each class refers to a digit.
Pre-processing programs made available by NIST were used to extract normalized bitmaps of handwritten digits from a preprinted form. From a total of 43 people, 30 contributed to the training set and different 13 to the test set. 32x32 bitmaps are divided into nonoverlapping blocks of 4x4 and the number of on pixels are counted in each block. This generates an input matrix of 8x8 where each element is an integer in the range 0 to 16. This reduces dimensionality and gives invariance to small distortions.


I have used cross-validation to find the best suited algorithm over three differently processed dataset. The dataset was clean with no missing values or duplicate data. As the number of attributes was large, we adopted three different ways to process data before using them for prediction.
+	Used full dataset.
+	Used dataset without attributes that were less correlated with target class.
All attributes whose magnitude of correlation was greater than 0.1 were only used. This reduced 64 attributes to 29 attributes.
+	Used PCA to extract important features.
PCA or Principal Component Analysis is a statistical procedure that uses transformation to convert a set of observations of possibly correlated variables into a smaller set of uncorrelated variables called principal components. It was applied on remaining 29 attributes from above step to reduce the dataset further to 20 attributes.

All these datasets were separately used for cross validation and best of them was chosen. For cross-validation we used the following algorithms:
+	Logistic Regression: It performs classification by estimating probabilities using a logistic function, which is the cumulative logistic distribution.
+	Support Vector Machine: Instead of a line in linear regression, it uses hyper-plane for making the decision boundary.
+	K Neighbours: It classifies to the class most common among its k nearest neighbours.
+	Decision Tree: It uses a decision tree to go from observations about an item to conclusions about the item's class.

Decision Tree with 29 attributes gave highest accuracy score of 100%.
