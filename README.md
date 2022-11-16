Two recommender systems developed as part of an assignment. One system is a context-aware content-based recommmender, the other is a collaborative filtering recommender.

On the command line, navigate to the directory which includes the recommender system files.
Run CLI.py using Python. This will open the command line interface for the recommender systems.

Enter an existing user ID to receive tailored recommendations or register as new user to receive random recommendations.

The evaluation for the systems is included in CACB.py and CF.py, respectively.

There are 3 .csv files containing the output of data_preparation.py
The original JSON files can be downloaded at https://www.yelp.com/dataset. 
These were too large to submit and the necessary data used is all included in the .csv files.

Libraries:
	pandas
	numpy
	math
	datetime
	sklearn.model_selection -> train_test_split
	sklearn.neighbors -> KNeighborsClassifier
	sklearn.metrics -> mean_squared_error
	scipy.sparse.linalg -> svds
	
