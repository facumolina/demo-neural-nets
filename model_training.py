# import pandas library
import pandas as pd

# import sklearn elements
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Other useful imports
from time import time

# Load the dataset and show some info
def load_dataset():
	print("> Loading the Iris dataset..")
	# Location of dataset
	url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
	# Assign colum names to the dataset
	column_names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'Class']
	# Read the data and show some info
	irisdata = pd.read_csv(url, names=column_names)
	print("samples:" , len(irisdata.index))
	print("attributes:" , (len(irisdata.columns)-1))
	print(irisdata.head())
	print()
	return irisdata


# Preprocess the dataset
def preprocess(irisdata):
	print("> Preprocessing the Iris dataset")
	# Assign data from first four columns to X variable
	X = irisdata.iloc[:, 0:4]
	# Assign data from first fifth columns to y variable
	y = irisdata.select_dtypes(include=[object])
	print("original classes:",y.Class.unique())
	le = preprocessing.LabelEncoder()
	y = y.apply(le.fit_transform)
	print("encoded classes:",y.Class.unique())
	print()
	return X, y

# Split the data into training and test sets
def split_data(X,y):
	print("> Splitting data into training and test set")
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state=0)
	print("training set size:",len(X_train))
	print("test set size:",len(X_test))
	print()
	return X_train, X_test, y_train, y_test

# Build and traing a neural net
def build_and_train_neural_net(X_train, y_train):
	print("> Training a Feed Forward Neural Net")
	hidden_layers = [10, 10] 
	iterations = 1000
	mlp = MLPClassifier(hidden_layer_sizes=hidden_layers, max_iter=iterations, random_state=0)
	print("hidden layers:",str(hidden_layers))
	print("iterations:",iterations)
	t0 = time()
	mlp.fit(X_train, y_train.values.ravel())
	print('training time: %0.3fs' % (time() - t0))
	print()
	return mlp

# Evaluate the trained neural net on the test set
def evaluate_neural_net(nn, X_test, Y_test):
	print("> Evaluating the Neural Net")
	predictions = nn.predict(X_test)
	print("Condusion matrix")
	print(confusion_matrix(y_test,predictions))
	print()
	print("Classification report")
	print(classification_report(y_test,predictions))


# Load and process the data
iris = load_dataset()
X,y = preprocess(iris)

# Split data
X_train, X_test, y_train, y_test = split_data(X,y)

# Train the neural net
nn = build_and_train_neural_net(X_train, y_train)

# Evaluate the neural net and measure the performance
evaluate_neural_net(nn, X_test, y_test)

print("Done!")

