---
title       : An Introduction to Classification
description : Training, Testing, and Evaluating a Simple Binary Classifier on Congressional Voting Data
attachments :
  slides_link : https://s3.amazonaws.com/assets.datacamp.com/course/teach/slides_example.pdf

--- type:NormalExercise lang:python xp:100 skills:1 key:72b185d77c
## Machine Learning: What is it?

This short course will immerse you into the world of machine learning. Background knowledge expected is a basic familiarity with the Python data analysis libraries NumPy and Pandas. You will train and evaluate a machine learning model to predict the party a US House of Representatives Congressman belongs to based on his/her past voting record. Though the focus of this course is on application and not theory, be prepared to learn some new terminology.

First, what is machine learning? In its broadest sense, it is the art and science of getting computers to act without being explicitly programmed. This course focuses on what is known as supervised machine learning, in which the computer learns from labeled historical data to make future predictions that may either be continuous (What is the value of this stock going to be tomorrow?) or categorical (Is the value of this stock going to go up or down tomorrow?). The latter is called classification, while the former is regression. In our case, the predictions are categorical: Based on his/her votes, is this Congressman a Democrat or a Republican? Therefore, our problem is one of classification and not regression.

Supervised machine learning models makes predictions based on features that represent some measurable properties relevant to the data.  In our case, the features consist of the voting decisions made by Congressmen on particular issues. Strong features lead to better performing models. Say we want to predict party affiliation. Is it a reasonable hypothesis to assume that democrats and republicans vote differently on certain issues? If so, then having data about past voting records can be a useful set of features. Else, we would have to look at other data sources - maybe age or demographic information. Machine learning allows us to test different hypotheses on an unprecedented scale, and this is where it draws much of its power.

Let's dive in. The dataset of Congressional voting records, `df`, has already been loaded into a pandas dataframe and is available in the workspace. If you want to learn how to import data in Python, refer to the DataCamp course on Importing Data in Python. 

Let's do some basic Exploratory Data Analysis (EDA) to understand its structure. 

*** =instructions
- Use the pandas function `df.shape`, to display the dimensionality of the dataset. This tells us how many samples there are in the data (number of rows), and how many columns.
- Practice indexing with pandas to access the first number returned `df.shape`. This is the number of samples.
- Now return the other number. This is the number of columns. 
- To return the names of the columns, use the function `df.columns`.

*** =hint
- Fill in the blanks with the appropriate pandas function and index values. 
- Remember, python, and by extension, pandas, is 0 indexed!

*** =pre_exercise_code
```{python}
import pandas as pd

df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/voting-records/house-votes-84.data',
                header=None, names = ['infants', 'water', 'budget', 'physician', 'salvador', 'religious',
                                     'satellite', 'aid', 'missile', 'immigration', 'synfuels', 'education',
                                     'superfund', 'crime', 'duty_free_exports', 'eaa_rsa'])

df = df.reset_index()
df.rename(columns = {'index': 'party'}, inplace = True)
```

*** =sample_code
```{python}
# What are the dimensions of this dataframe?
print(df._)

# How many samples does this dataset have?
print(df._[_])

# How many columns does this dataset have?
print(df._[_])

# List out the columns of this dataset
print(df._)

```

*** =solution
```{python}
# What are the dimensions of this dataframe?
print(df.shape)

# How many samples does this dataset have?
print(df.shape[0])

# How many columns does this dataset have?
print(df.shape[1])

# List out the columns of this dataset
print(df.columns)
```

*** =sct
```{python}
msg1 = "Make sure to print out the dimensions of `df` like this: `print(df.shape)`."
test_function("print", 1, incorrect_msg = msg1)

msg2 = "Make sure to print out the number of samples in `df` like this: `print(df.shape[0])`."
test_function("print", 2, incorrect_msg = msg2)

msg3 = "Make sure to print out the number of columns in `df` like this: `print(df.shape[1])`."
test_function("print", 3, incorrect_msg = msg3)

msg4 = "Make sure to print out the columns in `df` like this: `print(df.columns)`."
test_function("print", 4, incorrect_msg = msg4)

success_msg("Great work!")
```
--- type:MultipleChoiceExercise lang:python xp:50 skills:1 key:6ec60c3fcd
## Exploratory Data Analysis

The point of the earlier exercise was to get us familiar with our the structure of our dataset. This is known as Exploratory Data Analysis (EDA), and it is a vital first step in any data analysis project: We need to know what our dataset looks like before we can think about what machine learning models we can apply to it!

EDA can be as basic or detailed as you need it to be. Let's continue our EDA and take a look at the first few rows of the dataframe. To do this, use the pandas function `head` like so: `[dataframe name].head()`. Here, the dataframe has been loaded into the workspace as `df`.

Once you print out the first few rows of the dataframe, notice how in addition to the "yes" and "no" votes, there are also missing values represented by the "?".

How many features and samples are in this dataframe? Refer to the previous exercise where we computed these values, and feel free to use the IPython Shell to continue exploring the data. And take care to note that the number of columns does not equal the number of features! 

*** =instructions
- 435 features, 16 samples
- 17 features, 435 samples
- 16 features, 17 samples 
- 16 features, 435 samples

*** =hint
Remember, the number of columns does not equal the number of features! One of the columns represents the target variable (republican/democrat). 

*** =pre_exercise_code
```{python}
# The pre exercise code runs code to initialize the user's workspace.
# You can use it to load packages, initialize datasets and draw a plot in the viewer

import pandas as pd

df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/voting-records/house-votes-84.data',
                header=None, names = ['infants', 'water', 'budget', 'physician', 'salvador', 'religious',
                                     'satellite', 'aid', 'missile', 'immigration', 'synfuels', 'education',
                                     'superfund', 'crime', 'duty_free_exports', 'eaa_rsa'])

```

*** =sample_code
```{python}
#Use the `head` function in Pandas to print the first few rows of the dataframe
print(df.___)
```


*** =sct
```{python}

msg1 = "That is not correct! We have more than 16 samples."
msg2 = "That is not correct! Is the number of features equal to the number of columns?"
msg3 = "That is not correct! The number of samples is equal to the number of rows, not columns."
msg_success = "Exactly! There are 16 features and 435 samples."
test_mc(4, [msg1, msg2, msg3, msg_success])
```

--- type:NormalExercise lang:python xp:100 skills:1 key:fad5659252
## Imputing Missing Data

As you saw in the previous exercise, we do not know the votes of certain Congressmen on certain issues. These are represented by the '?' instead of a 'yes' ('y') or a 'no' ('n'). Do we just delete these rows? That would throw away most of our data! Let's instead develop a reasonable strategy to fill in these missing values (the technical term is imputation). This is where domain knowledge comes in handy. For our purposes, let's replace the '?'s with the probability of all other representatives voting 'yes' on that particular issue.  

Imputing missing data is a part of data preprocessing. Real world datasets are often very messy and require some preprocessing before they can be fed into a machine learning model. For example, machine learning algorithms take in numeric values. A 'yes' or a 'no' is not numeric: So let's convert the 'y's to 1s and the 'n's to 0s.

Recall that in supervised machine learning, the model learns from labeled data to make future predictions on unlabeled data. Here, the party affiliation ('Democrat' or 'Republican') represents the labels. If our model is successful (and we will explore in future lessons what constitutes 'successful'), then when unlabeled data about voting information comes in, our model will use what it has learned from the labeled data to assign labels ('Democrat' or 'Republican') to the new, unlabeled, data points.

In unsupervised machine learning, there is no labeled data. There is no initial learning, or 'supervision', in which the model looks at past labeled data to try and capture underlying patterns. This is where the terms "Supervised" and "Unsupervised" come from.

*** =instructions
- Replace all the ys with 1s.
- Replace all the ns with 0s.
- Replace all the ?s with NaNs (This is an efficient way to internally represent missing data, and allows us to impute more easily)
- Use the 'fillna' function to replace the NaNs with the mean. This code has been written for you: There is no need to change it!

*** =hint
- We want to convert all 'yes' votes to 1s, and 'no' votes to 0s, and all '?' to NaNs (Not a Number). Fill in the missing code appropriately.
- Do not be intimidated by the last line of code: We are just filling in the missing values (with `fillna`) with the average of all votes on that issue (with the `mean` function). Just fill in the blank with `mean`.

*** =pre_exercise_code
```{python}
import pandas as pd
import numpy as np
df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/voting-records/house-votes-84.data',
                header=None, names = ['infants', 'water', 'budget', 'physician', 'salvador', 'religious',
                                     'satellite', 'aid', 'missile', 'immigration', 'synfuels', 'education',
                                     'superfund', 'crime', 'duty_free_exports', 'eaa_rsa'])

df = df.reset_index()
df.rename(columns = {'index': 'party'}, inplace = True)
```

*** =sample_code
```{python}
# Change all the y's to 1's
df[df == 'y'] = _

# Change all the n's to 0's
df[df == '_'] = _

# Change the ?'s to NaNs
df[df == '_'] = np.nan

# Now, impute the NaNs with the mean of each column
df.iloc[:, 1:] = df.iloc[:, 1:].apply(lambda x: x.fillna(x._()))

```

*** =solution
```{python}
# Change all the y's to 1's
df[df == 'y'] = 1

# Change all the n's to 0's
df[df == 'n'] = 0

# Change the ?'s to NaNs
df[df == '?'] = np.nan

# Now, impute the NaNs with the mean of each column
df.iloc[:, 1:] = df.iloc[:, 1:].apply(lambda x: x.fillna(x.mean()))

```
*** =sct
```{python}

success_msg("Great work!")
```
--- type:NormalExercise lang:python xp:100 skills:1 key:2b648b3e6b
## Train and Test Split

Here, we are doing supervised machine learning. We want our model to learn from past voting records to approximate a function that effectively maps future voting decisions to party affiliation. The historical, labeled data we feed into the model for it to learn from is known as training data.

We don't want to feed it all the data, however. We need it to be able to generalize well to unseen future data. Otherwise, it leans too heavily on the idiosyncracies (or noise) in the training data, and fails to make good predictions when new data comes in.

For this purpose, we hold out some data for model evaluation. We train our model using the training data, and evaluate it on the testing data. Scikit-learn provides us a useful `train_test_split` function for this.

It is crucial to note that when testing, we do not want the model to be able to see the labels! This would be cheating. So we drop the response variable ('party', in our case) from the test data.


*** =instructions
- Import `train_test_split` from `sklearn.cross_validation`.
- Create an array, y, for the response variable ('party').
- Create the feature vector, X, by dropping the response variable 'party'
- Split the data so that 60% (0.6) is available for training and 40% (0.4) is held out for testing.

*** =hint
- 

*** =pre_exercise_code
```{python}
import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split

df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/voting-records/house-votes-84.data',
                header=None, names = ['infants', 'water', 'budget', 'physician', 'salvador', 'religious',
                                     'satellite', 'aid', 'missile', 'immigration', 'synfuels', 'education',
                                     'superfund', 'crime', 'duty_free_exports', 'eaa_rsa'])

df = df.reset_index()
df.rename(columns = {'index': 'party'}, inplace = True)

df[df == 'y'] = 1
df[df == 'n'] = 0
df[df == '?'] = np.nan
df.iloc[:, 1:] = df.iloc[:, 1:].apply(lambda x: x.fillna(x.mean()))
```

*** =sample_code
```{python}
# Import train_test_split from sklearn.cross_validation

# Create arrays for the features and the response variable. As a reminder, the response variable is 'party'
y = df['_']
X = df.drop('_', axis=1)

# Split the data into a training and testing set, such that the training set size is 60% of the data
X_train, X_test, y_train, y_test = train_test_split(_, _, test_size = _)
```
*** =solution
```{python}
# Import train_test_split from sklearn.cross_validation
from sklearn.cross_validation import train_test_split

# Create arrays for the features and the response variable. As a reminder, the response variable is 'party'
y = df['party']
X = df.drop('party', axis=1)

# Split the data into a training and testing set, such that the training set size is 60% of the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.4)

```
*** =sct
```{python}
# SCT written with pythonwhat: https://github.com/datacamp/pythonwhat/wiki
test_import("sklearn.cross_validation.train_test_split")

test_object("y")
test_object("X")
test_object("X_train")
test_object("X_test")
test_object("y_train")
test_object("y_test")
test_function("sklearn.cross_validation.train_test_split")

success_msg("Great work!")
```
--- type:NormalExercise lang:python xp:100 skills:1 key:b0329a110c
## K Nearest Neighbors: A simple classifier

Of all the numerous classification algorithms that are used today, K Nearest Neighbors is the most intuitive, and is what we will use in this course. In essence, it makes its predictions by taking a majority vote of its nearest neighbors. All of our training samples are internally represented as vectors in a multidimensional feature space, each with a label ('democrat'/'republican'). When we want to predict the party affiliation of a new sample, we look at the party affiliations of the points closest to our new sample. If we look at 5 neighbors, and 3 of them are republican, while 2 are democrat, we predict that our new sample will be a republican.

Let's make a prediction using 5 neighbors.

*** =instructions
- All you need to do is specify how many neighbors you want to train the model on
- Observe how we first train the model with the `fit` function, and then evaluate it on the test data

*** =hint


*** =pre_exercise_code
```{python}
import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split

df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/voting-records/house-votes-84.data',
                header=None, names = ['infants', 'water', 'budget', 'physician', 'salvador', 'religious',
                                     'satellite', 'aid', 'missile', 'immigration', 'synfuels', 'education',
                                     'superfund', 'crime', 'duty_free_exports', 'eaa_rsa'])

df = df.reset_index()
df.rename(columns = {'index': 'party'}, inplace = True)

df[df == 'y'] = 1
df[df == 'n'] = 0
df[df == '?'] = np.nan
df.iloc[:, 1:] = df.iloc[:, 1:].apply(lambda x: x.fillna(x.mean()))

# Create arrays for the features and the response variable. As a reminder, the response variable is 'party'
y = df['party']
X = df.drop('party', axis=1)

# Split the data into a training and testing set, such that the training set size is 60% of the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.4)
```

*** =sample_code
```{python}

#Import KNeighborsClassifier from sklearn.neighbors
from sklearn.neighbors import KNeighborsClassifier

#Import accuracy_score from sklearn.metrics
from sklearn.metrics import accuracy_score

#Create the classifier with 5 neighbors
knn = KNeighborsClassifier(n_neighbors=_)

#Fit the classifier to the training data (X_train and y_train)
knn.fit(X_train, y_train)

# Use the fitted model to make predictions on the test data (knn.predict(X_test)), and compute the accuracy
accuracy = accuracy_score(y_test, knn.predict(X_test))

#Print the accuracy
print(____)

```

*** =solution
```{python}
#Import KNeighborsClassifier from sklearn.neighbors
from sklearn.neighbors import KNeighborsClassifier

#Import accuracy_score from sklearn.metrics
from sklearn.metrics import accuracy_score

#Create the classifier with 5 nieghbors
knn = KNeighborsClassifier(n_neighbors=5)

#Fit the classifier to the training data (X_train and y_train)
knn.fit(X_train, y_train)

# Use the fitted model to make predictions on the test data (knn.predict(X_test)), and compute the accuracy
accuracy = accuracy_score(y_test, knn.predict(X_test))

#Print the accuracy
print(accuracy)
```
*** =sct
```{python}
# SCT written with pythonwhat: https://github.com/datacamp/pythonwhat/wiki

test_import("sklearn.neighbors.KNeighborsClassifier")
test_import("sklearn.metrics.accuracy_score")

test_object("knn", do_eval=False)

test_function("sklearn.neighbors.KNeighborsClassifier")

test_object("accuracy")

test_function("sklearn.metrics.accuracy_score")

test_function("print")

success_msg("Great work!")
```
--- type:NormalExercise lang:python xp:100 skills:1 key:9a56717464
## K Nearest Neighbors: Your turn!

Follow the steps in the previous exercise to build your own K Nearest Neighbors classifier. Try 7 neighbors, and print out the accuracy score.

*** =instructions
- First, import the relevant modules
- Create the KNN classifier with 7 neighbors and assign to the variable knn
- Fit the classifier to the training data
- Predict on the testing data, and compare the predicted values (from knn.predict(X_test) with the actual values (y_test)) to compute the accuracy score.

*** =hint


*** =pre_exercise_code
```{python}
import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split

df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/voting-records/house-votes-84.data',
                header=None, names = ['infants', 'water', 'budget', 'physician', 'salvador', 'religious',
                                     'satellite', 'aid', 'missile', 'immigration', 'synfuels', 'education',
                                     'superfund', 'crime', 'duty_free_exports', 'eaa_rsa'])

df = df.reset_index()
df.rename(columns = {'index': 'party'}, inplace = True)

df[df == 'y'] = 1
df[df == 'n'] = 0
df[df == '?'] = np.nan
df.iloc[:, 1:] = df.iloc[:, 1:].apply(lambda x: x.fillna(x.mean()))

# Create arrays for the features and the response variable. As a reminder, the response variable is 'party'
y = df['party']
X = df.drop('party', axis=1)

# Split the data into a training and testing set, such that the training set size is 60% of the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.4)
```

*** =sample_code
```{python}
#Import KNeighborsClassifier from sklearn.neighbors
from sklearn.neighbors import ____

#Import accuracy_score from sklearn.metrics
from sklearn.metrics import ____

#Create the classifier with 7 neighbors
knn = KNeighborsClassifier(n_neighbors = _)

#Fit the classifier to the training data (X_train and y_train)
knn.fit(____, _____)

# Use the fitted model to make predictions on the test data (knn.predict(X_test)), and compute the accuracy
accuracy = ____(____, ____)

#Print the accuracy
print(_)
```

*** =solution
```{python}
#Import KNeighborsClassifier from sklearn.neighbors
from sklearn.neighbors import KNeighborsClassifier

#Import accuracy_score from sklearn.metrics
from sklearn.metrics import accuracy_score

#Create the classifier with 7 neighbors
knn = KNeighborsClassifier(n_neighbors=7)

#Fit the classifier to the training data (X_train and y_train)
knn.fit(X_train, y_train)

# Use the fitted model to make predictions on the test data (knn.predict(X_test)), and compute the accuracy
accuracy = accuracy_score(y_test, knn.predict(X_test))

#Print the accuracy
print(accuracy)
```
*** =sct
```{python}
test_import("sklearn.neighbors.KNeighborsClassifier")
test_import("sklearn.metrics.accuracy_score")

test_object("knn", do_eval=False)

test_function("sklearn.neighbors.KNeighborsClassifier")

test_object("accuracy", do_eval=False)

test_function("sklearn.metrics.accuracy_score")

test_function("print")

success_msg("Great work!")
```
--- type:NormalExercise lang:python xp:100 skills:1 key:46064f2b22
## How many neighbors to use?

Code has been written on the right to visualize how the accuracy score varies with different values of k. The `matplotlib.pyplot` Python plotting library has been used for this purpose. To display a plot, we need to end each sequence of plotting statements with a `matplotlib.pyplot.show();`. Here, as is standard practice, `matplotlib.pyplot` has been imported as `plt` for convenience.

To display the plot, type in `plt.show();` at the end of the code! You can also practice labeling your plots to be more descriptive. The x-axis has been labeled for you. Try labeling the y-axis or giving the plot a title!

*** =instructions
- To produce the plot, we first run `plt.plot(ks, scores)`. The values of k are on the X-axis, and the accuracy scores on the Y-axis. We then give our x-axis a label using `plot.xlabel("Number of nearest neighbors")` 
- To display the plot, enter `plt.show();`
- To label the y-axis, use `plt.ylabel("Label")`
- To give the plot a title, use `plt.title("Title")`

*** =hint


*** =pre_exercise_code
```{python}
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cross_validation import train_test_split

df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/voting-records/house-votes-84.data',
                header=None, names = ['infants', 'water', 'budget', 'physician', 'salvador', 'religious',
                                     'satellite', 'aid', 'missile', 'immigration', 'synfuels', 'education',
                                     'superfund', 'crime', 'duty_free_exports', 'eaa_rsa'])

df = df.reset_index()
df.rename(columns = {'index': 'party'}, inplace = True)

df[df == 'y'] = 1
df[df == 'n'] = 0
df[df == '?'] = np.nan
df.iloc[:, 1:] = df.iloc[:, 1:].apply(lambda x: x.fillna(x.mean()))

# Create arrays for the features and the response variable. As a reminder, the response variable is 'party'
y = df['party']
X = df.drop('party', axis=1)

# Split the data into a training and testing set, such that the training set size is 60% of the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.4, random_state = 42)
```

*** =sample_code
```{python}
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

scores = []
ks = range(1, 21)
for k in ks:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    score = accuracy_score(y_test, knn.predict(X_test))
    scores.append(score)
    

plt.plot(ks, scores)
plt.xlabel("Number of nearest neighbors")

#Practice giving the y-axis a label, and the plot a title. When you are done, "show" the plot!

plt.___;
```

*** =solution
```{python}
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

scores = []
ks = range(1, 21)
for k in ks:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    score = accuracy_score(y_test, knn.predict(X_test))
    scores.append(score)
    

plt.plot(ks, scores)
plt.xlabel("Number of nearest neighbors")
plt.show();
```
*** =sct
```{python}
# SCT written with pythonwhat: https://github.com/datacamp/pythonwhat/wiki

success_msg("Proceed to see how the accuracy varies with different numbers of neighbors!")
```

--- type:MultipleChoiceExercise lang:python xp:50 skills:1 key:1c7935280e
## Overfitting, Underfitting, and the Bias-Variance Tradeoff

As you can see, when the number of neighbors are either too high or too low, accuracy suffers. Why is this?

Remember our motivation for splitting the data into training and testing sets: We want our model to generalize well to unseen data.

Now consider what happens when we fit a model that considers only the next nearest neighbor. Such a model ends up being excessively complicated, and fits too closely to the training data. This is known as overfitting, where the model fits to the noise in the data instead of capturing the underlying pattern. If your results look suspiciously good (98% accuracy, say), it is likely the model is overfitting to the data it has been trained on. High accuracy on training data and low accuracy on test data is the hallmark of an overfit model. 

On the other end of the spectrum is underfitting. What happens when we look at, say, the 100 nearest neighbors? Such a model can be too simple - it fails to capture the underlying pattern in the data, and is just as poor at generalizing.

We want to find a balance between underfitting and overfitting. Our ideal model is complicated enough to capture the intricacies of the training data, but doesn't rely excessively on the idiosyncracies of the training data: It is simple enough such that when new data comes in, it can make strong predictions based on the underlying relationships it has captured. Trying to find this balance is known as the bias-variance tradeoff.

An overfit model, when tested on new data, can provide wildly varying results when run repeatedly. An underfit model, on the other hand, is too simple - it provides similar (incorrect) results when run each time. A useful analogy to understand this is a dart board. A highly biased model will have darts that are off target, but hit a similar location each time. A model with high variance will have darts that are all over the board, sometimes on target (by chance), and most of the times off target.

It is a tradeoff because to add model complexity is to increase variance and reduce bias, while to reduce model complexity (and resulting overfitting) is to increase bias while reducing variance.

Consider the scenario in which you suspect your model is overfit. Where does it fall on the bias-variance spectrum?

*** =instructions
- High bias, low variance
- Medium bias, medium variance
- Overfitting is unrelated to bias-variance
- Low bias, high variance

*** =hint

*** =pre_exercise_code
```{python}
# The pre exercise code runs code to initialize the user's workspace.
# You can use it to load packages, initialize datasets and draw a plot in the viewer
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cross_validation import train_test_split

df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/voting-records/house-votes-84.data',
                header=None, names = ['infants', 'water', 'budget', 'physician', 'salvador', 'religious',
                                     'satellite', 'aid', 'missile', 'immigration', 'synfuels', 'education',
                                     'superfund', 'crime', 'duty_free_exports', 'eaa_rsa'])

df = df.reset_index()
df.rename(columns = {'index': 'party'}, inplace = True)

df[df == 'y'] = 1
df[df == 'n'] = 0
df[df == '?'] = np.nan
df.iloc[:, 1:] = df.iloc[:, 1:].apply(lambda x: x.fillna(x.mean()))

# Create arrays for the features and the response variable. As a reminder, the response variable is 'party'
y = df['party']
X = df.drop('party', axis=1)

# Split the data into a training and testing set, such that the training set size is 60% of the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.4, random_state = 42)

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

scores = []
ks = range(1, 21)
for k in ks:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    score = accuracy_score(y_test, knn.predict(X_test))
    scores.append(score)
    

plt.plot(ks, scores)
plt.xlabel("Number of nearest neighbors")
plt.show();

```

*** =sct
```{python}
# SCT written with pythonwhat: https://github.com/datacamp/pythonwhat/wiki

msg_bad = "That is not correct!"
msg_success = "Exactly! Overfit models exhibit high variance and low bias"
test_mc(4, [msg_bad, msg_bad, msg_bad, msg_success])
```


