---
title       : An Introduction to Classification
description : Insert the chapter description here
attachments :
  slides_link : https://s3.amazonaws.com/assets.datacamp.com/course/teach/slides_example.pdf

--- type:MultipleChoiceExercise lang:python xp:50 skills:1 key:6ec60c3fcd
## A really bad movie

Have a look at the plot that showed up in the viewer to the right. Which type of movies have the worst rating assigned to them?

*** =instructions
- 435 features, 16 samples
- 17 features, 435 samples
- 16 features, 17 samples 
- 16 features, 435 samples

*** =hint
Have a look at the plot. Do you see a trend in the dots?

*** =pre_exercise_code
```{r}
# The pre exercise code runs code to initialize the user's workspace.
# You can use it to load packages, initialize datasets and draw a plot in the viewer

import pandas as pd

df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/voting-records/house-votes-84.data',
                header=None, names = ['infants', 'water', 'budget', 'physician', 'salvador', 'religious',
                                     'satellite', 'aid', 'missile', 'immigration', 'synfuels', 'education',
                                     'superfund', 'crime', 'duty_free_exports', 'eaa_rsa'])

df.head(5)
print(df.head(5))
```

*** =sample_code
```{python}
# What are the dimensions of this dataframe?
df.___

# How many samples does this dataset have?
df.___[__]

# How many features does this dataset have?
df.___[__]

# List out the features of this dataset
df.___

```

*** =solution
```{python}
# What are the dimensions of this dataframe?
df.shape

# How many samples does this dataset have?
df.shape[0]

# How many features does this dataset have?
df.shape[1]

# List out the features of this dataset
df.columns
```

*** =sct
```{r}
# SCT written with pythonwhat: https://github.com/datacamp/pythonwhat/wiki

msg_bad = "That is not correct!"
msg_success = "Exactly! There are 16 features and 435 samples."
test_mc(4, [msg_bad, msg_bad, msg_bad, msg_success])
```

--- type:NormalExercise lang:python xp:100 skills:1 key:72b185d77c
## Plot the movies yourself

Do you remember the plot of the last exercise? Let's make an even cooler plot!

A dataset of movies, `movies`, is available in the workspace.

*** =instructions
- The first function, `np.unique()`, uses the `unique()` function of the `numpy` package to get integer values for the movie genres. You don't have to change this code, just have a look!
- Import `pyplot` in the `matplotlib` package. Set an alias for this import: `plt`.
- Use `plt.scatter()` to plot `movies.runtime` onto the x-axis, `movies.rating` onto the y-axis and use `ints` for the color of the dots. You should use the first and second positional argument, and the `c` keyword.
- Show the plot using `plt.show()`.

*** =hint
- You don't have to program anything for the first instruction, just take a look at the first line of code.
- Use `import ___ as ___` to import `matplotlib.pyplot` as `plt`.
- Use `plt.scatter(___, ___, c = ___)` for the third instruction.
- You'll always have to type in `plt.show()` to show the plot you created.

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
df._

# How many samples does this dataset have?
df._[_]

# How many features does this dataset have?
df._[_]

# List out the features of this dataset
df._

```

*** =solution
```{python}
# What are the dimensions of this dataframe?
df.shape

# How many samples does this dataset have?
df.shape[0]

# How many features does this dataset have?
df.shape[1]

# List out the features of this dataset
df.columns
```

*** =sct
```{python}
# SCT written with pythonwhat: https://github.com/datacamp/pythonwhat/wiki

test_function("df.columns",
              incorrect_msg = "You didn't use `plt.scatter()` correctly, have another look at the instructions.")

success_msg("Great work!")
```
