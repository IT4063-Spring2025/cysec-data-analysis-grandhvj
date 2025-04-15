#!/usr/bin/env python
# coding: utf-8

# ## In-Class Activity - Cyber Security Data Analysis 
# This notebook will guide you through the process of analyzing a cyber security dataset. Follow the TODO tasks to complete the assignment.
# 

# # Step 1: Importing the required libraries
# 
# TODO: Import the necessary libraries for data analysis and visualization.

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
pd.options.display.max_rows = 999
warnings.filterwarnings("ignore")


# # Step 2: Loading the dataset
# 
# TODO: Load the given dataset.

# In[3]:


link = "./Data/CySecData.csv"
df = pd.read_csv(link)


# # Step 3: Display the first few rows of the dataset
# TODO: Import the necessary libraries for data analysis and visualization.

# In[4]:


df.head()


# # Step 4: Initial info on the dataset.
# 
# TODO: Provide a summary of the dataset.

# In[5]:


df.info()
df.describe(include='all')  # Gives a summary including categorical columns


# # Step 5: Creating dummy variables
# TODO: Create dummy variables for the categorical columns except for the label column "class".

# In[6]:


dfDummies = pd.get_dummies(df.drop('class', axis=1), drop_first=True)


# # Step 6: Dropping the target column
# TODO: Drop the target column 'class' from the dataset.

# In[7]:


dfDummies = dfDummies.drop


# # Step 7: Importing the Standard Scaler
# TODO: Import the `StandardScaler` from `sklearn.preprocessing`.

# In[8]:


from sklearn.preprocessing import StandardScaler


# # Step 8: Scaling the dataset
# TODO: Scale the dataset using the `StandardScaler`.

# In[17]:


print(type(dfDummies))


# In[18]:


# Assuming df is your original DataFrame
dfDummies = pd.get_dummies(df, drop_first=True)


# In[19]:


import numpy as np

# Step 1: Check column data types
print("Column data types:\n", dfDummies.dtypes)

# Step 2: Identify non-numeric columns
non_numeric_cols = dfDummies.select_dtypes(exclude=[np.number]).columns
print("Non-numeric columns:", list(non_numeric_cols))

# Step 3: Drop non-numeric columns
dfDummies_cleaned = dfDummies.drop(columns=non_numeric_cols)

# Step 4: Scaling
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
dfNormalized = pd.DataFrame(scaler.fit_transform(dfDummies_cleaned), columns=dfDummies_cleaned.columns)


# In[20]:


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
dfNormalized = pd.DataFrame(scaler.fit_transform(dfDummies), columns=dfDummies.columns)


# # Step 9: Splitting the dataset
# TODO: Split the dataset into features (X) and target (y).

# In[21]:


y = df['class']
X = dfNormalized


# # Step 10: Importing the required libraries for the models
# TODO: Import the necessary libraries for model training and evaluation.

# In[22]:


from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, KFold
import numpy as np


# # Step 11: Defining the models (Logistic Regression, Support Vector Machine, Random Forest)
# TODO: Define the models to be evaluated.

# In[23]:


models = []
models.append(('LR', LogisticRegression()))
models.append(('SVM', SVC()))
models.append(('RandomForestClassifier', RandomForestClassifier()))




# # Step 12: Evaluating the models
# TODO: Evaluate the models using 10 fold cross-validation and display the mean and standard deviation of the accuracy.
# Hint: Use Kfold cross validation and a loop

# In[24]:


# 10-fold cross-validation
kfold = KFold(n_splits=10, shuffle=True, random_state=42)

for name, model in models:
    cv_results = cross_val_score(model, X, y, cv=kfold, scoring='accuracy')
    print(f"{name}: Mean Accuracy = {cv_results.mean():.4f}, Std Dev = {cv_results.std():.4f}")


# # Step 13: Converting the notebook to a script
# TODO: Convert the notebook to a script using the `nbconvert` command.

# In[26]:


#get_ipython().system('jupyter nbconvert --to script notebook.ipynb')

