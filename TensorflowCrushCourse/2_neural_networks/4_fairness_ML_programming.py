'''
The aim of this exercise is to explore datasets and evaluate classifiers with fairness in mind, 
noting the ways undesirable biases can creep into machine learning (ML).

Learning Objectives:
 - Increase awareness of different types of biases that can manifest in model data.
 - Explore feature data to proactively identify potential sources of bias before training a model.
 - Evaluate model performace by subgroup rather than in aggregate.

Each example in the dataset contains the following demographic data for a set of individuals who took part in the 1994 Census:
Numeric Features
 - age: The age of the individual in years.
 - fnlwgt: The number of individuals the Census Organizations believes that set of observations represents.
 - education_num: An enumeration of the categorical representation of education. 
                  The higher the number, the higher the education that individual achieved. 
                  For example, an education_num of 11 represents Assoc_voc (associate degree at a vocational school), an education_num of 13 represents Bachelors, 
                  and an education_num of 9 represents HS-grad (high school graduate).
 - capital_gain: Capital gain made by the individual, represented in US Dollars.
 - capital_loss: Capital loss mabe by the individual, represented in US Dollars.
 - hours_per_week: Hours worked per week.

Categorical Features
 - workclass: The individual's type of employer. Examples include: Private, Self-emp-not-inc, Self-emp-inc, Federal-gov, Local-gov, State-gov, Without-pay, and Never-worked.
 - education: The highest level of education achieved for that individual.
 - marital_status: Marital status of the individual. Examples include: Married-civ-spouse, Divorced, Never-married, Separated, Widowed, Married-spouse-absent, and Married-AF-spouse.
 - occupation: The occupation of the individual. Example include: tech-support, Craft-repair, Other-service, Sales, Exec-managerial and more.
 - relationship: The relationship of each individual in a household. Examples include: Wife, Own-child, Husband, Not-in-family, Other-relative, and Unmarried.
 - gender: Gender of the individual available only in binary choices: Female or Male.
 - race: White, Asian-Pac-Islander, Amer-Indian-Eskimo, Black, and Other.
 - native_country: Country of origin of the individual. 
                   Examples include: United-States, Cambodia, England, Puerto-Rico, Canada, Germany, Outlying-US(Guam-USVI-etc), India, Japan, 
                   United-States, Cambodia, England, Puerto-Rico, Canada, Germany, Outlying-US(Guam-USVI-etc), India, Japan, and more.
'''

import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
import tempfile
import seaborn as sns
import itertools
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.metrics import precision_recall_curve

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
tf.logging.set_verbosity(tf.logging.ERROR)
pd.options.display.max_rows = 10
pd.options.display.float_format = '{:.1f}'.format

print('Modules are imported.')

COLUMNS = ["age", "workclass", "fnlwgt", "education", "education_num",
           "marital_status", "occupation", "relationship", "race", "gender",
           "capital_gain", "capital_loss", "hours_per_week", "native_country",
           "income_bracket"]

train_df = pd.read_csv(
    "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data",
    names=COLUMNS,
    sep=r'\s*,\s*',
    engine='python',
    na_values="?")
test_df = pd.read_csv(
    "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test",
    names=COLUMNS,
    sep=r'\s*,\s*',
    skiprows=[0],
    engine='python',
    na_values="?")

# Drop rows with missing values
train_df = train_df.dropna(how="any", axis=0)
test_df = test_df.dropna(how="any", axis=0)
print('UCI Adult Census Income dataset loaded.')

'''
We now want to print the dataframe into a csv and explore it with Factes Overview.
Dataframe is also exported in json to be explored in Facets Dive
https://pair-code.github.io/facets/
In order to do a guided analysis with facets, follow the instructions on the course page:
https://colab.research.google.com/github/google/eng-edu/blob/master/ml/cc/exercises/intro_to_fairness.ipynb?utm_source=mlcc&utm_campaign=colab-external&utm_medium=referral&utm_content=fairness-colab&hl=it#scrollTo=qZ-9vJgSEpHj
'''
#train_df.to_csv('census.csv')
#train_df.to_json('census.json')

'''
After checking out data on Facets a problem came out, a data skew on gender feature was found.
ooking at the ratio between men and women shows how disproportionate the data  is compared to the real world 
where the ratio (at least in the US) is closer to 1:1. This could pose a huge probem in performance across gender.
Considerable measures may need to be taken to upsample the underrepresented group.
More of that the age could be bucketized to become a more relevant categorical feature.
'''
def csv_to_pandas_input_fn(data, batch_size=100, num_epochs=1, shuffle=False):
  return tf.estimator.inputs.pandas_input_fn(
      x=data.drop('income_bracket', axis=1),
      y=data['income_bracket'].apply(lambda x: ">50K" in x).astype(int),
      batch_size=batch_size,
      num_epochs=num_epochs,
      shuffle=shuffle,
      num_threads=1)

print('csv_to_pandas_input_fn() defined.')

'''
TensorFlow requires that data maps to a model. 
To accomplish this, you have to use tf.feature_columns to ingest and represent features in TensorFlow.
'''
#numeric features
age_column = tf.feature_column.numeric_column("age")
fnlwgt_column = tf.feature_column.numeric_column("fnlwgt")   
education_num_column = tf.feature_column.numeric_column("education_num")  
capital_gain_column = tf.feature_column.numeric_column("capital_gain")  
capital_loss_column = tf.feature_column.numeric_column("capital_loss")  
hours_per_week_column = tf.feature_column.numeric_column("hours_per_week")  
  
print('Numeric feature columns defined.')

#categorical features
# Since we don't know the full range of possible values with occupation and
# native_country, we'll use categorical_column_with_hash_bucket() to help map
# each feature string into an integer ID.
occupation_column = tf.feature_column.categorical_column_with_hash_bucket(
    "occupation", hash_bucket_size=1000)
native_country_column = tf.feature_column.categorical_column_with_hash_bucket(
    "native_country", hash_bucket_size=1000)

# For the remaining categorical features, since we know what the possible values
# are, we can be more explicit and use categorical_column_with_vocabulary_list()
gender_column = tf.feature_column.categorical_column_with_vocabulary_list(
    "gender", ["Female", "Male"])
race_column = tf.feature_column.categorical_column_with_vocabulary_list(
    "race", [
        "White", "Asian-Pac-Islander", "Amer-Indian-Eskimo", "Other", "Black"
    ])
education_column = tf.feature_column.categorical_column_with_vocabulary_list(
    "education", [
        "Bachelors", "HS-grad", "11th", "Masters", "9th",
        "Some-college", "Assoc-acdm", "Assoc-voc", "7th-8th",
        "Doctorate", "Prof-school", "5th-6th", "10th", "1st-4th",
        "Preschool", "12th"
    ])
marital_status_column = tf.feature_column.categorical_column_with_vocabulary_list(
    "marital_status", [
        "Married-civ-spouse", "Divorced", "Married-spouse-absent",
        "Never-married", "Separated", "Married-AF-spouse", "Widowed"
    ])
relationship_column = tf.feature_column.categorical_column_with_vocabulary_list(
    "relationship", [
        "Husband", "Not-in-family", "Wife", "Own-child", "Unmarried",
        "Other-relative"
    ])
workclass_column = tf.feature_column.categorical_column_with_vocabulary_list(
    "workclass", [
        "Self-emp-not-inc", "Private", "State-gov", "Federal-gov",
        "Local-gov", "?", "Self-emp-inc", "Without-pay", "Never-worked"
    ])

age_buckets_column = tf.feature_column.bucketized_column(age_column, boundaries=[18, 25, 30, 35, 40, 45, 50, 55, 60, 65])

print('Categorical feature columns defined.')

'''
Now we can explicitly define which feature we will include in our model.
We'll consider gender a subgroup and save it in a separate subgroup_variables list, 
so we can add special handling for it as needed.
'''
variables = [native_country_column, education_column, occupation_column, workclass_column,
             relationship_column, age_buckets_column]
subgroup_variables = [gender_column]
feature_columns = variables + subgroup_variables

'''
With the features now ready to go, we can try predicting income using deep learning.
But first, we have to convert our high-dimensional categorical features into a low-dimensional and dense real-valued vector, which we call an embedding vector. 
Luckily, indicator_column (think of it as one-hot encoding) and embedding_column (that converts sparse features into dense features) helps us streamline the process.
'''

deep_columns = [
    tf.feature_column.indicator_column(workclass_column),
    tf.feature_column.indicator_column(education_column),
    tf.feature_column.indicator_column(age_buckets_column),
    tf.feature_column.indicator_column(gender_column),
    tf.feature_column.indicator_column(relationship_column),
    tf.feature_column.embedding_column(native_country_column, dimension=8),
    tf.feature_column.embedding_column(occupation_column, dimension=8),
]

print('Deep columns created.')

my_optimizer = tf.train.AdagradOptimizer(learning_rate=0.1)
my_optimizer = tf.contrib.estimator.clip_gradients_by_norm(my_optimizer, 5.0)
classifier = tf.estimator.DNNClassifier(
  feature_columns= deep_columns,
  hidden_units=[1024,512],
  optimizer=my_optimizer
)

classifier.train(
  input_fn=csv_to_pandas_input_fn(train_df),
  steps=1000)

evaluation_metrics = classifier.evaluate(
  input_fn=csv_to_pandas_input_fn(train_df),
  steps=1000)

print("Training set metrics:")
for m in evaluation_metrics:
  print(m, evaluation_metrics[m])
print("---")

evaluation_metrics = classifier.evaluate(
  input_fn=csv_to_pandas_input_fn(test_df),
  steps=1000)

print("Test set metrics:")
for m in evaluation_metrics:
  print(m, evaluation_metrics[m])
print("---")
