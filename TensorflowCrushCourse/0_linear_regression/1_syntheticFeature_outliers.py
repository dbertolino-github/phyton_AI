'''
The aim of this exercise is to work with features and data to be analysed.
For the form element we create a synthetic feature, in this case simply a product of two existing.
For the latter we present a simple process to analyse the presence of outliers and we 
demonstrate how to delete them.
'''

# Import needed libraries
from __future__ import print_function
import math
from IPython import display
from matplotlib import cm
from matplotlib import gridspec
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics
import os
import tensorflow as tf
from tensorflow.python.data import Dataset

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
tf.logging.set_verbosity(tf.logging.ERROR)
pd.options.display.max_rows = 10
pd.options.display.float_format = '{:.1f}'.format

# Load the dataset
# Randomize the data to avoid pothologic order effects that might harm the performance of Stochastic Gradient Descent
california_housing_dataframe = pd.read_csv("https://download.mlcc.google.com/mledu-datasets/california_housing_train.csv", sep=",")
california_housing_dataframe = california_housing_dataframe.reindex(np.random.permutation(california_housing_dataframe.index))
california_housing_dataframe["median_house_value"] /= 1000.0
california_housing_dataframe
display.display(california_housing_dataframe.describe())

#input function defined to feed the linear regression with our data
def my_input_fn(feature_inputs, targets, batch_size=1, shuffle=True, num_epochs=None):
    """Trains a linear regression model of one feature.
    Args:
      features: pandas DataFrame of features
      targets: pandas DataFrame of targets
      batch_size: Size of batches to be passed to the model
      shuffle: True or False. Whether to shuffle the data.
      num_epochs: Number of epochs for which data should be repeated. None = repeat indefinitely
    Returns:
      Tuple of (features, labels) for next data batch
    """

    # Convert pandas data into a dict of np arrays.
    feature_inputs = {key:np.array(value) for key,value in dict(feature_inputs).items()}

    # Construct a dataset, and configure batching/repeating.
    ds = Dataset.from_tensor_slices((feature_inputs,targets)) # warning: 2GB limit
    ds = ds.batch(batch_size).repeat(num_epochs)

    # Shuffle the data, if specified.
    if shuffle:
      ds = ds.shuffle(buffer_size=10000)

    # Return the next batch of data.
    feature_inputs, labels = ds.make_one_shot_iterator().get_next()
    return feature_inputs, labels

def train_and_evaluate(learning_rate,periods,steps,batch_size, input_feature, targets_label):
    """
    Trains a linear regression model of one feature.
    Args:
    learning_rate: A 'float', the learning rate.
    periods: A non-zero 'int', the total number of training periods.
    steps: A non-zero 'int', the total number of training steps. A training step consists of a forward and backward pass using a single batch.
    batch_size: A non-zero 'int', the batch size.
    input_feature: A 'string' specifying a column from 'california_housing_dataframe' to use as input feature.
    targets_label: A 'string' specifying a column from 'california_housing_dataframe' to use as targets label.
    """

    steps_per_period = periods / steps

    feature_data = california_housing_dataframe[[input_feature]]
    targets_data = california_housing_dataframe[[targets_label]]
    feature_columns = [tf.feature_column.numeric_column(input_feature)]

    # Create input functions, wrapping the one defined above with a lambda function
    training_input_fn = lambda: my_input_fn(feature_data, targets_data, batch_size=batch_size)
    prediction_input_fn = lambda: my_input_fn(feature_data, targets_data, num_epochs=1, shuffle=False)

    # Create a linear regressor object.
    my_optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    my_optimizer = tf.contrib.estimator.clip_gradients_by_norm(my_optimizer, 5.0)
    linear_regressor = tf.estimator.LinearRegressor(
        feature_columns=feature_columns,
        optimizer=my_optimizer
    )

    # Set up to plot the state of our model's line each period.
    plt.figure(figsize=(15, 6))
    plt.subplot(1, 2, 1)
    plt.title("Learned Line by Period")
    plt.ylabel(targets_label)
    plt.xlabel(input_feature)
    sample = california_housing_dataframe.sample(n=300)
    plt.scatter(sample[input_feature], sample[targets_label])
    colors = [cm.coolwarm(x) for x in np.linspace(-1, 1, periods)]

    # Train the model, but do so inside a loop so that we can periodically assess
    # loss metrics.
    print("\nTraining model:" + "\nInput feature: " + input_feature + "\nTargets label: " + targets_label)
    print("RMSE (on training data):")
    root_mean_squared_errors = []
    for period in range (0, periods):
        # Train the model, starting from the prior state.
        linear_regressor.train(
            input_fn=training_input_fn,
            steps=steps_per_period
        )
        # Take a break and compute predictions.
        predictions = linear_regressor.predict(input_fn=prediction_input_fn)
        predictions = np.array([item['predictions'][0] for item in predictions])

        # Compute loss.
        root_mean_squared_error = math.sqrt(
            metrics.mean_squared_error(predictions, targets_data))
        # Occasionally print the current loss.
        print("  period %02d : %0.2f" % (period, root_mean_squared_error))
        # Add the loss metrics from this period to our list.
        root_mean_squared_errors.append(root_mean_squared_error)
        # Finally, track the weights and biases over time.
        # Apply some math to ensure that the data and line are plotted neatly.
        y_extents = np.array([0, sample[targets_label].max()])

        weight = linear_regressor.get_variable_value('linear/linear_model/%s/weights' % input_feature)[0]
        bias = linear_regressor.get_variable_value('linear/linear_model/bias_weights')

        x_extents = (y_extents - bias) / weight
        x_extents = np.maximum(np.minimum(x_extents,
                                          sample[input_feature].max()),
                                          sample[input_feature].min())
        y_extents = weight * x_extents + bias
        plt.plot(x_extents, y_extents, color=colors[period])

    print("Model training finished.\n")

    # Output a graph of loss metrics over periods.
    plt.subplot(1, 2, 2)
    plt.ylabel('RMSE')
    plt.xlabel('Periods')
    plt.title("Root Mean Squared Error vs. Periods")
    plt.tight_layout()
    plt.plot(root_mean_squared_errors)

    # Output a table with calibration data.
    calibration_data = pd.DataFrame()
    calibration_data["predictions"] = pd.Series(predictions)
    calibration_data["targets"] = targets_data
    display.display(calibration_data.describe())

    print("Final RMSE (on training data): %0.2f" % root_mean_squared_error)
    return calibration_data

'''
Both the total_rooms and population features count totals for a given city block.
But what if one city block were more densely populated than another? 
We can explore how block density relates to median house value by creating a synthetic feature that's a ratio of total_rooms and population.
'''
#Create a sythetic feature beacause different cities probably have different population density
california_housing_dataframe["rooms_per_person"] = (california_housing_dataframe["total_rooms"]/california_housing_dataframe["population"])

#Adjust our input deleting outliers
california_housing_dataframe["rooms_per_person"] = (
    california_housing_dataframe["rooms_per_person"]).apply(lambda x: min(x, 5))

calibration_data = train_and_evaluate(
    learning_rate=0.00005, 
    periods=10, 
    steps=500, 
    batch_size=5, 
    input_feature='rooms_per_person', 
    targets_label='median_house_value'
)

#We plot a scattergram to represents the relation between Predictions and targets
plt.figure(figsize=(15, 6))
plt.subplot(1, 2, 1)
display.display(calibration_data.describe())
plt.ylabel('Predictions')
plt.xlabel('Targets')
plt.scatter(calibration_data["predictions"], calibration_data["targets"])

'''
Histograms of feature are a pretty good visualition to point out outliers presence.
'''
#We plot an histogram of our synthetic feature in order to detect outliers and maybe adjust parameters at line 155
plt.subplot(1, 2, 2)
plt.ylabel('Frequency')
plt.xlabel('Value')
_ = california_housing_dataframe["rooms_per_person"].hist()

plt.show()


