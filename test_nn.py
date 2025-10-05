# test_model.py
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler
import pandas as pd

def scale_dataset(dataframe, oversample = False):
  X = dataframe[dataframe.columns[:-1]].values
  y = dataframe[dataframe.columns[-1]].values

  # Convert y to numeric, coercing errors to NaN
  y = pd.to_numeric(y, errors='coerce')

  # Drop rows where y is NaN
  nan_mask = np.isnan(y)
  X = X[~nan_mask]
  y = y[~nan_mask]

  scaler = StandardScaler()
  X = scaler.fit_transform(X)

  data = np.hstack((X, np.reshape(y, (-1,1))))

  if oversample:
    ros = RandomOverSampler()
    X, y = ros.fit_resample(X, y)
    data = np.hstack((X, np.reshape(y, (-1,1)))) # Recreate data with oversampled data


  return data, X, y

# Load test data
df_clean = pd.read_excel('cleaned_data.xlsx')  # same source as training



_, _, test = np.split(df_clean.sample(frac=1), [int(0.6 * len(df)), int(0.8 * len(df))])
test, X_test, y_test = scale_dataset(test, oversample=False)

# Load trained model
model = tf.keras.models.load_model("nn_model.h5") #the weight file

# Predict
y_pred = model.predict(X_test)
y_pred = np.argmax(y_pred, axis=1) + 1

# Evaluate
print(classification_report(y_test, y_pred))

y_true_bin = np.where(np.isin(y_test, [2, 3]), 1, 0)
y_pred_bin = np.where(np.isin(y_pred, [2, 3]), 1, 0)
tn, fp, fn, tp = confusion_matrix(y_true_bin, y_pred_bin).ravel()
print("False Negatives (FN):", fn)
print("FN Rate:", fn / len(y_test))