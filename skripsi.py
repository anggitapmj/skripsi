import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
from pylab import rcParams
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.utils import np_utils
from keras.layers import Dense, Activation, Dropout, Conv1D, MaxPooling1D
import pickle
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from datetime import datetime
dateparse = lambda x: datetime.strptime(x, '%d/%m/%Y %H.%M')

st.title ("AWS Predictive Maintenance System")

st.write("""
# AWS Prediction App
### Created By : [Anggita Putri](https://deantevin.github.io/)
Data obtained from the AWS Rekayasa (http://202.90.199.132/aws-new/) by BMKG.
""")

st.sidebar.header('User Input Features')
st.sidebar.markdown("""
[Example CSV input file](https://raw.githubusercontent.com/anggitapmj/bebas/main/data_temp.csv)
""")

# Collects user input features into dataframe
uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])
if uploaded_file is not None:
      df_Data = pd.read_csv(uploaded_file, parse_dates=['datetime'], index_col='datetime',dayfirst=True)
else:
   urlData = 'https://raw.githubusercontent.com/anggitapmj/bebas/main/citeko-2018-2020.csv'
   df_Data = pd.read_csv(urlData, parse_dates=['datetime'], index_col='datetime',dayfirst=True)

# Combines user input features with entire penguins dataset
# This will be useful for the encoding phase

# START OF CODE TEMPERATURE ANOMALY
with st.beta_expander("Temperature Anomalies"):
  train_size = int(len(df_Data) * .80)
  test_size = len(df_Data) - train_size

  train, test = df_Data.iloc[0:train_size], df_Data.iloc[train_size:len(df_Data)]

  scaler = StandardScaler()
  scaler = scaler.fit(train[['temp']])

  train['temp'] = scaler.transform(train[['temp']])
  test['temp'] = scaler.transform(test[['temp']]) 

  def create_dataset (X, y, time_steps=1):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
      v = X.iloc[i:(i + time_steps)].values
      Xs.append(v)
      ys.append(y.iloc[i + time_steps])
    return np.array(Xs), np.array(ys)

  TIME_STEPS= 30

  X_train, y_train = create_dataset(train[['temp']], train.temp, TIME_STEPS)
  X_test, y_test = create_dataset(test[['temp']], test.temp, TIME_STEPS)

  reconstructed_model=keras.models.load_model("modelDataTemp")

  history=reconstructed_model.fit(
      X_train, y_train,
      epochs=10,
      batch_size=32,
      validation_split=0.1,
      shuffle=False
  )

  X_train_pred = reconstructed_model.predict(X_train)
  train_mae_loss = np.mean(np.abs(X_train_pred - X_train), axis=1)

  train_mae_loss = np.mean(np.abs(X_train_pred, X_train), axis=1)
  train_mae_loss2 = np.square(np.subtract(X_train_pred, X_train)).mean()

  X_test_pred = reconstructed_model.predict(X_test)
  test_mae_loss = np.mean(np.abs(X_test_pred - X_test), axis=1)

  THRESHOLD = 0.4
  test_score_df = pd.DataFrame(index=test[TIME_STEPS:].index)
  test_score_df['loss'] = test_mae_loss
  test_score_df['threshold'] = THRESHOLD
  test_score_df['anomaly'] = test_score_df.loss > test_score_df.threshold
  test_score_df['temp'] = test[TIME_STEPS:].temp

  anomalies = test_score_df[test_score_df.anomaly == True]
  st.write("""### Checking Time-Series Data""")
  fig = plt.figure()
  plt.plot(df_Data.temp, label='temp');
  st.pyplot(fig)

  st.write("""### Value Loss for Model Train and Test """)
  fig2 = plt.figure()
  plt.plot(history.history['loss'], label='train')
  plt.plot(history.history['val_loss'], label='test')
  plt.legend();
  st.pyplot(fig2)

  st.write("""### Data train MAE """)
  fig3 = plt.figure()
  sns.distplot(train_mae_loss, bins=50, kde=True)
  st.pyplot(fig3)

  st.write("""### Data score and threshold """)
  fig4 = plt.figure()
  plt.plot(test_score_df.index, test_score_df.loss, label='loss')
  plt.plot(test_score_df.index, test_score_df.threshold, label='threshold')
  plt.xticks(rotation=25)
  plt.legend();
  st.pyplot(fig4)

  st.write("""### Data temp and anomalies """)
  fig5 = plt.figure()

  plt.plot(
    test[TIME_STEPS:].index, 
    scaler.inverse_transform(test[TIME_STEPS:].temp), 
    label='temp'
  );

  sns.scatterplot(
    anomalies.index,
    scaler.inverse_transform(anomalies.temp),
    color=sns.color_palette()[3],
    s=52,
    label='anomaly'
  )
  plt.xticks(rotation=25)
  plt.legend();
  st.pyplot(fig5)
# END OF CODE TEMPERATURE ANOMALY

# START OF CODE RH ANOMALY
with st.beta_expander("RH Anomalies"):
  train_size = int(len(df_Data) * .80)
  test_size = len(df_Data) - train_size

  train, test = df_Data.iloc[0:train_size], df_Data.iloc[train_size:len(df_Data)]

  scaler = StandardScaler()
  scaler = scaler.fit(train[['rh']])

  train['rh'] = scaler.transform(train[['rh']])
  test['rh'] = scaler.transform(test[['rh']]) 

  def create_dataset (X, y, time_steps=1):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
      v = X.iloc[i:(i + time_steps)].values
      Xs.append(v)
      ys.append(y.iloc[i + time_steps])
    return np.array(Xs), np.array(ys)

  TIME_STEPS= 30

  X_train, y_train = create_dataset(train[['rh']], train.rh, TIME_STEPS)
  X_test, y_test = create_dataset(test[['rh']], test.rh, TIME_STEPS)

  reconstructed_model=keras.models.load_model("modelDataRH")

  history=reconstructed_model.fit(
      X_train, y_train,
      epochs=10,
      batch_size=32,
      validation_split=0.1,
      shuffle=False
  )

  X_train_pred = reconstructed_model.predict(X_train)
  train_mae_loss = np.mean(np.abs(X_train_pred - X_train), axis=1)

  train_mae_loss = np.mean(np.abs(X_train_pred, X_train), axis=1)
  train_mae_loss2 = np.square(np.subtract(X_train_pred, X_train)).mean()

  X_test_pred = reconstructed_model.predict(X_test)
  test_mae_loss = np.mean(np.abs(X_test_pred - X_test), axis=1)

  THRESHOLD = 0.4
  test_score_df = pd.DataFrame(index=test[TIME_STEPS:].index)
  test_score_df['loss'] = test_mae_loss
  test_score_df['threshold'] = THRESHOLD
  test_score_df['anomaly'] = test_score_df.loss > test_score_df.threshold
  test_score_df['rh'] = test[TIME_STEPS:].rh

  anomalies = test_score_df[test_score_df.anomaly == True]
  st.write("""### Checking Time-Series Data""")
  fig6 = plt.figure()
  plt.plot(df_Data.rh, label='rh');
  st.pyplot(fig6)

  st.write("""### Value Loss for Model Train and Test """)
  fig7 = plt.figure()
  plt.plot(history.history['loss'], label='train')
  plt.plot(history.history['val_loss'], label='test')
  plt.legend();
  st.pyplot(fig7)

  st.write("""### Data train MAE """)
  fig8 = plt.figure()
  sns.distplot(train_mae_loss, bins=50, kde=True)
  st.pyplot(fig8)

  st.write("""### Data score and threshold """)
  fig9 = plt.figure()
  plt.plot(test_score_df.index, test_score_df.loss, label='loss')
  plt.plot(test_score_df.index, test_score_df.threshold, label='threshold')
  plt.xticks(rotation=25)
  plt.legend();
  st.pyplot(fig9)

  st.write("""### Data RH and anomalies """)
  fig10 = plt.figure()

  plt.plot(
    test[TIME_STEPS:].index, 
    scaler.inverse_transform(test[TIME_STEPS:].rh), 
    label='rh'
  );

  sns.scatterplot(
    anomalies.index,
    scaler.inverse_transform(anomalies.rh),
    color=sns.color_palette()[3],
    s=52,
    label='anomaly'
  )
  plt.xticks(rotation=25)
  plt.legend();
  st.pyplot(fig10)
# END OF cODE RH ANOMALY


with st.beta_expander("CH Anomalies"):
  train_size = int(len(df_Data) * .80)
  test_size = len(df_Data) - train_size

  train, test = df_Data.iloc[0:train_size], df_Data.iloc[train_size:len(df_Data)]

  scaler = StandardScaler()
  scaler = scaler.fit(train[['ch']])

  train['ch'] = scaler.transform(train[['ch']])
  test['ch'] = scaler.transform(test[['ch']]) 

  def create_dataset (X, y, time_steps=1):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
      v = X.iloc[i:(i + time_steps)].values
      Xs.append(v)
      ys.append(y.iloc[i + time_steps])
    return np.array(Xs), np.array(ys)

  TIME_STEPS= 30

  X_train, y_train = create_dataset(train[['ch']], train.ch, TIME_STEPS)
  X_test, y_test = create_dataset(test[['ch']], test.ch, TIME_STEPS)

  reconstructed_model=keras.models.load_model("modelDataCH")

  history=reconstructed_model.fit(
      X_train, y_train,
      epochs=10,
      batch_size=32,
      validation_split=0.1,
      shuffle=False
  )

  X_train_pred = reconstructed_model.predict(X_train)
  train_mae_loss = np.mean(np.abs(X_train_pred - X_train), axis=1)

  train_mae_loss = np.mean(np.abs(X_train_pred, X_train), axis=1)
  train_mae_loss2 = np.square(np.subtract(X_train_pred, X_train)).mean()

  X_test_pred = reconstructed_model.predict(X_test)
  test_mae_loss = np.mean(np.abs(X_test_pred - X_test), axis=1)

  THRESHOLD = 0.4
  test_score_df = pd.DataFrame(index=test[TIME_STEPS:].index)
  test_score_df['loss'] = test_mae_loss
  test_score_df['threshold'] = THRESHOLD
  test_score_df['anomaly'] = test_score_df.loss > test_score_df.threshold
  test_score_df['ch'] = test[TIME_STEPS:].ch

  anomalies = test_score_df[test_score_df.anomaly == True]
  st.write("""### Checking Time-Series Data""")
  fig11 = plt.figure()
  plt.plot(df_Data.ch, label='ch');
  st.pyplot(fig11)

  st.write("""### Value Loss for Model Train and Test """)
  fig12 = plt.figure()
  plt.plot(history.history['loss'], label='train')
  plt.plot(history.history['val_loss'], label='test')
  plt.legend();
  st.pyplot(fig12)

  st.write("""### Data train MAE """)
  fig13 = plt.figure()
  sns.distplot(train_mae_loss, bins=50, kde=True)
  st.pyplot(fig13)

  st.write("""### Data score and threshold """)
  fig14 = plt.figure()
  plt.plot(test_score_df.index, test_score_df.loss, label='loss')
  plt.plot(test_score_df.index, test_score_df.threshold, label='threshold')
  plt.xticks(rotation=25)
  plt.legend();
  st.pyplot(fig14)

  st.write("""### Data temp and anomalies """)
  fig15 = plt.figure()

  plt.plot(
    test[TIME_STEPS:].index, 
    scaler.inverse_transform(test[TIME_STEPS:].ch), 
    label='ch'
  );

  sns.scatterplot(
    anomalies.index,
    scaler.inverse_transform(anomalies.ch),
    color=sns.color_palette()[3],
    s=52,
    label='anomaly'
  )
  plt.xticks(rotation=25)
  plt.legend();
  st.pyplot(fig15)

with st.beta_expander("Pressure Anomalies"):
  train_size = int(len(df_Data) * .80)
  test_size = len(df_Data) - train_size

  train, test = df_Data.iloc[0:train_size], df_Data.iloc[train_size:len(df_Data)]

  scaler = StandardScaler()
  scaler = scaler.fit(train[['pressure']])

  train['pressure'] = scaler.transform(train[['pressure']])
  test['pressure'] = scaler.transform(test[['pressure']]) 

  def create_dataset (X, y, time_steps=1):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
      v = X.iloc[i:(i + time_steps)].values
      Xs.append(v)
      ys.append(y.iloc[i + time_steps])
    return np.array(Xs), np.array(ys)

  TIME_STEPS= 30

  X_train, y_train = create_dataset(train[['pressure']], train.pressure, TIME_STEPS)
  X_test, y_test = create_dataset(test[['pressure']], test.pressure, TIME_STEPS)

  reconstructed_model=keras.models.load_model("modelDataPressure")

  history=reconstructed_model.fit(
      X_train, y_train,
      epochs=10,
      batch_size=32,
      validation_split=0.1,
      shuffle=False
  )

  X_train_pred = reconstructed_model.predict(X_train)
  train_mae_loss = np.mean(np.abs(X_train_pred - X_train), axis=1)

  train_mae_loss = np.mean(np.abs(X_train_pred, X_train), axis=1)
  train_mae_loss2 = np.square(np.subtract(X_train_pred, X_train)).mean()

  X_test_pred = reconstructed_model.predict(X_test)
  test_mae_loss = np.mean(np.abs(X_test_pred - X_test), axis=1)

  THRESHOLD = 0.05
  test_score_df = pd.DataFrame(index=test[TIME_STEPS:].index)
  test_score_df['loss'] = test_mae_loss
  test_score_df['threshold'] = THRESHOLD
  test_score_df['anomaly'] = test_score_df.loss > test_score_df.threshold
  test_score_df['pressure'] = test[TIME_STEPS:].pressure

  anomalies = test_score_df[test_score_df.anomaly == True]
  st.write("""### Checking Time-Series Data""")
  fig16 = plt.figure()
  plt.plot(df_Data.pressure, label='pressure');
  st.pyplot(fig16)

  st.write("""### Value Loss for Model Train and Test """)
  fig17 = plt.figure()
  plt.plot(history.history['loss'], label='train')
  plt.plot(history.history['val_loss'], label='test')
  plt.legend();
  st.pyplot(fig17)

  st.write("""### Data train MAE """)
  fig18 = plt.figure()
  sns.distplot(train_mae_loss, bins=50, kde=True)
  st.pyplot(fig18)

  st.write("""### Data score and threshold """)
  fig19 = plt.figure()
  plt.plot(test_score_df.index, test_score_df.loss, label='loss')
  plt.plot(test_score_df.index, test_score_df.threshold, label='threshold')
  plt.xticks(rotation=25)
  plt.legend();
  st.pyplot(fig19)

  st.write("""### Data pressure and anomalies """)
  fig20 = plt.figure()

  plt.plot(
    test[TIME_STEPS:].index, 
    scaler.inverse_transform(test[TIME_STEPS:].pressure), 
    label='pressure'
  );

  sns.scatterplot(
    anomalies.index,
    scaler.inverse_transform(anomalies.pressure),
    color=sns.color_palette()[3],
    s=52,
    label='anomaly'
  )
  plt.xticks(rotation=25)
  plt.legend();
  st.pyplot(fig20)

with st.beta_expander("Solrad Anomalies"):
  train_size = int(len(df_Data) * .80)
  test_size = len(df_Data) - train_size

  train, test = df_Data.iloc[0:train_size], df_Data.iloc[train_size:len(df_Data)]

  scaler = StandardScaler()
  scaler = scaler.fit(train[['solrad']])

  train['solrad'] = scaler.transform(train[['solrad']])
  test['solrad'] = scaler.transform(test[['solrad']]) 

  def create_dataset (X, y, time_steps=1):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
      v = X.iloc[i:(i + time_steps)].values
      Xs.append(v)
      ys.append(y.iloc[i + time_steps])
    return np.array(Xs), np.array(ys)

  TIME_STEPS= 30

  X_train, y_train = create_dataset(train[['solrad']], train.solrad, TIME_STEPS)
  X_test, y_test = create_dataset(test[['solrad']], test.solrad, TIME_STEPS)

  reconstructed_model=keras.models.load_model("modelDataSolrad")

  history=reconstructed_model.fit(
      X_train, y_train,
      epochs=10,
      batch_size=32,
      validation_split=0.1,
      shuffle=False
  )

  X_train_pred = reconstructed_model.predict(X_train)
  train_mae_loss = np.mean(np.abs(X_train_pred - X_train), axis=1)

  train_mae_loss = np.mean(np.abs(X_train_pred, X_train), axis=1)
  train_mae_loss2 = np.square(np.subtract(X_train_pred, X_train)).mean()

  X_test_pred = reconstructed_model.predict(X_test)
  test_mae_loss = np.mean(np.abs(X_test_pred - X_test), axis=1)

  THRESHOLD = 0.4
  test_score_df = pd.DataFrame(index=test[TIME_STEPS:].index)
  test_score_df['loss'] = test_mae_loss
  test_score_df['threshold'] = THRESHOLD
  test_score_df['anomaly'] = test_score_df.loss > test_score_df.threshold
  test_score_df['solrad'] = test[TIME_STEPS:].solrad

  anomalies = test_score_df[test_score_df.anomaly == True]
  st.write("""### Checking Time-Series Data""")
  fig21 = plt.figure()
  plt.plot(df_Data.solrad, label='solrad');
  st.pyplot(fig21)

  st.write("""### Value Loss for Model Train and Test """)
  fig22 = plt.figure()
  plt.plot(history.history['loss'], label='train')
  plt.plot(history.history['val_loss'], label='test')
  plt.legend();
  st.pyplot(fig22)

  st.write("""### Data train MAE """)
  fig23 = plt.figure()
  sns.distplot(train_mae_loss, bins=50, kde=True)
  st.pyplot(fig23)

  st.write("""### Data score and threshold """)
  fig24 = plt.figure()
  plt.plot(test_score_df.index, test_score_df.loss, label='loss')
  plt.plot(test_score_df.index, test_score_df.threshold, label='threshold')
  plt.xticks(rotation=25)
  plt.legend();
  st.pyplot(fig24)

  st.write("""### Data solrad and anomalies """)
  fig25 = plt.figure()

  plt.plot(
    test[TIME_STEPS:].index, 
    scaler.inverse_transform(test[TIME_STEPS:].solrad), 
    label='solrad'
  );

  sns.scatterplot(
    anomalies.index,
    scaler.inverse_transform(anomalies.solrad),
    color=sns.color_palette()[3],
    s=52,
    label='anomaly'
  )
  plt.xticks(rotation=25)
  plt.legend();
  st.pyplot(fig25)

with st.beta_expander("WD Anomalies"):
  train_size = int(len(df_Data) * .80)
  test_size = len(df_Data) - train_size

  train, test = df_Data.iloc[0:train_size], df_Data.iloc[train_size:len(df_Data)]

  scaler = StandardScaler()
  scaler = scaler.fit(train[['wd']])

  train['wd'] = scaler.transform(train[['wd']])
  test['wd'] = scaler.transform(test[['wd']]) 

  def create_dataset (X, y, time_steps=1):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
      v = X.iloc[i:(i + time_steps)].values
      Xs.append(v)
      ys.append(y.iloc[i + time_steps])
    return np.array(Xs), np.array(ys)

  TIME_STEPS= 30

  X_train, y_train = create_dataset(train[['wd']], train.wd, TIME_STEPS)
  X_test, y_test = create_dataset(test[['wd']], test.wd, TIME_STEPS)

  reconstructed_model=keras.models.load_model("modelDataWD")

  history=reconstructed_model.fit(
      X_train, y_train,
      epochs=10,
      batch_size=32,
      validation_split=0.1,
      shuffle=False
  )

  X_train_pred = reconstructed_model.predict(X_train)
  train_mae_loss = np.mean(np.abs(X_train_pred - X_train), axis=1)

  train_mae_loss = np.mean(np.abs(X_train_pred, X_train), axis=1)
  train_mae_loss2 = np.square(np.subtract(X_train_pred, X_train)).mean()

  X_test_pred = reconstructed_model.predict(X_test)
  test_mae_loss = np.mean(np.abs(X_test_pred - X_test), axis=1)

  THRESHOLD = 0.4
  test_score_df = pd.DataFrame(index=test[TIME_STEPS:].index)
  test_score_df['loss'] = test_mae_loss
  test_score_df['threshold'] = THRESHOLD
  test_score_df['anomaly'] = test_score_df.loss > test_score_df.threshold
  test_score_df['wd'] = test[TIME_STEPS:].wd

  anomalies = test_score_df[test_score_df.anomaly == True]
  st.write("""### Checking Time-Series Data""")
  fig26 = plt.figure()
  plt.plot(df_Data.wd, label='wd');
  st.pyplot(fig26)

  st.write("""### Value Loss for Model Train and Test """)
  fig27 = plt.figure()
  plt.plot(history.history['loss'], label='train')
  plt.plot(history.history['val_loss'], label='test')
  plt.legend();
  st.pyplot(fig27)

  st.write("""### Data train MAE """)
  fig28 = plt.figure()
  sns.distplot(train_mae_loss, bins=50, kde=True)
  st.pyplot(fig28)

  st.write("""### Data score and threshold """)
  fig29 = plt.figure()
  plt.plot(test_score_df.index, test_score_df.loss, label='loss')
  plt.plot(test_score_df.index, test_score_df.threshold, label='threshold')
  plt.xticks(rotation=25)
  plt.legend();
  st.pyplot(fig29)

  st.write("""### Data solrad and anomalies """)
  fig30 = plt.figure()

  plt.plot(
    test[TIME_STEPS:].index, 
    scaler.inverse_transform(test[TIME_STEPS:].wd), 
    label='WD'
  );

  sns.scatterplot(
    anomalies.index,
    scaler.inverse_transform(anomalies.wd),
    color=sns.color_palette()[3],
    s=52,
    label='anomaly'
  )
  plt.xticks(rotation=25)
  plt.legend();
  st.pyplot(fig30)

with st.beta_expander("WS Anomalies"):
  train_size = int(len(df_Data) * .80)
  test_size = len(df_Data) - train_size

  train, test = df_Data.iloc[0:train_size], df_Data.iloc[train_size:len(df_Data)]

  scaler = StandardScaler()
  scaler = scaler.fit(train[['ws']])

  train['ws'] = scaler.transform(train[['ws']])
  test['ws'] = scaler.transform(test[['ws']]) 

  def create_dataset (X, y, time_steps=1):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
      v = X.iloc[i:(i + time_steps)].values
      Xs.append(v)
      ys.append(y.iloc[i + time_steps])
    return np.array(Xs), np.array(ys)

  TIME_STEPS= 30

  X_train, y_train = create_dataset(train[['ws']], train.ws, TIME_STEPS)
  X_test, y_test = create_dataset(test[['ws']], test.ws, TIME_STEPS)

  reconstructed_model=keras.models.load_model("modelDataWS")

  history=reconstructed_model.fit(
      X_train, y_train,
      epochs=10,
      batch_size=32,
      validation_split=0.1,
      shuffle=False
  )

  X_train_pred = reconstructed_model.predict(X_train)
  train_mae_loss = np.mean(np.abs(X_train_pred - X_train), axis=1)

  train_mae_loss = np.mean(np.abs(X_train_pred, X_train), axis=1)
  train_mae_loss2 = np.square(np.subtract(X_train_pred, X_train)).mean()

  X_test_pred = reconstructed_model.predict(X_test)
  test_mae_loss = np.mean(np.abs(X_test_pred - X_test), axis=1)

  THRESHOLD = 0.4
  test_score_df = pd.DataFrame(index=test[TIME_STEPS:].index)
  test_score_df['loss'] = test_mae_loss
  test_score_df['threshold'] = THRESHOLD
  test_score_df['anomaly'] = test_score_df.loss > test_score_df.threshold
  test_score_df['ws'] = test[TIME_STEPS:].ws

  anomalies = test_score_df[test_score_df.anomaly == True]
  st.write("""### Checking Time-Series Data""")
  fig31 = plt.figure()
  plt.plot(df_Data.ws, label='ws');
  st.pyplot(fig31)

  st.write("""### Value Loss for Model Train and Test """)
  fig32 = plt.figure()
  plt.plot(history.history['loss'], label='train')
  plt.plot(history.history['val_loss'], label='test')
  plt.legend();
  st.pyplot(fig32)

  st.write("""### Data train MAE """)
  fig33 = plt.figure()
  sns.distplot(train_mae_loss, bins=50, kde=True)
  st.pyplot(fig33)

  st.write("""### Data score and threshold """)
  fig34 = plt.figure()
  plt.plot(test_score_df.index, test_score_df.loss, label='loss')
  plt.plot(test_score_df.index, test_score_df.threshold, label='threshold')
  plt.xticks(rotation=25)
  plt.legend();
  st.pyplot(fig34)

  st.write("""### Data ws and anomalies """)
  fig35 = plt.figure()

  plt.plot(
    test[TIME_STEPS:].index, 
    scaler.inverse_transform(test[TIME_STEPS:].ws), 
    label='WS'
  );

  sns.scatterplot(
    anomalies.index,
    scaler.inverse_transform(anomalies.ws),
    color=sns.color_palette()[3],
    s=52,
    label='anomaly'
  )
  plt.xticks(rotation=25)
  plt.legend();
  st.pyplot(fig25)


# # Displays the user input features
# st.subheader('User Input features')

# if uploaded_file is not None:
#     st.write(df)
# else:
#     st.write('Awaiting CSV file to be uploaded. Currently using example input parameters (shown below).')
#     st.write(df)

# Reads in saved classification model
# load_clf = pickle.load(open('penguins_clf.pkl', 'rb'))

# # Apply model to make predictions
# prediction = load_clf.predict(df)
# prediction_proba = load_clf.predict_proba(df)

# st.subheader('Prediction')
# penguins_species = np.array(['Adelie','Chinstrap','Gentoo'])
# st.write(penguins_species[prediction])

# st.subheader('Prediction Probability')
# st.write(prediction_proba)