import numpy as np
import pandas as pd
from color_difference import deltaE_94
from skimage.color import deltaE_cie76, deltaE_ciede2000
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error, mean_absolute_error, median_absolute_error, r2_score
from sklearn.model_selection import train_test_split
np.random.seed(10)

filepath="C:/Users/hj123/Desktop/ACA_2019/Test_03.xlsx"
Data_ = pd.read_excel(filepath)
#print (Data_[:2])

x_OneHot_df = pd.get_dummies(data = Data_, columns = ['XYZw', 'Surround'])
cols = ['X', 'Y', 'Z', 'La', 'Yb', 'XYZw_A', 'XYZw_D50', 'XYZw_D65', 'Surround_avg', 'Surround_dark', 'Surround_dim', 'J', 'a', 'b']
all_df = x_OneHot_df[cols]

#x_OneHot_df = pd.get_dummies(data = Data_, columns = ['Surround'])
#cols = ['X', 'Y', 'Z', 'X_w', 'Y_w', 'Z_w', 'La', 'Yb', 'Surround_avg', 'Surround_dark', 'Surround_dim', 'J', 'a', 'b']
#all_df = x_OneHot_df[cols]

ndarray = all_df.values
print(ndarray.shape)

X = ndarray[::,0:11]
y = ndarray[::,11:]

minmax_scale = preprocessing.MinMaxScaler(feature_range=(0,1))
scaledFeatures = minmax_scale.fit_transform(X)

x_train, x_test, y_train, y_test = train_test_split(scaledFeatures, y, test_size=0.3)

train_p = x_train
train_d = y_train

test_p = x_test
test_d = y_test

print('total:',len(Data_),
      'train:',len(train_p),
      'test:',len(test_p))

#from keras import optimizers
from keras.models import Sequential
from keras.layers import Dense,Dropout,Activation
#from keras.layers.normalization import BatchNormalization
#from keras.layers.advanced_activations import LeakyReLU

model = Sequential()
model.add(Dense(units=30, input_dim=11, kernel_initializer='uniform'))
#model.add(BatchNormalization())
model.add(Activation('relu'))
#model.add(LeakyReLU(alpha=0.2))
#model.add(Dropout(0.25))

model.add(Dense(units=30, kernel_initializer='uniform'))
#model.add(BatchNormalization())
model.add(Activation('relu'))
#model.add(LeakyReLU(alpha=0.2))
#model.add(Dropout(0.25))

model.add(Dense(units=30, kernel_initializer='uniform'))
#model.add(BatchNormalization())
model.add(Activation('relu'))
#model.add(LeakyReLU(alpha=0.2))
#model.add(Dropout(0.25))

model.add(Dense(units=3, kernel_initializer='uniform'))
model.add(Activation('linear'))

print(model.summary())


#/////Optimizers/////
#rmsprop = optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0)
#adagrad = optimizers.Adagrad(lr=0.01, epsilon=None, decay=0.0)
#adamax = optimizers.Adamax(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0)
#adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
#nadam = optimizers.Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004)
model.compile(loss='mse', 
              optimizer= 'adam', metrics=['mae'])

train_history =model.fit(x=train_p, 
                         y=train_d, 
                         validation_split=0.1, 
                         epochs=1500, 
                         batch_size=128,verbose=2)


import matplotlib.pyplot as plt
import matplotlib.lines as mlines 


def show_train_history(train_history,train,validation):
    plt.plot(train_history.history[train], color='green', linestyle='solid')
    plt.plot(train_history.history[validation], color='blue', linestyle='solid')
    plt.title('Train History')
    plt.ylabel(train)
    plt.xlabel('Epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()

show_train_history(train_history,'mean_absolute_error','val_mean_absolute_error')
show_train_history(train_history,'loss','val_loss')
Test_predict = model.predict(test_p)
scores = model.evaluate(x=test_p, 
                        y=test_d)
print(scores[1])


#//////////////////////////////////////////////////////////////////////////////
#Root-mean-square error
rmse_J = np.sqrt(mean_squared_error(y_test[:,0], Test_predict[:,0]))
rmse_a = np.sqrt(mean_squared_error(y_test[:,1], Test_predict[:,1]))
rmse_b = np.sqrt(mean_squared_error(y_test[:,2], Test_predict[:,2]))

#Coefficient of determination
r2score_J = r2_score(y_test[:,0], Test_predict[:,0])
r2score_a = r2_score(y_test[:,1], Test_predict[:,1])
r2score_b = r2_score(y_test[:,2], Test_predict[:,2])

#Mean absolute error                   
mae_J = mean_absolute_error(y_test[:,0], Test_predict[:,0])
mae_a = mean_absolute_error(y_test[:,1], Test_predict[:,1])
mae_b = mean_absolute_error(y_test[:,2], Test_predict[:,2])


max_ae_J = np.max(np.abs(y_test[:,0] - Test_predict[:,0]))
max_ae_a = np.max(np.abs(y_test[:,1] - Test_predict[:,1]))
max_ae_b = np.max(np.abs(y_test[:,2] - Test_predict[:,2]))

#Pearson R
correlation_J = np.corrcoef(Test_predict[:,0],y_test[:,0])[0,1]
correlation_a = np.corrcoef(Test_predict[:,1],y_test[:,1])[0,1]
correlation_b = np.corrcoef(Test_predict[:,2],y_test[:,2])[0,1]

#Color difference
D_76 = deltaE_cie76(y_test, Test_predict)
D_94 = deltaE_94(y_test, Test_predict)
D_2000 = deltaE_ciede2000(y_test, Test_predict)

Mean_D76 = np.mean(D_76)
Mean_D94 = np.mean(D_94)
Mean_D2000 = np.mean(D_2000)

Max_D76 = np.max(D_76)
Max_D94 = np.max(D_94)
Max_D2000 = np.max(D_2000)

#//////////////////////////////////////////////////////////////////////////////
#Figure plots
fig, ax = plt.subplots() 
ax.scatter(y_test[:,0], Test_predict[:,0], marker='o',c='',edgecolors='r') 
line = mlines.Line2D([0, 1], [0, 1], color='black') 
transform = ax.transAxes 
line.set_transform(transform) 
ax.add_line(line)
plt.xlabel('Lightness J ')
plt.ylabel('Predicted Lightness J ')
plt.title('RMSE = %.2f, MAE = %.2f, r2_score = %.2f' %(rmse_J, mae_J, r2score_J)) 
plt.show() 

fig, ax = plt.subplots() 
ax.scatter(y_test[:,1], Test_predict[:,1], marker='o',c='',edgecolors='r') 
line = mlines.Line2D([0, 1], [0, 1], color='black') 
transform = ax.transAxes 
line.set_transform(transform) 
ax.add_line(line)
plt.xlabel('green–red component a')
plt.ylabel('Predicted green–red component a')
plt.title('RMSE = %.2f, MAE = %.2f, r2_score = %.2f' %(rmse_a, mae_a, r2score_a)) 
plt.show() 

fig, ax = plt.subplots() 
ax.scatter(y_test[:,2], Test_predict[:,2], marker='o',c='',edgecolors='r') 
line = mlines.Line2D([0, 1], [0, 1], color='black') 
transform = ax.transAxes 
line.set_transform(transform) 
ax.add_line(line)
plt.xlabel('blue–yellow component b')
plt.ylabel('Predicted blue–yellow component b')
plt.title('RMSE = %.2f, MAE = %.2f, r2_score = %.2f' %(rmse_b, mae_b, r2score_b)) 
plt.show() 