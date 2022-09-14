import os
import pandas as pd



path_X_train = os.path.join("data\\train_test_para_modelo_definitivo_con_outliers", "X_train.csv")
path_y_train = os.path.join("data\\train_test_para_modelo_definitivo_con_outliers", "y_train.csv")

path_X_test = os.path.join("data\\train_test_para_modelo_definitivo_con_outliers", "X_test.csv")
path_y_test = os.path.join("data\\train_test_para_modelo_definitivo_con_outliers", "y_test.csv")


X_train = pd.read_csv(path_X_train).drop('Unnamed: 0', axis=1)
y_train = pd.read_csv(path_y_train).drop('Unnamed: 0', axis=1)

X_test = pd.read_csv(path_X_test).drop('Unnamed: 0', axis=1)
y_test = pd.read_csv(path_y_test).drop('Unnamed: 0', axis=1)



#path_my_model= os.path.join('model', 'my_model.pkl')
#object = pd.read_pickle(path_my_model)

from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(random_state=7)

model.fit(X_train, y_train)

model.predict(X_test)