# import libraries
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
import warnings
warnings.filterwarnings('ignore')

# Data gathering
df=pd.read_csv('media prediction and its cost.csv')

# Encoding
lbl_enco=LabelEncoder()
df['food_category']=lbl_enco.fit_transform(df['food_category'])
df['food_department']=lbl_enco.fit_transform(df['food_department'])
df['food_family']=lbl_enco.fit_transform(df['food_family'])
df['promotion_name']=lbl_enco.fit_transform(df['promotion_name'])
df['sales_country']=lbl_enco.fit_transform(df['sales_country'])
df['marital_status']=lbl_enco.fit_transform(df['marital_status'])
df['gender']=lbl_enco.fit_transform(df['gender'])
df['education']=lbl_enco.fit_transform(df['education'])
df['member_card']=lbl_enco.fit_transform(df['member_card'])
df['occupation']=lbl_enco.fit_transform(df['occupation'])
df['houseowner']=lbl_enco.fit_transform(df['houseowner'])
df['avg. yearly_income']=lbl_enco.fit_transform(df['avg. yearly_income'])
df['brand_name']=lbl_enco.fit_transform(df['brand_name'])
df['store_type']=lbl_enco.fit_transform(df['store_type'])
df['store_city']=lbl_enco.fit_transform(df['store_city'])
df['store_state']=lbl_enco.fit_transform(df['store_state'])
df['media_type']=lbl_enco.fit_transform(df['media_type'])

# After feature selection 
l1=['promotion_name', 'store_type', 'store_city', 'store_sqft',
       'frozen_sqft', 'meat_sqft', 'coffee_bar', 'salad_bar', 'prepared_food',
       'florist', 'media_type']

# Data splitiing
x = df[l1]
y = df['cost']
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size= 0.2, stratify=y ,random_state = 2 )

# Model building and training
rf_reg = RandomForestRegressor(max_depth=14, max_features='sqrt', min_samples_leaf=7,
                      min_samples_split=17, n_estimators=50, random_state=30)
rf_reg.fit(x_train, y_train)


# Testing Data Evaluation
y_pred = rf_reg.predict(x_test)
mse = mean_squared_error(y_test, y_pred)
print('MSE :', mse)
rmse = np.sqrt(mse)
print('RMSE :', rmse)
mae = mean_absolute_error(y_test, y_pred)
print('MAE :', mae)
accuracy = r2_score(y_test, y_pred)
print('R2_Score :', accuracy)

# Training Data Evaluation
y_pred_train = rf_reg.predict(x_train)
mse = mean_squared_error(y_train, y_pred_train)
print('MSE :', mse)
rmse = np.sqrt(mse)
print('RMSE :', rmse)
mae = mean_absolute_error(y_train, y_pred_train)
print('MAE :', mae)
accuracy = r2_score(y_train, y_pred_train)
print('R2_Score :', accuracy)

# Pickle dump
with open('cost_rf.pkl', 'wb') as file:
    pickle.dump(rf_reg, file)