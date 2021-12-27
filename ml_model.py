import numpy as np
import pandas as pd
import joblib 
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


df = pd.read_csv(r"data\clean_banglore_price.csv")

x = df.drop('price',axis=1)
y = df['price']
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=51)

l = list(x.columns)
loc_list = l[9:]
new_loc_list = []
for val in loc_list:
  new_list = val.split("_")
  new_loc_list.append(new_list[-1])
loc_list = new_loc_list.copy()

sc = StandardScaler()
sc.fit(x_train)
X_train = sc.transform(x_train)
X_test = sc.transform(x_test)

ml_model = joblib.load(r'banglore_house_price_rfr.pkl')

def predict_house_price(bath,balcony,total_sqft_int,bhk,area_type,availability,location):

  arr = np.zeros(len(x.columns))

  arr[0] = bath
  arr[1] = balcony
  arr[2] = total_sqft_int
  arr[3] = bhk
  
  if area_type + " Area" in x.columns:
    index = np.where(x.columns == area_type + " Area")[0][0]
    arr[index] = 1

  if "availability"=="available":
    arr[7]=1

  if 'loc_'+location in x.columns:
    loc_index = np.where(x.columns=="loc_"+location)[0][0]
    arr[loc_index] =1

  arr = sc.transform([arr])[0]
  return ml_model.predict([arr])[0]

#print(loc_list)
#print(len(loc_list))