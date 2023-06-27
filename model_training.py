import pandas as pd
import numpy as np
import re
from datetime import datetime
import pickle
api_key='****'

def clean_data (data):
    from dateutil.relativedelta import relativedelta
    import json
    import requests
    def make_float(data,column):
        data[column]=data[column].apply(lambda x:float(re.findall(r'[\d.]+', str(x).replace('₪', '').replace(',', ''))[0])if re.findall(r'\d+', str(x)) else np.nan)
        return data

    def fix_text(data,column):
        data[column]=data[column].apply(lambda x: re.sub(r'[^\w\s\d]', '', x)if isinstance(x,str) and x.strip()!='None' else np.nan)
        return data

    def make_bool(data,column):
        data[column]=data[column].apply(lambda x: 1 if isinstance(x, str) and x.split()[0] in ['יש','כן','yes'] else 1 if isinstance(x, bool) and x==True else 0)
        return data
    def get_dis_Eilat(city):
        url_dis="https://maps.googleapis.com/maps/api/distancematrix/json?origins=%s&destinations=%s&key=%s" % ('Eilat',city,api_key)
        try:
            response_dis = requests.get(url_dis)
            if response_dis.status_code != 200:
                return np.nan
            else:
                try:
                    res_dis = response_dis.json()
                    return res_dis['rows'][0]['elements'][0]['distance']['value']/1000
                except:
                    return np.nan
        except Exception as e:
            return np.nan

    if 'price' in data:
        data=make_float(data,'price')
    data=make_float(data,'Area')
    data=make_float(data,'room_number')
    if 'publishedDays' in data:
        data=make_float(data,'publishedDays')
    ratio_area_rooms = data['Area'] / data['room_number']
    mean_ratio=ratio_area_rooms.mean()
    data['Area'].fillna(data['room_number'] * mean_ratio, inplace=True)
    data['room_number'].fillna(data['Area'] / mean_ratio, inplace=True)
    data=data=make_float(data,'number_in_street')
    data=fix_text(data,'description')
    data=fix_text(data,'city_area')
    data=fix_text(data,'Street')
    data=fix_text(data,'city_area')
    data['City']=data['City'].apply(lambda x: 'נהריה' if x.strip()=='נהרייה' else x.strip())
    data['condition']=data['condition'].apply(lambda x: 'לא צויין' if x =='None' or pd.isna(x) or x==False else x)
    data['floor_out_of']=data['floor_out_of'].apply(lambda x: x.split() if isinstance(x,str) and x.strip()!='None' else np.nan)
    data['floor']=data['floor_out_of'].apply(lambda x: np.nan if not isinstance(x,list) else 0 if x[1]=='קרקע' else -1 if x[1]=='מרתף' else int(x[1]))
    data['total_floors']=data['floor_out_of'].apply(lambda x:  np.nan if not isinstance(x,list) else 0 if x[1] in ['קרקע','מרתף'] and len(x)==2 else int(x[1]) if len(x)==2 else int(x[3]))
    data['entranceDate']=data['entranceDate'].apply(lambda x: relativedelta(x.date(),datetime.today().date()).months if isinstance(x,datetime) else x)
    data['entranceDate'] = data['entranceDate'].map(
        lambda x: 'less_than_6months' if x == 'מיידי' or isinstance(x, int) and x < 6
        else 'months_6_12' if isinstance(x, int) and 6 <= x < 12
        else 'above_year' if isinstance(x, int) and x >= 12 
        else 'flexible' if x == 'גמיש'
        else 'not_defined')
    data=make_bool(data,'hasElevator')
    data=make_bool(data,'hasParking')
    data=make_bool(data,'hasBars')
    data=make_bool(data,'hasStorage')
    data=make_bool(data,'hasAirCondition')
    data=make_bool(data,'hasBalcony')
    data=make_bool(data,'hasMamad')
    data=make_bool(data,'handicapFriendly')
    data=data.drop(['index','floor_out_of'],axis=1)
    data['view']=data['description'].apply(lambda x: 1 if re.search('נוף', str(x)) else 0)
    if api_key != '****':
        data['dis_from_Eilat']=data['City'].apply(lambda x: get_dis_Eilat(x))
    return data

def model (data):
    from sklearn.model_selection import train_test_split,cross_val_score
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    from sklearn.linear_model import ElasticNet
    from sklearn.compose import ColumnTransformer
    import numpy as np
    from sklearn.metrics import mean_squared_error

    X=data.drop(['price','description'],axis=1)
    y=data['price']
    column_names = X.select_dtypes(exclude=['int']).columns.tolist()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2,random_state = 50)
    column_transformer = ColumnTransformer(
        transformers=[
            ('standanisation',StandardScaler(),['Area']),
            ('one_hot_encoder', OneHotEncoder(handle_unknown='ignore'), column_names)],
            remainder='passthrough')
    pipeline = Pipeline([('preprocessor', column_transformer),('elastic_net', ElasticNet(alpha=0.05, l1_ratio=0.9))])
    scores = cross_val_score(pipeline, X_train, y_train, cv=10, scoring='neg_mean_squared_error')
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    cv_rmse = np.sqrt(np.mean(np.abs(scores)))
    cv_std = np.std(scores)
    test_score = pipeline.score(X_test, y_test)
    mse = mean_squared_error(y_test, y_pred)
    results = {
        'CV_RMSE': cv_rmse,
        'CV_StdDev': cv_std,
        'Test_Score': test_score,
        'MSE': np.sqrt(mse)}
    pickle.dump(pipeline, open("trained_model.pkl","wb"))
    return results

#data = pd.read_excel('C:/Users/yratz/House_sell/output_all_students_Train_v10.xlsx')
#data = data.dropna(subset=['price'])
#data=data.drop(data[data['price'].apply(lambda x: isinstance(x, str) and not bool(re.search(r'\d', x)))].index).reset_index()
#data.columns = data.columns.str.rstrip()
#data=clean_data(data)
#print(model(data))


