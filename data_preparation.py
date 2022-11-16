import json
import random
import pandas as pd
import numpy as np
from ast import literal_eval

def read_data(filename):
    with open(filename, encoding='UTF-8') as json_data:
        data = [json.loads(line) for line in json_data]
    df = pd.json_normalize(data)
    return df

def export_data(df, filename):
    df.to_csv(filename, index=False)

def dict_to_columns(df, name, fill):
    df[name] = df[name].fillna(str(fill))
    df[name] = df[name].replace(to_replace = 'None', value = str(fill))
    df[name] = df[name].apply(literal_eval)
    df = df.join(pd.DataFrame(df[name].to_dict()).T)
    df = df.drop([name], axis = 1)
    return df

def get_value():
    prob = random.randrange(100)
    if prob < 50:
        return True
    else:
        return False

def business_data(df):


    df = df[df['categories'].str.contains('Restaurant|Food', na = False)]
    df = df[df['state'] == 'FL']
    df = df[df['review_count'] >= 10]
    

    to_drop = ['address', 'city', 'state', 'postal_code', 'latitude', 'longitude', 'is_open', 'hours.Monday',
               'hours.Tuesday', 'hours.Wednesday', 'hours.Thursday', 'hours.Friday', 'hours.Saturday', 'hours.Sunday',
               'hours', 'attributes', 'attributes.HairSpecializesIn', 'attributes.BYOB', 'attributes.CoatCheck',
               'attributes.BYOBCorkage', 'attributes.Corkage', 'attributes.AgesAllowed', 'attributes.ByAppointmentOnly',
               'attributes.AcceptsInsurance', 'attributes.GoodForDancing', 'attributes.Smoking', 'attributes.DriveThru',
               'attributes.RestaurantsCounterService', 'attributes.DietaryRestrictions', 'attributes.Open24Hours',
               'attributes.Music']
    
    df = df.drop(to_drop, axis = 1)


    for feature in df.columns.values.tolist()[5:]:
        df = df.rename(columns = {feature: feature[11:].lower()})    


    df['wifi'] = np.where(df['wifi'].str.contains('paid|free'), True, df['wifi'])    
    df['wifi'] = np.where(df['wifi'].str.contains('no'), False, df['wifi'])
    
    df['businessparking'] = np.where(df['businessparking'].str.contains('True'), True, df['businessparking'])
    df['businessparking'] = np.where(df['businessparking'].str.contains('False'), False, df['businessparking'])

    df['alcohol'] = np.where(df['alcohol'].str.contains('bar|beer'), True, df['alcohol'])
    df['alcohol'] = np.where(df['alcohol'].str.contains('none'), False, df['alcohol'])


    
    update_nan = {'nan': np.nan, None: np.nan}


    df['restaurantspricerange2'] = df['restaurantspricerange2'].map(update_nan).fillna(df['restaurantspricerange2'])
    df['restaurantspricerange2'] = df['restaurantspricerange2'].apply(lambda x: random.choice(['1','2','3','4']) if x is np.nan else x)
    df['price1'] = np.where(df['restaurantspricerange2'].str.contains('1'), True, False)
    df['price2'] = np.where(df['restaurantspricerange2'].str.contains('2'), True, False)
    df['price3'] = np.where(df['restaurantspricerange2'].str.contains('3'), True, False)
    df['price4'] = np.where(df['restaurantspricerange2'].str.contains('4'), True, False)
    df = df.drop(['restaurantspricerange2'], axis = 1)


    df['noiselevel'] = df['noiselevel'].map(update_nan).fillna(df['noiselevel'])
    df['noiselevel'] = df['noiselevel'].apply(lambda x: random.choice(['quiet', 'average', 'loud', 'very_loud']) if x is np.nan else x)
    df['noisevloud'] = np.where(df['noiselevel'].str.contains('very_loud'), True, False)
    df['nosieloud'] = np.where(df['noiselevel'].str.contains('loud'), True, False)
    df['noiseaverage'] = np.where(df['noiselevel'].str.contains('average'), True, False)
    df['noisequiet'] = np.where(df['noiselevel'].str.contains('quiet'), True, False)
    df = df.drop(['noiselevel'], axis = 1)


    df['restaurantsattire'] = df['restaurantsattire'].map(update_nan).fillna(df['restaurantsattire'])
    df['restaurantsattire'] = df['restaurantsattire'].apply(lambda x: random.choice(['casual', 'dressy', 'formal']) if x is np.nan else x)    
    df['attirecasual'] = np.where(df['restaurantsattire'].str.contains('casual'), True, False)
    df['attiredressy'] = np.where(df['restaurantsattire'].str.contains('dressy'), True, False)
    df['attireformal'] = np.where(df['restaurantsattire'].str.contains('formal'), True, False)
    df = df.drop(['restaurantsattire'], axis = 1)
    
    

    ambience_fill = {
        'touristy': None,
        'hipster': None,
        'romantic': None,
        'divey': None,
        'intimate': None,
        'trendy': None,
        'upscale': None,
        'classy': None,
        'casual': None
        }

    meal_fill = {
        'dessert': None,
        'latenight': None,
        'lunch': None,
        'dinner': None,
        'brunch': None,
        'breakfast': None
        }

    nights_fill = {
        'monday': None,
        'tuesday': None,
        'wednesday': None,
        'thursday': None,
        'friday': None,
        'saturday': None,
        'sunday': None
        }

    df = dict_to_columns(df, 'ambience', ambience_fill)
    df = dict_to_columns(df, 'goodformeal', meal_fill)
    df = dict_to_columns(df, 'bestnights', nights_fill)
        

    features = ['restaurantstableservice', 'wifi', 'bikeparking', 'businessparking', 'businessacceptscreditcards',
                'restaurantsreservations', 'wheelchairaccessible', 'caters', 'outdoorseating', 'restaurantsgoodforgroups',
                'happyhour', 'businessacceptsbitcoin', 'hastv', 'alcohol', 'dogsallowed', 'restaurantstakeout',
                'restaurantsdelivery', 'goodforkids', 'touristy', 'hipster', 'romantic', 'divey', 'intimate', 'trendy',
                'upscale', 'classy', 'casual', 'dessert', 'latenight', 'lunch', 'dinner', 'brunch', 'breakfast',
                'monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']

    
    for feature in features:

        bool_types = {'nan': np.nan, None: np.nan, 'False': False, 'True': True}
        df[feature] = df[feature].map(bool_types)
        df[feature] = df[feature].apply(lambda x: get_value() if x is np.nan else x)


    unique = {}
    df = df.reset_index(drop = True)
    for i in range(len(df)):
        unique_2 = set(df['categories'][i].split(', '))
        for j in unique_2:
            if j != 'Restaurants' and j != 'Food':
                if j not in unique:
                    unique[j] = 1
                else:
                    unique[j] += 1
                    
    unique_cat = [key for key, value in unique.items() if value >= 10]

    temp = pd.Series(df['categories']).str.get_dummies(',')
    temp = temp.filter(unique_cat)

    new_df = pd.concat([df[df.columns[:2]], df[df.columns[3]], temp, df[df.columns[5:]], df[df.columns[2]]], axis = 1)
    new_df = new_df.replace({False: 0, True: 1})

    print(new_df.dtypes)
                       
    return new_df

def user_data(df):
    df = df.filter(['user_id', 'name', 'review_count', 'average_stars'])
    df = df[df['review_count'] >= 10]
    return df
    

def review_data(df, df2, df3):
    df = df.filter(['review_id', 'user_id', 'business_id', 'stars'])
    df = df[df.business_id.isin(df2['business_id'].unique().tolist()) & df.user_id.isin(df3['user_id'].unique().tolist())]
    df3 = df3[df3.user_id.isin(df['user_id'].unique().tolist())]
    df2 = df2[df2.business_id.isin(df['business_id'].unique().tolist())]
    return df, df2, df3

def update_ids(df, df2, df3):
    
    business_list = df2['business_id'].unique().tolist()
    business_dict = {}
    for i in range(1, len(business_list) + 1):
        business_dict[business_list[i - 1]] = i
    df2['business_id'] = df2['business_id'].map(business_dict)

    user_list = df3['user_id'].unique().tolist()
    user_dict = {}
    for i in range(1, len(user_list) + 1):
        user_dict[user_list[i - 1]] = i
    df3['user_id'] = df3['user_id'].map(user_dict)

    df['business_id'] = df['business_id'].map(business_dict)
    df['user_id'] = df['user_id'].map(user_dict)

    for i in range(1, len(df) + 1):
        df.at[i - 1, 'review_id'] = i
    
    return df, df2, df3

def update_review_stats(df, df2, df3):
    
    business_counts = df['business_id'].value_counts()
    user_counts = df['user_id'].value_counts()
    
    for i in range(1, len(df2) + 1):
        df2.at[i - 1, 'review_count'] = business_counts[i]
    for i in range(1, len(df3) + 1):
        df3.at[i - 1, 'review_count'] = user_counts[i]

    business_stars = df['stars'].groupby(df['business_id']).sum()
    user_stars = df['stars'].groupby(df['user_id']).sum()
    
    for i in range(1, len(df2) + 1):
        df2.at[i - 1, 'stars'] = round(business_stars[i] / business_counts[i], 2)
    for i in range(1, len(df3) + 1):
        df3.at[i - 1, 'average_stars'] = round(user_stars[i] / user_counts[i], 2)
    
    return df, df2, df3


# Code used to conduct the data preparation
'''
#restaurants = read_data('yelp_academic_dataset_business.json')
#users = read_data('yelp_academic_dataset_user.json')
#reviews = read_data('yelp_academic_dataset_review.json')


#restaurants = business_data(restaurants)
#users = user_data(users)
#reviews, restaurants, users = review_data(reviews, restaurants, users)
#reviews, restaurants, users = update_ids(reviews, restaurants, users)
#reviews, restaurants, users = update_review_stats(reviews, restaurants, users)

#export_data(restaurants, 'restaurants.csv')
#export_data(users, 'users.csv')
#export_data(reviews, 'reviews.csv')
'''
