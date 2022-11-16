import pandas as pd
import numpy as np
import math
import datetime
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error 

restaurants = pd.read_csv('restaurants.csv')
users = pd.read_csv('users.csv')
reviews = pd.read_csv('reviews.csv')

def cosine_sim(vectors, n):
    cos_sim = {}
    temp = []
    for i in range(len(vectors)):
        dot = np.dot(vectors[i],vectors[n])
        mag = np.linalg.norm(vectors[i]) * np.linalg.norm(vectors[n])
        similarity = dot / mag
        temp.append(similarity)
    cos_sim[n] = temp
    return cos_sim

def cos_sim_mul(vectors):
    averages = []
    for i in range(len(vectors)):
        temp = []
        for j in range(len(vectors)):
            if i != j:
                dot = np.dot(vectors[i],vectors[j])
                mag = np.linalg.norm(vectors[i]) * np.linalg.norm(vectors[j])
                similarity = dot / mag
                temp.append(similarity)
        averages.append(temp)
    return averages

def get_vectors(df):
    vectors = df.values.tolist()
    return vectors

def get_day():
    mapping = {0: 'monday', 1: 'tuesday', 2: 'wednesday', 3: 'thursday', 4: 'friday', 5: 'saturday', 6: 'sunday', 7: 'sunday'}
    day = datetime.datetime.today().weekday()
    day = mapping[day]
    return day

def recommend(df, lst, n):
    day = get_day()
    for i in range(len(lst)):
        if df[day][i] == False:
            lst[i] *= 0.9
    lst[n - 1] = 0
    idx = sorted(range(len(lst)), key=lambda i: lst[i], reverse=True)[:n]
    recommendations = []
    for i in idx:
        rec = restaurants['name'][i]
        recommendations.append(rec)
    return recommendations

def get_user_top(df, user_id):
    top = df.sort_values(['business_id', 'stars'],ascending=False).groupby(['user_id']).first().reset_index()
    index = top['business_id'][user_id]
    return index
    
def runCB(df, df2, user_id, x):
    bus_id = get_user_top(df2, user_id - 1)
    df = df[df.columns[3:-1]]
    vectors = get_vectors(df)
    cosine = cosine_sim(vectors, bus_id - 1)
    similarities = cosine[bus_id - 1]
    recommendations = recommend(df, similarities, x)
    print('\nHere are your recommendations:\n')
    for i in recommendations:
        print(i)

def evaluate_RMSE(df):
    y = df.stars   
    x = df[df.columns[3:-1]]
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42)
    knn = KNeighborsRegressor(n_neighbors=5)
    knn.fit(X_train, y_train)
    predictions = knn.predict(X_train)
    MSE = mean_squared_error(y_train, predictions)
    RMSE = math.sqrt(MSE)
    print(RMSE)
    
    
def evaluate_diversity(df, df2):
    averages = []
    user_ids = df['user_id'].tolist()
    random = np.random.choice(user_ids, size = 10)
    df2 = df2[df2.columns[3:-1]]
    vectors = get_vectors(df2)
    averages = []
    for i in random:
        temp = get_user_top(df, i - 1)
        cosine = cosine_sim(vectors, temp - 1)
        sim = cosine[temp - 1]
        idx = sorted(range(len(sim)), key=lambda i: sim[i], reverse=True)[1:11]
        lst = []
        for i in idx:
            lst.append(vectors[i])
        averages_temp = cos_sim_mul(lst)
        for i in range(len(averages_temp)):
            averages.append(sum(averages_temp[i]) / len(averages_temp[i]))
    print(sum(averages)/len(averages))
