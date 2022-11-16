import pandas as pd
import numpy as np
from scipy.sparse.linalg import svds

restaurants = pd.read_csv('restaurants.csv')
users = pd.read_csv('users.csv')
reviews = pd.read_csv('reviews.csv')
reviews = reviews.groupby('user_id').filter(lambda x : len(x)>10)

def get_user_top(df, user_id):
    top = df.sort_values(['business_id', 'stars'],ascending=False).groupby(['user_id']).first().reset_index()
    index = top['business_id'][user_id]
    return index

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

def train(df):
    ratings = reviews.pivot_table(index="user_id", columns="business_id", values="stars").fillna(0)
    u, s, v = svds(ratings, k = 20)
    s = np.diag(s)
    pred = np.dot(np.dot(u, s), v) 
    predicted = pd.DataFrame(pred, columns = ratings.columns)
    return ratings, predicted

def recommender(user, ratings, predicted, n):
    user = user - 1
    sorted_ratings = ratings.iloc[user].sort_values(ascending = False)
    sorted_pred = predicted.iloc[user].sort_values(ascending = False)
    rec_df = pd.concat([sorted_ratings, sorted_pred], axis = 1)
    rec_df.columns = ['user_ratings', 'user_predictions']
    rec_df = rec_df.loc[rec_df.user_ratings == 0]
    rec_df = rec_df.sort_values('user_predictions', ascending=False)
    temp = list(rec_df.index.values)
    return temp[:n]

def runCF(df, df2, user_id, x):
    ratings, predicted = train(df)
    recommendations = recommender(user_id, ratings, predicted, x)
    print('\nHere are your recommendations:\n')
    for i in recommendations:
        print(restaurants['name'][i])

def evaluate_RMSE(actual, predicted):
    actual = actual.mean()
    predicted = predicted.mean()
    evaluation = pd.concat([actual, predicted], axis = 1)
    evaluation.columns = ['Actual', 'Predicted']
    RMSE = round((((evaluation.Actual - evaluation.Predicted) ** 2).mean() ** 0.5), 5)
    print(RMSE)

def evaluate_diversity(df, df2, ratings, predicted):
    averages = []
    user_ids = df['user_id'].tolist()
    random = np.random.choice(user_ids, size = 10)
    for i in random:
        temp = recommender(i, ratings, predicted, 10)
    vectors = get_vectors(df2)
    for i in random:
        temp = recommender(i, ratings, predicted, 10)
        lst = []
        for i in temp:
            lst.append(vectors[i])
        averages_temp = cos_sim_mul(lst)
        for i in range(len(averages_temp)):
            averages.append(sum(averages_temp[i]) / len(averages_temp[i]))
    print(sum(averages)/len(averages))

