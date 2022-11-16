import pandas as pd
import numpy as np
from CACB import runCB
from CF import runCF

# Load the relevant datasets
restaurants = pd.read_csv('restaurants.csv')
users = pd.read_csv('users.csv')
reviews = pd.read_csv('reviews.csv')

# Check if the user_id is valid and if so return this as an integer
def get_user_id(user_id):
    while True:
        try:
            user_id = int(input())
            break
        except:
            print('Invalid User ID, please try again.')
            print('The User ID should be an integer value')
    return user_id

# For new users, generate their user_id
def get_new_user_id(df, name):
    new_id = users['user_id'].max() + 1
    df.loc[-1] = [new_id, name, 0, 0]  # adding a row
    return new_id

# Get the name of the user using the user_id
def get_user_name(df, user_id):
    name = users['name'][user_id - 1]
    return name

# Get 10 random recommendations as a starting point for new users
def get_random(df):
    temp = df['name'].values.tolist()
    random = np.random.choice(temp, size = 10)
    print('\nHere are some recommendations to get you started.\n')
    for i in random:
        print(i)
    print('\nThank you for using the system.')

# Run the recommender system
def main_menu():

    user_id_list = users['user_id'].tolist()
    
    print('\nWelcome to the Restaurant Recommender System')
    print('At any time, you can enter EXIT to exit the system.')
    print('If you are an exisiting user, please enter your User ID')
    print('Otherwise, please enter REGISTER to create an account')

    flag = True

    while flag:
        
        flag = False

        temp = input()
        print(temp)

        if temp == 'EXIT':
            exit()

        elif temp == 'REGISTER':
            name = input('Please enter your name:\n')
            if name == 'EXIT':
                exit()
            user_id = get_new_user_id(users, name)
            print('Welcome ' + name + '. Your new User ID is ' + str(user_id))
            get_random(restaurants)
            exit()
        try:
            if int(temp) in user_id_list:
                user_id = int(temp)
                name = get_user_name(users, user_id)
                print('Welcome back ' + name)
            else:
                print('The given user ID does not exist. Please try again.')
                print('The User ID should be an integer value')
                print('Alternatively, enter REGISTER to create an account.')
                flag = True                
        except:
            print('The given user ID does not exist. Please try again.')
            print('The User ID should be an integer value')
            print('Alternatively, enter REGISTER to create an account.')
            flag = True

    print('\nHow many recommendations would you like?')
    n = int(input())
    
    print('\nWhich Recommender System would you like to use today?')
    print('CB - Content Based')
    print('CF - Collaborative Filtering')

    flag = True

    while flag:

        flag = False

        recommender = input()
    
        if recommender == 'CB':
            print('\nYour recommendations are being generated...')
            runCB(restaurants, reviews, user_id, n)
            print('\nThank you for using this system.')
            exit()
        elif recommender == 'CF':
            print('\nYour recommendations are being generated...')
            runCF(reviews, restaurants, user_id, n)
            print('\nThank you for using this system.')
            exit()
        else:
            print('Invalid input. Please try again.')
            flag = True

if __name__ == '__main__':
    main_menu()

