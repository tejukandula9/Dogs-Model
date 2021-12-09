import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier,DecisionTreeRegressor, plot_tree
from sklearn import tree
from sklearn import metrics
from matplotlib import pyplot as plt
import plotly.express as px

# Creating breeds Dict
def create_breeds_dict(breeds):
    breeds_dict = {}
    for col in breeds:
        curr_dict = dict.fromkeys(breeds[col].dropna(), col)
        breeds_dict.update(curr_dict)
    breeds_dict['Chihuahua Shorthair']='Toy'
    breeds_dict['Chihuahua Longhair']='Toy'
    breeds_dict['Pit Bull']='Pit Bull'
    breeds_dict['German Shepherd']='Herding'
    breeds_dict['Australian Kelpie']='Herding'
    breeds_dict['Doberman Pinsch'] = 'Working'
    breeds_dict['Alaskan Husky'] = 'Working'
    breeds_dict['Staffordshire'] = 'Terrier'
    breeds_dict['Collie Smooth'] = 'Herding'
    breeds_dict['Catahoula'] = 'Working'
    breeds_dict['Presa Canario'] = 'Herding'
    breeds_dict['Black Mouth Cur'] = 'Herding'
    breeds_dict['Anatol Shepherd'] = 'Working'
    return breeds_dict

breeds = pd.read_csv('dog_breeds.csv')
breeds = breeds.loc[:, 'Herding':'Working']
breeds_dict = create_breeds_dict(breeds)

def find_breed_type(breed):
    if 'Mix' in breed:
        breed = breed.replace(' Mix', '')
    if '/' in breed:
        breed_list = breed.split('/', 1)
        for breed in breed_list:
            if breed in breeds_dict:
                return breeds_dict[breed]
            if 'Hound' in breed:
                return 'Hound'
            if 'Poodle' in breed:
                return 'Non-Sporting'
        return 'Not Classified'
    if 'Bulldog' in breed:
        breed = 'Bulldog'
    if 'Dachshund' in breed:
        breed = 'Dachshund'
    if 'Pit Bull' in breed:
        breed = 'Pit Bull'
    if 'Chesa Bay' in breed:
        breed = 'Chesapeake Bay Retriever'
    if 'Queensland Heeler' in breed:
        breed = 'Australian Cattle Dog'
    if breed in breeds_dict:
        return breeds_dict[breed]
    if 'Terrier' in breed:
        return 'Terrier'
    if 'Toy' in breed or 'Miniature' in breed:
        return 'Toy'
    if 'Poodle' in breed:
        return 'Non-Sporting'
    if 'Hound' in breed:
        return 'Hound'
    return 'Not Classified'

def create_dogs_df():
    dogs = pd.read_csv('dog_data.csv')
    dogs['Breed Type'] = dogs['Breed'].map(find_breed_type)
    dogs = dogs[dogs['Outcome Type'].isin(['Adoption', 'Euthanasia'])]

    # Clean Variables to reduce categories
    dogs = dogs[dogs['Sex upon Outcome'] != 'Unknown']
    dogs['Adoption Status'] = np.where(dogs['Outcome Type'] == 'Adoption', 'Adopted', 'Euthanised')
    dogs['Sex'] = np.where(dogs['Sex upon Outcome'].str.contains('Female'), 'Female', 'Male')
    dogs['Fixed_Status'] = np.where(dogs['Sex upon Outcome'].str.contains('Intact'), 'Intact', 'Fixed')
    dogs['Pitbull_Status'] = np.where(dogs['Breed Type'].str.contains('Pit Bull'), 'Pit Bull', 'Not Pit Bull')
    dogs['Condition_Status'] = np.where(dogs['Intake Condition'].str.contains('Normal'), 'Normal', 'Not Normal')
    dogs['Senior_Status'] = np.where(dogs['Age upon Outcome (months)']<132, 'Not Senior', 'Senior')
    dogs.reset_index(drop=True, inplace = True)

    # Create Dummy Variables
    dogs = pd.concat([dogs, pd.get_dummies(dogs['Sex'], prefix='Sex')], axis=1)
    dogs = pd.concat([dogs, pd.get_dummies(dogs['Fixed_Status'], prefix='Fixed_Status')], axis = 1)
    dogs = pd.concat([dogs, pd.get_dummies(dogs['Pitbull_Status'], prefix='Pitbull_Status')], axis=1)
    dogs = pd.concat([dogs, pd.get_dummies(dogs['Condition_Status'], prefix='Condition_Status')], axis=1)
    dogs = pd.concat([dogs, pd.get_dummies(dogs['Senior_Status'], prefix='Senior_Status')], axis=1)
    dogs = pd.concat([dogs, pd.get_dummies(dogs['Breed Type'], prefix='Breed_Type')], axis=1)
    dogs = pd.concat([dogs, pd.get_dummies(dogs['Color'], prefix='Color')], axis=1)
    return dogs

def create_model(x_vals, depth=2):
    dogs = create_dogs_df()
    x = dogs[x_vals]
    y = dogs['Adoption Status']
    x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.30)

    dtree =  DecisionTreeClassifier(max_depth=depth)
    dtree.fit(x_train,y_train)

    y_pred = dtree.predict(x_test)
    accuracy = metrics.accuracy_score(y_test, y_pred)
    return (dtree, x.columns, accuracy)

def visualize_tree(model, cols):
    fig = plt.figure(figsize=(3,1))
    tree.plot_tree(model, feature_names = cols, class_names = ['Adopted', 'Euthanised'], filled = True, proportion = True, rounded = True)

def find_importance(model, col_names):
    importance = pd.Series(model.feature_importances_, index = col_names).sort_values(ascending=False).to_frame()
    importance.reset_index(inplace=True)
    importance.columns = ['Factor', 'Importance Level']
    return importance

    
def get_col_names(substr):
    cols = list(create_dogs_df().columns)
    cols_subset = [col for col in cols if substr in col]
    return cols_subset
