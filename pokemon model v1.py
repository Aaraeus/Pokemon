"""
First machine learning model
This one uses the pokemon data + combats to generate a list of Probability of Win (pw) values using Linear Regression
Each pokemon gets a pw value, then whichever is highest is deemed the winner.

Gotta spruce this up but for now, call this v1
"""


import numpy as np  #linear algebra
import pandas as pd  #data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns  # visualization tool
from sklearn.model_selection import train_test_split #model selection
from scipy import stats

# Tutorial used to base this off
# https://www.kaggle.com/mmetter/pokemon-data-analysis-tutorial


loc = 'C:\\Users\\Kintesh\\Desktop\\code\\kaggle\\pokemon\\'

combats = pd.read_csv(loc + "combats.csv")
pokemon = pd.read_csv(loc + "pokemon.csv")
test_01 = pd.read_csv(loc + "tests.csv")


# Rename pokemon column called hashtag to number instead
pokemon = pokemon.rename(index=str, columns={"#": "Number"})

print("Starting dataframe shapes: ")
print(combats.shape)
print(pokemon.shape)
print(test_01.shape)


# Also change combats to show who won - 0 for the first pokemon, and 1 for the second

def boolean_func(df):
    """ Test Function for generating new value"""
    if df['Winner'] == df['First_pokemon']:
        return 0
    elif df['Winner'] == df['Second_pokemon']:
        return 1
    

def normality_test(df,var):
    plt.figure(figsize=(6,8))
    plt.subplot(211)
    sns.distplot(df[var], fit=stats.norm)
    plt.subplot(212)
    stats.probplot(df[var], plot=plt)
    plt.tight_layout()
    #plt.savefig('C:/Users/G01180794/Desktop/python_out/%s.png' % var, dpi=400)
    plt.show()
    print('Skewness: %f' % df[var].skew())
    print('Kurtosis: %f' % df[var].kurt())
    
# STATISTICAL TEST - TREND/FIT
    
def trend(df,var1,var2):
    plt.figure()
    sns.jointplot(x=df[var1], y=df[var2], kind='reg')
    #plt.savefig('C:/Users/G01180794/Desktop/python_out/%s.png' % var, dpi=400)
    plt.show()
    print('Skewness: %f' % df[var1].skew())
    print('Kurtosis: %f' % df[var1].kurt())
    

combats['winner_boolean'] = combats.apply(boolean_func, axis=1)

# Quick view of datasets
print(pokemon.head(n=5))
print(combats.head(n=5))

print("Dimensions of Pokemon: " + str(pokemon.shape))
print("Dimensions of Combat: " + str(combats.shape))


# Check pokemon dataset
pokemon.isnull().sum()

# Pull out the record with a missing pokemon name
print(pokemon[pokemon['Name'].isnull()])

# This is Row # 62, let's see what comes before and after
print("This pokemon is before the missing Pokemon: " + pokemon['Name'][61])
print("This pokemon is after the missing Pokemon: " + pokemon['Name'][63])

# It's Primape - update this manually.
pokemon['Name'][62] = "Primeape"

# Check row is updated - iloc does integer values
pokemon.iloc[[62]]

print(pokemon.info())

# Can we put the number of wins and losses? Create dataset of total number of wins and losses

numberOfWins = combats.groupby('Winner').count()
countByFirst = combats.groupby('Second_pokemon').count()
countBySecond = combats.groupby('First_pokemon').count()

# Drop boolean field because we're just counting for no reason.
numberOfWins = numberOfWins.drop(['winner_boolean'],axis=1)
numberOfWins['Winner'] = numberOfWins.index


numberOfWins['total_fights'] = countByFirst.Winner + countBySecond.Winner
numberOfWins['win_pct'] = numberOfWins.First_pokemon/numberOfWins['total_fights']

print(list(pokemon))
print(list(numberOfWins))

print(pokemon.head(n=5))
print(numberOfWins.head(n=5))

# Merge to create an MRD (Model Ready Dataset)

print("Merge prep:")
print("Pokemon Shape: " + str(pokemon.shape))
print("numberOfWins Shape: " + str(numberOfWins.shape))

mrd = pd.merge(pokemon,numberOfWins,how='left',left_on ='Number', right_on='Winner',indicator=True)
mrd.drop(['Winner'],axis=1)

# Also merge back on the boolean winner since this has been messed up by our grouping...
print(list(mrd))

# Change all NaNs to 0 in win_count
#mrd['win_pct'].fillna(0, inplace=True)

print("MRD Shape: " + str(mrd.shape))

pd.set_option('display.max_columns', 20)

# Take a look at top 20 pokemon by win percentage
print(mrd.sort_values(by='win_pct', ascending=False).head(n=20))

# Take a look at worst 20 pokemon by win percentage
print(mrd.sort_values(by='win_pct', ascending=True).head(n=20))

# Win percentage split by Type 1
mrd.groupby('Type 1').agg({"win_pct": "mean"}).sort_values(by = "win_pct", ascending = False )

# Correlation graphs for all stats by win percentage
col = ['Type 1', 'HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed', 'win_pct']
sns.pairplot(mrd.loc[:, col].dropna())
plt.show()


# Correlation table
mrd.loc[:,col].corr()

# Data analysis conclusions
# Speed is heavily correlated with winning
# Top winners are Flying, Dragon, Electric and Dark types
# Lowest Winners are Fairy, Rock, Steel and Poison types

# Actually build the model using a new dataset called final_mrd
# First, we need to split into a TESTING (80%) and TRAINING (20%) dataset

print(mrd.head(n=5))

#remove rows with NA values because it will cause errors when fitting to the model
#Take off the type2 field (easier to just specify the ones we wanna keep) because this would cause a lot of
# single typed pokemon to be removed from the analysis!!
mrd_final = mrd[['Name', 'Number', 'HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed', 'win_pct']].dropna(axis=0, how='any')


# Normality test on win_pct
normality_test(mrd_final,'win_pct')


mrd_final['SpeedAndAtk']=mrd_final['Speed']+mrd_final['Attack']

print(mrd_final.shape)
# Trend

trend(mrd_final,'Speed','win_pct')
trend(mrd_final,'SpeedAndAtk','win_pct')


# Take main stat variables as x
featEngVars = ['HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed']

x = mrd_final[featEngVars]
print(x)


y = mrd_final['win_pct']
print(y)


#x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)

print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)

print(y_train)

# Random forest
from sklearn.ensemble import RandomForestRegressor

# RANDOM FOREST VARIABLE IMPORTANCE
     
def variable_importance(classif):
    importance = classif.feature_importances_
    importance = pd.DataFrame(importance, index=x.columns, 
                              columns=["Importance"])
    importance = importance.sort_values(['Importance'], ascending=[0])
    plt.subplots(figsize=(12,10))
    sns.barplot(importance['Importance'], importance.index)
    #plt.savefig('C:/Users/G01180794/Desktop/python_out/importance.png', dpi=400)
    plt.show()


rf = RandomForestRegressor(n_estimators=200, max_depth=12, random_state=0)
rf.fit(x, y)
importance = rf.feature_importances_
df = variable_importance(rf)



''' LINEAR REGRESSION '''

from sklearn.linear_model import LinearRegression

regressor = LinearRegression()
regressor.fit(x_train, y_train)
print(regressor.score(x, y))

y_pred = regressor.predict(x_train)

# Validating the results using mean absolute error

from sklearn.metrics import mean_absolute_error

mae = mean_absolute_error(y_train, y_pred)
print("Mean Absolute Error: " + str(mae))

# Now populate model
model = regressor.fit(x,y)
# Create a probability of win dataset for pokemon! This will be a lookup table.
pw = model.predict(x)
pw_lookup = mrd_final[['Name', 'Number']]
print(pw_lookup.shape)
print(pw.shape)
# Create probability of win variable in pw_lookup table
pw_lookup['pw'] = pw

print(pw_lookup.head(n=10))


# Now that we have a pw_lookup table, merge onto our test_01 to get new results...
print(test_01.head(n=5))
pw1 = pd.merge(test_01[['First_pokemon', 'Second_pokemon']], pw_lookup, how='left', left_on ='First_pokemon', right_on='Number',indicator=True)


pw1.columns = ['First_pokemon', 'Second_pokemon', 'Name1', 'Number1', 'pw1', '_merge1']

pw2 = pd.merge(pw1, pw_lookup, how='left', left_on ='Second_pokemon', right_on='Number',indicator=True)

pw2.columns = ['First_pokemon', 'Second_pokemon', 'Name1', 'Number1', 'pw1', '_merge1', 'Name2', 'Number2', 'pw2', '_merge2']

print(pw2.head(n=5))

import os
cwd = os.getcwd()
print(cwd)


def final_winner(df):
    if pd.isnull(df['pw1']) and not pd.isnull(df['pw2']):
        return df['Name1']
    elif pd.isnull(df['pw2']) and not pd.isnull(df['pw1']):
        return df['Name1']
    elif df['pw2'] > df['pw1']:
        return df['Name2']
    else:
        return df['Name1']

pw2['Winner'] = pw2.apply(final_winner, axis=1)

print(list(pw2))

final_result_detailed = pw2[['First_pokemon', 'Second_pokemon', 'Name1', 'Name2', 'pw1', 'pw2', 'Winner']]
final_result = pw2[['First_pokemon', 'Second_pokemon', 'Winner']]


print(list(pokemon))

writer = pd.ExcelWriter(cwd + '\\' + 'pw2.xlsx')
pw2.to_excel(writer,'Sheet1')
writer.save()

writer2 = pd.ExcelWriter(cwd + '\\' + 'output_model.xlsx')
pw_lookup.to_excel(writer2,'Sheet1')
writer2.save()

writer3 = pd.ExcelWriter(cwd + '\\' + 'output_mrd_final.xlsx')
mrd_final.to_excel(writer3,'Sheet1')
writer3.save()

writer4 = pd.ExcelWriter(cwd + '\\' + 'output_mrd.xlsx')
mrd.to_excel(writer4,'Sheet1')
writer4.save()

writer5 = pd.ExcelWriter(cwd + '\\' + 'output_final_result_detailed.xlsx')
final_result_detailed.to_excel(writer5,'Sheet1')
writer5.save()

writer6 = pd.ExcelWriter(cwd + '\\' + 'output_final_result.xlsx')
final_result.to_excel(writer6,'Sheet1')
writer6.save()

print("Pokemon model complete, and outputs created.")

# # Root mean square error
# from sklearn.model_selection import cross_val_score,KFold
#
#
# def cross_val_rmse(folds, model, data_set, features, target):
#     kf = KFold(folds, shuffle=True, random_state=42).get_n_splits(data_set.values)
#     rmse = np.sqrt(-cross_val_score(model, features.values,
#                                     target.values,
#                                     scoring="neg_mean_squared_error", cv=kf))
#     return("Score: {:.4f}({:.4f})".format(rmse.mean(), rmse.std()))
#
# cross_val_rmse(3, regressor, mrd_final, x, y)

# FOR LATER WHEN I WANNA TRY TO AUTOMATE THIS BAD BOI
# from sklearn.metrics import accuracy_score
# from sklearn.linear_model import LogisticRegression
# from sklearn.naive_bayes import GaussianNB
# from sklearn import linear_model
# from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.svm import LinearSVC
# from sklearn.tree import DecisionTreeClassifier

# clf_dict = {
#             # 'log reg': LogisticRegression(),
#             'random forest': RandomForestClassifier(n_estimators=100),
#             'naive bayes': GaussianNB(),
#             'knn': KNeighborsClassifier(),
#             'linear svc': LinearSVC(),
#             'ada boost': AdaBoostClassifier(n_estimators=100),
#             'gradient boosting': GradientBoostingClassifier(n_estimators=100),
#             'CART': DecisionTreeClassifier()
#             }
#
# for name, clf in clf_dict.items():
#     model = clf.fit(x_train, y_train)
#     pred = model.predict(x_test)
#     print('Accuracy of {}:'.format(name), accuracy_score(pred, y_test))
#


