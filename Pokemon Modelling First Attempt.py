import numpy as np  #linear algebra
import pandas as pd  #data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns  # visualization tool
from sklearn.model_selection import train_test_split #model selection

loc = 'C:\\Users\\Kintesh\\Desktop\\code\\kaggle\\pokemon\\'

combats = pd.read_csv(loc + "combats.csv")
pokemon = pd.read_csv(loc + "pokemon.csv")
test_01 = pd.read_csv(loc + "tests.csv")


# Rename pokemon column called hashtag to number instead
pokemon = pokemon.rename(index=str, columns={"#": "Number"})

# Also change combats to show who won - 0 for the first pokemon, and 1 for the second


def test_func(df):
    """ Test Function for generating new value"""
    if df['Winner'] == df['First_pokemon']:
        return 0
    elif df['Winner'] == df['Second_pokemon']:
        return 1


combats['winner_boolean'] = combats.apply(test_func, axis=1)

# Quick view of datasets
print(pokemon.head(n=5))

print(combats.head(n=5))

print("Dimensions of Pokemon: " + str(pokemon.shape))
print("Dimensions of Combat: " + str(combats.shape))


# The pokemon dataset is the one we want to analyse, so just check whether we have any missing values
pokemon.isnull().sum()

# Pull out the record with a missing pokemon name
print(pokemon[pokemon['Name'].isnull()])

# This is Row # 62, let's see what comes before and after
print("This pokemon is before the missing Pokemon: " + pokemon['Name'][61])
print("This pokemon is after the missing Pokemon: " + pokemon['Name'][63])

# It's Primape - update this manually.
pokemon['Name'][62] = "Primeape"

# Check row is updated - iloc does integer values
# pokemon.iloc[[62]]

# START ANALYSING THE DATASET BY FEATURE ENGINEERING - WHAT VARIABLES MIGHT I WANT?
print(pokemon.info())

# Can we put the number of wins and losses? Create dataset of total number of wins and losses

numberOfWins = combats.groupby('Winner').count()
countByFirst = combats.groupby('Second_pokemon').count()
countBySecond = combats.groupby('First_pokemon').count()

numberOfWins['total_fights'] = countByFirst.Winner + countBySecond.Winner
numberOfWins['win_pct'] = numberOfWins.First_pokemon/numberOfWins['total_fights']

print(numberOfWins.head(n=5))

# Merge to create an MRD (Model Ready Dataset)

print("Merge prep:")
print("Pokemon Shape: " + str(pokemon.shape))
print("numberOfWins Shape: " + str(numberOfWins.shape))

mrd = pokemon.merge(numberOfWins,how='left',left_on='Number',right_on='Winner',
                    left_index=False, right_index=False, sort=True,
                    suffixes=('_x', '_y'), copy=True, indicator=True,validate=None)

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
mrd_final = mrd.dropna(axis=0, how='any')

print(list(mrd_final))

# Take main stat variables as x
x = mrd_final.iloc[:, 5:11].values
print(x)

# Take winner_boolean in y... but check it's in col 14 first
print(list(mrd_final)[14])

y = mrd_final.iloc[:, 14].values
print(y)

print(mrd_final['winner_boolean'])

#x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)

print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)

print(y_train)
# Encode values to int so they work with sklearn
from sklearn import preprocessing
from sklearn import utils

lab_enc = preprocessing.LabelEncoder()
y_train_enc = lab_enc.fit_transform(y_train)

print(y_train_enc)

# y train encoded
print(utils.multiclass.type_of_target(y_train_enc))


# Perform linear regression on win_pct to find our new values

from sklearn.linear_model import LinearRegression

regressor = LinearRegression()
regressor.fit(x_train, y_train)
print(regressor.score(x_train, y_train))

y_pred = regressor.predict(x_test)

# Validating the results using mean absolute error

from sklearn.metrics import mean_absolute_error

mae = mean_absolute_error(y_test, y_pred)
print("Mean Absolute Error: " + str(mae))


# Try LogReg model

from sklearn.linear_model import LogisticRegression

regressor = LogisticRegression()
regressor.fit(x_train, y_train)
print(regressor.score(x_train, y_train))

y_pred = regressor.predict(x_test)

# OK, now that we have a modelling method, let's just model on the whole dataset so we can see the win_pct values populated

# print(y_pred.shape)
#
# # Model on the full data - dataframes 'x' and 'y'
# model = regressor.fit(x, y)
# prediction = model.predict(mrd)
#
# print(prediction)

# Populated the test01 table that we were asked to do in the first place

test_01['Winner'] = prediction
test_01['Winner'][test_01['Winner'] <0.5] = test_01['First_pokemon']
test_01['Winner'][test_01['Winner'] >=0.5] = test_01['Second_pokemon']

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

