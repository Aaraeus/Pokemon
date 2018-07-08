import pandas as pd

loc = 'C:\\Users\\Kintesh\\Desktop\\code\\kaggle\\pokemon\\'

combats = pd.read_csv(loc + "combats.csv")
pokemon = pd.read_csv(loc + "pokemon.csv")
test_01 = pd.read_csv(loc + "tests.csv")

print(combats.head(n=5))

