import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns  # visualization tool

loc = 'C:\\Users\\Kintesh\\Desktop\\code\\kaggle\\pokemon\\'

combats = pd.read_csv(loc + "combats.csv")
pokemon = pd.read_csv(loc + "pokemon.csv")
test_01 = pd.read_csv(loc + "tests.csv")

print(combats.head(n=5))

print(pokemon.info())

list = [1,4,5,6,8,9,11,12,14,15,16,17]
median = np.median(list)

