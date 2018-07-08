import pandas as pd

combats = pd.read_csv(r"C:\Users\Kintesh\Desktop\code\kaggle\pokemon\combats.csv")
pokemon = pd.read_csv(r"C:\Users\Kintesh\Desktop\code\kaggle\pokemon\pokemon.csv")
test_01 = pd.read_csv(r"C:\Users\Kintesh\Desktop\code\kaggle\pokemon\tests.csv")

print(combats.head(n=5))

