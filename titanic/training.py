import pandas as pd

train_data = pd.read_csv("data/train.csv")
women = train_data.loc[train_data.Sex == "female"]
women_ct = len(women)
women_survived = women.loc[women["Survived"] == 1]
women_survived_ct = len(women_survived)
