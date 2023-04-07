import pandas as pd
from sklearn.tree import DecisionTreeRegressor

melb = pd.read_csv("./melb_data.csv")
melb = melb.dropna(axis = 0)
y = melb.Price
melb_feats = ["Rooms", "Bathroom", "Landsize", "Lattitude", "Longtitude"]
X = melb[melb_feats]
melb_model = DecisionTreeRegressor(random_state = 1)
melb_model.fit(X, y)

# Now for Iowa
iowa = pd.read_csv("./iowa-train.csv")
y = iowa["SalePrice"]
iowa_feats = [
  "LotArea",
  "YearBuilt",
  "1stFlrSF",
  "2ndFlrSF",
  "FullBath",
  "BedroomAbvGr",
  "TotRmsAbvGrd",
]
X = iowa[iowa_feats]
iowa_model = DecisionTreeRegressor(random_state = 1)
iowa_model.fit(X, y)
predictions = iowa_model.predict(X)
