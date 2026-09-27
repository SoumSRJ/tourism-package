import pandas as pd
from sklearn.model_selection import train_test_split

RAW_PATH = "tourism_project/data/tourism.csv"

# Load the raw dataset
df = pd.read_csv(RAW_PATH)
df = df.drop(df.columns[0], axis=1) #dropping the first column
df = df.drop(columns=["CustomerID"])
df['Gender'] = df['Gender'].replace('Fe male', 'Female') # rectifying the gender column
df['MaritalStatus'] = df['MaritalStatus'].replace('Unmarried', 'Single') #merging the unmarried values into single


X = df.drop(columns=["ProdTaken"])
y = df["ProdTaken"]

# stratify=y keeps the (imbalanced) failure ratio consistent across splits
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

Xtrain.to_csv("Xtrain.csv", index=False)
Xtest.to_csv("Xtest.csv", index=False)
ytrain.to_csv("ytrain.csv", index=False)
ytest.to_csv("ytest.csv", index=False)

print("Data prepared: train/test splits written.")
print("TypeofContact values kept as:", sorted(X["TypeofContact"].unique()))
