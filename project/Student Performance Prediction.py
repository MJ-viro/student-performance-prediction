import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

data = pd.read_csv("Students.csv")
print(data)
data.fillna(data.mean(numeric_only=True),inplace=True)
print(data)
data["Result"] = data["Result"].map({"Pass":1,"Fail":0})
print(data)
x=data[["Maths","Science","English"]]
y=data["Result"]
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)
model = LinearRegression()
model.fit(x_train,y_train)

print("Accuracy:", model.score(x_test, y_test))




