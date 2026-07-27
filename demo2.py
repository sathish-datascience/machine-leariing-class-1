from sklearn.linear_model import LogisticRegression

#this is student study hours
x = [[1],[2],[3],[4],[5],[6]]

#this is result
y = [0,0,0,1,1,1]

#model creation
model = LogisticRegression()

#model training
model.fit(x,y)

#prediction
prediction = model.predict([[4.5]])


if prediction == 1:
    print("pass")
else:
    print('fail')    
    
from sklearn.naive_bayes import GaussianNB

model1 = GaussianNB()    

model1.fit(x,y)

print(model.predict([[5.5],[6.5]]))