import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import style
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

style.use('ggplot')  # setting style

# reading data
url = "http://bit.ly/w-data"
df = pd.read_csv(url)

df.plot(x='Hours', y='Scores', style='o')
plt.title('Hours vs. Percentage')
plt.xlabel('Hours Studied')
plt.ylabel('Percentage Score')
plt.show()

X = df.iloc[:, :-1].values
y = df.iloc[:, 1].values
X = preprocessing.scale(X)
df.dropna(inplace=True)

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.2, random_state=0)

clf = LinearRegression(n_jobs=-1)
clf.fit(X_train, y_train)
accuracy = clf.score(X_test, y_test)
# print(accuracy)
pred = clf.predict(X_test)  # Predicting scores
# print(pred)

df2 = pd.DataFrame({'Actual': y_test, 'Predicted': pred})
# print(df2)
df3 = df.join(df2)
print(df3[0:5])

