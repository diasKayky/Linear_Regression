import LinearRegression__ as lr
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split

#Limpa o dataset
df = pd.read_csv("dataset.csv")
del df["sqft_living15"]
del df["sqft_above"]
del df["sqft_basement"]
n_df = df.iloc[:, [2, 3, 4, 5, 7]]
N_df = n_df.drop(columns=["floors"])

#Splita os dados de X e Y
features = ["bedrooms", "bathrooms", "sqft_living"]
X = N_df[features]
Y = N_df["price"]
X_train, X_test, Y_train, Y_test = train_test_split(X, Y)

#Fita o modelo
modelo_ = lr.Linear_Regression()
modelo_.fit_model(X_train, Y_train)

#Salva o modelo no pickle
pickle.dump(modelo_, open('modelo.pkl', 'wb'))
modelo = pickle.load(open('modelo.pkl', 'rb'))

#Testa se está armazenado no pickle
print(modelo.predict([[4, 5, 600]]))