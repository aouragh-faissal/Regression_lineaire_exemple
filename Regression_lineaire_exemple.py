import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd  
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


 #1fst commit
data_url = "http://lib.stat.cmu.edu/datasets/boston"
raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
target = raw_df.values[1::2, 2]
columns = ['CRIM','ZN','INDUS','CHAS','NOX','RM','AGE','DIS','RAD','TAX','PTRATIO','B','LSTAT']
  
boston = pd.DataFrame(data=data,columns=columns)
boston['MEDV'] = target

# vérification des valeurs manquantes dans toutes les colonnes
print(boston.isnull().sum())

# tracer un histogramme montrant la distrribution de la variable cible (target)
sns.displot(data=boston,x='MEDV', bins=30,kde=True);
plt.show()

# calculer la corrélation par paire pour toutes les colonnes  
correlation_matrix = boston.corr().round(2)

# utiliser la fonction heatmap de seaborn pour tracer la matrice de corrélation
# annot = True pour imprimer les valeurs à l'intérieur da chaque case
sns.heatmap(data=correlation_matrix, annot=True);
plt.show()


plt.figure(figsize=(15, 5))
features = ['LSTAT', 'RM']
target = boston['MEDV']
for i, col in enumerate(features):
    plt.subplot(1, len(features) , i+1)
    x = boston[col]
    y = target
    plt.scatter(x, y, marker='o')
    plt.title(col)
    plt.xlabel(col)
    plt.ylabel('MEDV')

    
plt.show()   


X = pd.DataFrame(np.c_[boston['LSTAT'], boston['RM']], columns = ['LSTAT','RM'])
Y = boston['MEDV']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state=101)

#Mise à l'échelle des donnéess
"""
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
"""

"""
print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)
"""

linear_model = LinearRegression()
linear_model.fit(X_train, Y_train)

# évaluation du modèle pour l'ensemble d'entraînement
y_train_predict = linear_model.predict(X_train)
rmse = (np.sqrt(mean_squared_error(Y_train, y_train_predict)))
r2 = r2_score(Y_train, y_train_predict)
print("La performance du Modele pour le set de Training")
print("------------------------------------------------")
print("l'erreur RMSE esst {}".format(rmse))
print('le score R2 est {}'.format(r2))
print("\n")
# évaluation du modèle pour le set de tesst
y_test_predict = linear_model.predict(X_test)
# racine carrée de l'erreur quadratique moyenne du modèle
rmse = (np.sqrt(mean_squared_error(Y_test, y_test_predict)))
# score R carré du modèle
r2 = r2_score(Y_test, y_test_predict)
print("La performance du Modele pour le set de Test")
print("--------------------------------------------")
print("l'erreur RMSE est {}".format(rmse))
print('le score R2 score est {}'.format(r2))

# On décompose le dataset et on le transforme en matrices pour pouvoir effectuer notre calcul
A = np.matrix([np.ones(Y_test.shape[0]), Y_test.values]).T
B = np.matrix(y_test_predict).T

# On effectue le calcul exact du paramètre theta
theta = np.linalg.inv(A.T.dot(A)).dot(A.T).dot(B)
print(theta)
sns.scatterplot(x=Y_test, y=y_test_predict)

plt.plot([0,50], [theta.item(0),theta.item(0) + 50 * theta.item(1)], linestyle='--', c='#000000')

plt.show()

