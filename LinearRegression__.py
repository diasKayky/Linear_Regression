import numpy as np

class Linear_Regression():
    
    def __init__(self):
        
        self.coef = None
        self.X = None
        self.Y = None
    
    def fit_model(self, X, Y):
        
        " " " Fita o modelo, isto é, calcula os coeficientes da equação de regressão linear " " "
        
        self.X = X
        self.Y = Y
        
        #Checa se os tipos de X e Y estão certos e converte-os caso não
        if type(X) == np.ndarray:
            pass
        elif type(X) == list:
        
            X = np.array(X)
            Y = np.array(Y)
        
        #Adiciona uma coluna com 1 em X por conta do coeficiente linear beta_0 
        X = np.column_stack((np.ones(X.shape[0]), X))
        
        # Soluciona a equação normal de least squares seguindo o teorema de Gauss-Markov
        self.coef = np.linalg.inv(X.T @ X) @ X.T @ Y
        
    def find_coef(self):
        
        " " " Retorna os coeficientes do modelo " " "
        return self.coef
    
    def predict(self, new_X):
        
        " " " Retorna um array com os dados preditos pelo modelo com base em X " " "
            
        self.new_X = new_X
        
        #Checa se o tipo de X está certo
        if type(new_X) == np.ndarray:
            pass
        elif type(new_X) == list:
            new_X = np.array(new_X)
            
        #Adiciona uma coluna com 1 em X por conta do coeficiente linear beta_0 
        new_X = np.column_stack((np.ones(new_X.shape[0]), new_X))
        
        return new_X @ self.coef
    
    def r_squared(self):
        
        " " " Retorna o valor R^2 do modelo " " "
        
        #Normaliza X para que o valor de beta_0 não afete a multiplicação matricial
        normalized_x = np.column_stack((np.zeros(self.X.shape[0]), self.X))
        
        # Prediz Y com base nos coeficientes do modelo
        Y_pred = normalized_x @ self.coef
        
        # Calcula da média de Y observado (Y dos dados que alimentam o modelo)
        Y_mean = np.mean(self.Y)
        
        # Calcula a soma dos quadrados dos resíduos
        SS_res = np.sum((self.Y - Y_pred) ** 2)
        
        #Calcula a soma dos quadrados totais
        SS_tot = np.sum((self.Y - Y_mean) ** 2)
        
        return 1 - (SS_res / SS_tot)

class TrainSplitTest:
    
    def __init__(self, X, Y, test_size=0.25):
        
        self.X = X
        self.Y = Y
        self.test_size = test_size
    
    def split(self):
        
        " " " Splita X e Y em X_train, X_test e Y_train, Y_test " " "
        
        # Reorganiza os dados aleatoriamente
        np.random.shuffle(self.X)
        np.random.shuffle(self.Y)
        
        # Calcula a quantidade de dados que vão pra train de X e Y
        index_X = int((1 - self.test_size) * len(self.X))
        index_Y = int((1 - self.test_size) * len(self.Y))
        
        # Splita os dados de X em Y em train e test com base em index_X e index_Y respectivamente
        train_X = self.X[:index_X]
        test_X = self.X[index_X:]
        train_Y = self.Y[:index_Y]
        test_Y = self.Y[index_Y:]
        
        return train_X, test_X, train_Y, test_Y