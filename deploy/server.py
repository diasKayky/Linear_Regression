from flask import Flask, request, jsonify, render_template
import numpy as np
import pickle

#Define o app Flask
app = Flask(__name__)

#Carrega o modelo no pickle
modelo = pickle.load(open('modelo.pkl', 'rb'))

#Define a rota de predição
@app.route('/predict', methods=['GET', 'POST'])

#Define a função que armazena o POST, prediz o valor do imóvel e retorna no JSON
def predict():
    #Checa se o método é 'POST'
    if request.method == 'POST':
        #Pega os dados do POST
        data = request.get_json()
        #Define 'bedrooms' a partir do POST
        bedrooms = np.array(data["bedrooms"]).tolist()
        # Define 'bathrooms' a partir do POST
        bathrooms = np.array(data["bathrooms"]).tolist()
        # Define 'sqft_living' a partir do POST
        sqft_living = np.array(data["sqft_living"]).tolist()

        # Usa os dados do POST para predição
        X = [[bedrooms, bathrooms, sqft_living]]
        pred = np.array(modelo.predict(X)).tolist()

        # Retorna o valor predito
        return {'preco_imovel_predito': pred[0]}

if __name__ == "__main__":
    app.run(debug=True)