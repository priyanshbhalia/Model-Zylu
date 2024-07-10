import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import NearestNeighbors
from flask import Flask, request, jsonify


transactions = pd.read_csv('transactions.csv') 
customers = pd.read_csv('customers.csv')        


transactions['purchase_date'] = pd.to_datetime(transactions['purchase_date'])
transactions['product_id'] = transactions['product_id'].astype(str)
customers['customer_id'] = customers['customer_id'].astype(str)


data = pd.merge(transactions, customers, on='customer_id')


data.fillna(method='ffill', inplace=True)
le = LabelEncoder()
data['product_id'] = le.fit_transform(data['product_id'])
data['customer_id'] = le.fit_transform(data['customer_id'])


data['total_spent'] = data.groupby('customer_id')['price'].transform('sum')


data['purchase_frequency'] = data.groupby('customer_id')['purchase_date'].transform(lambda x: (x.max() - x.min()).days / x.nunique())


data['returned'] = data['purchase_date'].diff().apply(lambda x: 1 if pd.notnull(x) and x.days > 30 else 0)
data['repurchased'] = data.duplicated(subset=['customer_id', 'product_id'], keep='first').astype(int)


data.drop(columns=['purchase_date', 'price'], inplace=True)


features = data.drop(columns=['returned', 'repurchased'])
labels_return = data['returned']
labels_repurchase = data['repurchased']


X_train_return, X_test_return, y_train_return, y_test_return = train_test_split(features, labels_return, test_size=0.2, random_state=42)
X_train_repurchase, X_test_repurchase, y_train_repurchase, y_test_repurchase = train_test_split(features, labels_repurchase, test_size=0.2, random_state=42)


model_return = RandomForestClassifier()
model_return.fit(X_train_return, y_train_return)

model_repurchase = RandomForestClassifier()
model_repurchase.fit(X_train_repurchase, y_train_repurchase)


user_item_matrix = data.pivot(index='customer_id', columns='product_id', values='total_spent').fillna(0)


model_recommend = NearestNeighbors(metric='cosine', algorithm='brute')
model_recommend.fit(user_item_matrix)


app = Flask(__name__)

@app.route('/predict_return', methods=['POST'])
def predict_return():
    data = request.get_json()
    input_data = pd.DataFrame([data])
    prediction = model_return.predict(input_data)
    return jsonify({'return_probability': prediction[0]})

@app.route('/predict_repurchase', methods=['POST'])
def predict_repurchase():
    data = request.get_json()
    input_data = pd.DataFrame([data])
    prediction = model_repurchase.predict(input_data)
    return jsonify({'repurchase_probability': prediction[0]})

@app.route('/recommend_products', methods=['POST'])
def recommend_products():
    data = request.get_json()
    customer_id = data['customer_id']
    input_data = user_item_matrix.loc[customer_id].values.reshape(1, -1)
    distances, indices = model_recommend.kneighbors(input_data, n_neighbors=5)
    recommended_product_ids = user_item_matrix.columns[indices.flatten()].tolist()
    return jsonify({'recommended_products': recommended_product_ids})

if __name__ == '__main__':
    app.run(debug=True)
