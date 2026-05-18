from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
import os

app = Flask(__name__)

# 데이터 로드 및 KNN 모델 학습
if not os.path.exists('senior_data.csv'):
    import subprocess
    subprocess.run(['python', 'generate_data.py'])

df = pd.read_csv('senior_data.csv')
X = df[['X1', 'X2', 'X3', 'X4', 'X5']]
y = df['type']

# 데이터 스케일링 및 모델 피팅
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_scaled, y)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json['answers'] # [q1, q2, q3, q4, q5]
    user_features = np.array(data).reshape(1, -1)
    
    # 유저 데이터 스케일링 및 예측
    user_scaled = scaler.transform(user_features)
    prediction = knn.predict(user_scaled)[0]
    
    # 시각화를 위해 기존 데이터와 유저 데이터를 2차원(X1: 사교, X2: 실전)으로 전송
    seniors_list = []
    for _, row in df.iterrows():
        seniors_list.append({
            'x': float(row['X1']),
            'y': float(row['X2']),
            'type': row['type']
        })
        
    return jsonify({
        'result': prediction,
        'user_coord': {'x': float(data[0]), 'y': float(data[1])},
        'all_data': seniors_list
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)