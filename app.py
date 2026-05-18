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
    data = request.json['answers']  # 유저의 5차원 답변 데이터 [q1, q2, q3, q4, q5]
    user_features = np.array(data).reshape(1, -1)
    
    # 1. 유저 데이터 스케일링 및 예측
    user_scaled = scaler.transform(user_features)
    prediction = knn.predict(user_scaled)[0]
    
    # 💡 2. 진짜 5차원 유클리드 거리가 가장 가까운 이웃 5명의 거리(distances)와 인덱스(indices) 추출
    distances, indices = knn.kneighbors(user_scaled, n_neighbors=5)
    
    nearest_seniors = []
    for rank, (idx, dist) in enumerate(zip(indices[0], distances[0])):
        row = df.iloc[idx]
        nearest_seniors.append({
            'x': float(row['X1']),
            'y': float(row['X2']),
            'type': row['type'],
            'distance': round(float(dist), 2)  # 💡 유클리드 거리 수치 (소수점 둘째자리 반올림)
        })
        
    return jsonify({
        'result': prediction,
        'user_coord': {'x': float(data[0]), 'y': float(data[1])},
        'nearest_data': nearest_seniors  # 💡 전체 데이터 대신 딱 5명만 전송!
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)