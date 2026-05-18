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
@app.route('/predict', methods=['POST'])
def predict():
    raw_data = request.json['answers']  # 💡 유저의 10문항 답변 데이터 (q1 ~ q10)
    
    # 💡 10개의 답변을 성향별로 2개씩 짝지어 평균(Mean) 계산 -> 5차원 공간으로 압축!
    x1 = (raw_data[0] + raw_data[1]) / 2.0  # 사교성 (Q1, Q2)
    x2 = (raw_data[2] + raw_data[3]) / 2.0  # 실전성 (Q3, Q4)
    x3 = (raw_data[4] + raw_data[5]) / 2.0  # 리더십 (Q5, Q6)
    x4 = (raw_data[6] + raw_data[7]) / 2.0  # 계획성 (Q7, Q8)
    x5 = (raw_data[8] + raw_data[9]) / 2.0  # 활동성 (Q9, Q10)
    
    user_5d = [x1, x2, x3, x4, x5]
    user_features = np.array(user_5d).reshape(1, -1)
    
    # 데이터 스케일링 및 예측
    user_scaled = scaler.transform(user_features)
    prediction = knn.predict(user_scaled)[0]
    
    # 5차원 유클리드 거리가 가장 가까운 이웃 5명의 거리와 인덱스 추출
    distances, indices = knn.kneighbors(user_scaled, n_neighbors=5)
    
    nearest_seniors = []
    for rank, (idx, dist) in enumerate(zip(indices[0], distances[0])):
        row = df.iloc[idx]
        nearest_seniors.append({
            'x': float(row['X1']),
            'y': float(row['X2']),
            'type': row['type'],
            'distance': round(float(dist), 2) 
        })
        
    return jsonify({
        'result': prediction,
        'user_coord': {'x': x1, 'y': x2}, 
        'nearest_data': nearest_seniors
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)