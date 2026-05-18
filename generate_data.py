import pandas as pd
import numpy as np

# 💡 구역(경계)이 눈에 보이도록 센트로이드 간격을 다시 명확하게 조정
centroids = {
    "학생회/과대형": [4.5, 3.5, 4.5, 3.2, 4.3],       # 우측 하단 (사교 높음, 실전 중간)
    "대외활동/네트워커형": [4.0, 4.5, 3.6, 2.5, 3.8],   # 우측 상단 (사교 높음, 실전 높음)
    "도서관 지박령형": [1.8, 2.2, 2.0, 1.5, 1.6],     # 좌측 하단 (사교 낮음, 실전 낮음)
    "실무 프로젝트형": [2.3, 4.6, 3.0, 4.0, 2.2],     # 좌측 상단 (사교 낮음, 실전 높음)
    "무난한 가성비형": [3.1, 2.8, 2.2, 3.6, 2.8]       # 정중앙 오차범위 구역
}

np.random.seed(42)
data = []

for label, center in centroids.items():
    for _ in range(100):  # 유형별 100개 개수 유지
        features = np.clip(np.random.normal(center, 0.7), 1.0, 5.0)
        features = np.round(features, 1)
        data.append(list(features) + [label])

df = pd.DataFrame(data, columns=['X1', 'X2', 'X3', 'X4', 'X5', 'type'])
df.to_csv('senior_data.csv', index=False, encoding='utf-8-sig')
print("경계선 황금 밸런스 패치 완료! 🚀")