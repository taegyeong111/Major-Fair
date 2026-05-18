import pandas as pd
import numpy as np

# 클래스별 중심점 설정 [사교, 실전, 리더, 벼락, 유흥]
centroids = {
    "학생회/과대형": [4.8, 3.5, 4.7, 3.2, 4.5],
    "대외활동/네트워커형": [4.2, 4.6, 3.8, 2.5, 3.8],
    "도서관 지박령형": [1.8, 2.2, 2.0, 1.2, 1.5],
    "실무 프로젝트형": [2.5, 4.8, 3.0, 4.2, 2.2],
    "무난한 가성비형": [3.0, 2.8, 2.2, 4.0, 3.2]
}

np.random.seed(42)
data = []

for label, center in centroids.items():
    for _ in range(30):  # 유형별 30명씩 생성
        # 중심점 기준 표준편차 0.4의 정규분포 노이즈 추가 후 1~5점 사이로 클리핑
        features = np.clip(np.random.normal(center, 0.4), 1.0, 5.0)
        # 소수점 첫째자리까지 반올림 (리커트 척도 느낌)
        features = np.round(features, 1)
        data.append(list(features) + [label])

df = pd.DataFrame(data, columns=['X1', 'X2', 'X3', 'X4', 'X5', 'type'])
df.to_csv('senior_data.csv', index=False, encoding='utf-8-sig')
print("가상 선배 데이터(senior_data.csv) 생성 완료!")