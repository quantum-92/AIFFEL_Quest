# README

작성자 : 이혁희

-----
## 개요
이 문서는 MainQuest에 대한 설명 문서입니다.
코드와 결과는 '전설의 포켓몬 찾아 삼만리.ipynb'화일에 있습니다.

-----
## 학습 포인트 정리
코드를 실행해 보면서 생긴 질문은 '[질문]', 중요하다고 생각되는 것은 '[중요]'라고 표시해 놓았습니다.

-----
## 추가 분석
추가로 분석한 모델, feature engineering은 마지막 3개 패러그래프에 있습니다.

### [추가 분석 1. 다른 모델 적용]

- 학습에서 feature engineering한 데이터에 RandomForest, XGBoost 모델을 적용하였다.
- recall은 RandomForest < DecisitionTree < XGBoost 순.
    - 특히, XGBoost의 recall은 1이 나옴
- f1은 DecisionTree < RandomForest < XGBoost 순

|모델명     | DecisionTree | RandomForest | XGBoost |
|---------|--------------|----------|---------|
|accuracy | 0.96         | 0.97 | 0.97 |
|recall   | 0.92         | 0.85 | 1.00 |
|precision| 0.67         | 0.79 | 0.72 |
|f1.      | 0.77         | 0.81 | 0.84 |

### [추가분석 2: 수치데이터를 정규화하여 모델 적용]

- 수치 데이터를 MinMaxScaler를 이용하여 정규화하였다.
    - 수치 데이터 컬럼 : 'Total', 'HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed',
       'Generation', 'name_count'
- 그런데, 정규화하기 전과 똑같은 결과가 나왔다.
- RandomForest와 XGBoost는 정규화가 모델의 성능에 영향을 미치지 않는 것일까?

### [추가분석 3: 수치데이터에 로그를 적용하여 모델 적용]

- stat 데이터들의 histogram을 그려보면 데이터들이 왼쪽으로 치우쳐 있다.
- 데이터에 log를 씌워서 histogram을 다시 그려 보았더니, 오른쪽으로 치우침
- log를 씌우는 전처리가 별로 안좋을 것 같지만, 혹시나 해서 모델에 적용해 봄.
- 여기서도 똑같은 결과가 나왔다. 전처리를 잘못했나? 아니면, 분류모델에는 전처리가 의미없는 것인가? 텍스트에는 전처리가 의미 있다고 나오는데..


## 확인한 것
- 추가분석2, 추가분석3에서 추가분석 1과 정확히 같은 결과가 나왔습니다.
- 코딩 오류가 있나 해서 확인해 보았는데 오류를 찾지 못했습니다.
- 수치데이터를 MinMaxScale하거나 log변환하는 것이 분류 트리모델의 성능에 영향을 주지 못하는 것 같습니다.
- 다음 문서에 Feature Scaling이나 Normalization이 Decition Tree의 성능에 별 영향을 주지 못한다는 주장이 나옵니다.
https://forecastegy.com/posts/do-decision-trees-need-feature-scaling-or-normalization/#:~:text=In%20general%2C%20no.,as%20we'll%20see%20later.


# 회고
## 잘한 것
    - Quest를 완성했다.
## 잘못한 것
    - Tree 기반의 classifier에서 feature의 scaling, normalization이 학습 성능에 별 영향을 주지 못한다는 것을 배운 기억이 마지막쯤에 났다. 그런데, 여기서 그것을 하느라 시간을 허비했다.
## 배운 점
    - Tree 기반의 classifier에서 feature의 scaling, normalization이 학습 성능에 별 영향을 주지 못한다는 것을 확인했다.