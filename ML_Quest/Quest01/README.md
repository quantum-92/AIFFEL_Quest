# 머신러닝 프로젝트

작성자 : 이혁희

1. '머신러닝_기초_노드10_프로젝트성능_향상_Tip.ipynb'를 변형하였습니다.
2. LabelEncoding된 범주형 데이터 컬럼값을 MinMaxScaler를 통해서 다시 표준화하였습니다.
3. ElasticNet모델을 추가하였습니다.
    - evaluation data에서 ElasticNet의 MSE가 가장 낮았습니다.
    - alpha = 0.1일 때 MSE = 11756.150907432313
    - 하지만, 최종 데이터에 적용해 보니 52000을 넘는 값이 나왔습니다.
4. xgboost모델을 적용해서 최종 42778.28477706157이 나왔습니다.
    - LabelEncoding된 값을 다시 표준화한 것이 의미가 없었습니다.