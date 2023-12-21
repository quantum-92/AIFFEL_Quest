# README.md

- 해파리 inference 모델은 VGG16을 그대로 사용하였습니다.
- 파일 설명
    - main.dart : 모바일 앱을 구현
    - server_fastapi_vgg16.py : fastAPI 서버 구현
    - vgg16_prediction_model.py : VGG16 prediction 모델 구현
        - DLthon에서 구현한 ResNet모델을 재학습하는 과정에서 커널이 죽어서 VGG16 모델을 학습모델로 했습니다.
- 결과
    - prediction_label.png : "예측결과" 버튼을 눌렀을 때 결과 화면
    - prediction_prob.png : "예측확률" 버튼을 눌렀을 때 결과 화면
    - debug.png : console 화면


## 회고
### 잘한 점
- fast api를 통하여 서버의 서비스를 모바일에서 보여주는 서비스를 구현하였다.
### 문제점
- DLthon에서 학습한 모델을 서비스 모델로 사용하지 못하였다. 
### 배운 점
- 모델학습할 때 학습된 모델을 저장하는 습관을 들이자.