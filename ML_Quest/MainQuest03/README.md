# README

- 작성자 : 이혁희

## 개요와 결과
- Test Accuracy : 90.85%
- 네트웍 정의와 파라미터
	- LeNet의 네트웍을 기본으로 하고 convolution layer를 하나 추가하였습니다.(10~20% 성능 향상)

  ```
	n_channel_1=16
	n_channel_2=32
	n_channel_3=64
	n_dense_1= 128
	n_dense_2 = 64
	n_train_epoch=20
	
	model=keras.models.Sequential()
	model.add(keras.layers.Conv2D(n_channel_1, (3,3), \
								  activation='relu', \
								  input_shape=(28,28,3)))
	model.add(keras.layers.MaxPool2D(2,2))
	model.add(keras.layers.Conv2D(n_channel_2, (3,3), activation='relu'))
	model.add(keras.layers.MaxPooling2D((2,2)))
	model.add(keras.layers.Conv2D(n_channel_3, (3,3), activation='relu'))
	model.add(keras.layers.MaxPooling2D((2,2)))
	
	model.add(keras.layers.Flatten())
	model.add(keras.layers.Dense(n_dense_1, activation='relu'))
	model.add(keras.layers.Dense(3, activation='softmax'))
  ```
- LeNet이 이미 좋은 모델이라 모델에 변화를 주는 것보다 다양한 이미지를 이용하여 학습하는 것이 성능향상 효과가 큰 듯 합니다.

## 시도한 것들
- 모델에 변화 주기
	- Node 수 변화
	- Layer 추가 삭제
	- 그 중 Convolution Layer를 추가 했을 때 가장 효과가 컸음
- 과최적화 대책
	- convolution layer에 L2 regularization
	- Dropout
	- BatchNormalization 등 시도 하였으나
	- 별 효과 없었음
- 학습/테스트 데이터
	- 그루들에게 받은 데이터로 학습해서 테스트한 결과 accuracy 50% 넘기기 힘들었다.
	- 사진 이미지를 열어 보니 눈으로도 인식하기 어려운 사진들이 많았다.
	- 나, 와이프, 딸의 사진을 새로 찍어서 학습과 테스트에 사용했다.
	- 학습용 이미지는 손을 든 상태에서 손목만 돌려서 손의 각도가 변하지 않게 조심
	- 테스트용  이미지는 정면에서만 찍음(정확도를 올리기 위한 꼼수)

##  알게 된 것
- 컨볼루션의 특징
	- 손의 각도가 변하면 인식을 잘 못한다.
		- 테스트 이미지의 손 각도가 옆으로 누워 있는 경우 정확도가 낮아지는 것을 확인.
		- 필터를 돌려서 인식하지는 않기 때문에 그런듯.
- 이미지의 품질이 중요하다.
	- 처음에 그루들이 생성한 이미지 2000장 정도로 학습했는데 테스트 정확도가 40% ~ 50% 정도 나옴
	- 그루들이 찍은 사진 중에 눈으로 판별이 어려운 이미지가 많았음.
		- 사진이 잘 찍히지 않은 그루의 사진은 학습에서 제외함.
		- 손의 각도가 다름.
- 학습 데이터는 다양하게 많이 모을 수록 좋다.
	- 내 사진을 class별로 500장씩 1500장을 찍어서 학습(좌우로만 돌림) -> 딸 사진(정면 사진)으로 테스트 60%까지 올라감.
	- 내 사진과 와이프 사진 3000장으로 학습 -> 딸 사진으로 90%대까지 올라감
	- 문제에 맞춰서 이미지를 생성해야 한다.
- 간단한 문제에는 간단한 네트웍이 좋다.
	- 이번 퀘스트에서 네트웍을 바꾼 것보다 이미지를 다양하게 모았을 때 test accuary가 훨씬 빠른 속도로 올라감
	- LeNet이 이미 상당히 좋은 네트웍이라 이것을 변형해서 오는 효과는 크지 않았음(10~20% 정도)
	- 비슷한 손 이미지 숫자를 늘여도 효과가 크지 않았음(내 손 이미지 1500장으로 학습해도 test accuracy는 크게 오르지 않음)
	- 다른 사람의 손이미지를 추가할 수록 성능이 급격하게 올라감.
	- 네트웍에 레이어를 추가하고 노드수를 늘이면 train loss는 빨리 낮아지는데 test accuracy는 더 안좋은 경우가 많았음.

KPT
- 잘 한점
	- 여러가지 시도는 했지만.. 없는 듯.
- 문제점
	- 이미지를 처음 만들다 보니 눈으로 봐도 손 모양을 알아 볼 수 없는 이미지를 만들었다.
- 배운점
	- CNN에서 이미지를 만들 때 문제의 목적에 맞게 만드는 것이 중요하다.
	- 다양한 이미지를 모으는 것이 성능 향상에 가장 중요한 것 같다.