# README

- 작성자 : 이혁희

<<<<<<< HEAD
### 본 README 화일에는 이미지에 대한 링크가 끊어져 있습니다. '10. 딥러닝 프로젝트.jpynb'화일의 마지막 셀이 README가 있으면 참고하세요.

## 개요
- 다음과 같이 4개의 모델을 분석 하였습니다.(우측에 순서대로 테스트 loss, metric)
=======
### 본 README 파일은 이미지에 대한 링크가 깨져 있습니다. 코드 파일의 마지막 셀에 README가 있으니 참고 하세요.

## 개요
- 다음과 같이 4개의 모델을 분석 하였습니다.(우측에 순서대로 loss, metric)
    - Boston 주택가격모델 : [29.61016082763672, 4.254533767700195]
    - Reuters 딥러닝 모델 : [1.0222852230072021, 0.7943009734153748]
    - CIFAR10 딥러닝 모델 : [1.4097532033920288, 0.4918999969959259]
    - CIFAR10 CNN 모델 : [0.8842432498931885, 0.7063000202178955]
        - (CIFAR10 CNN 모델은 딥러닝 모델의 예측력이 너무 약해서 CNN 모델은 어떨까 싶어서 분석해 보았습니다.)

- 학습 방법에 대하여
    - 과적합 방지를 위한 Dropout의 효과는 잘 모르겠지만, BatchNormalization을 했을 때 학습속도가 빨라지는 것은 확인할 수 있었습니다.
    - BatchNormalization과 EarlyStopping을 이용해서 학습시 과적합 되기전에 학습을 중단하는 것이 매우 유용했습니다.

- 학습 결과에 대하여
    - Boston 주택가격과 Reuter 모델은 DNN을 썼고 데이터 양이 그렇게 많지 않은 것에 비하면 예측력이 매우 인상적이었습니다.
    - CIFAR10 데이터를 딥러닝으로 분석했을 때는 예측력이 없었지만, CNN으로 학습했을 때는 70%의 테스트 정확도를 보였습니다.(LeNet 구성을 거의 그대로 사용) CNN이 이미지를 인식하는데 매우 유리함을 알았습니다.



## Boston 주택가격 예측 모델
1. 모델 서머리

```
Model: "sequential_1"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
dense_3 (Dense)              (None, 64)                896       
_________________________________________________________________
dense_4 (Dense)              (None, 32)                2080      
_________________________________________________________________
dense_5 (Dense)              (None, 1)                 33        
=================================================================
Total params: 3,009
Trainable params: 3,009
Non-trainable params: 0
```
2. loss, accuracy history
![image.png](attachment:image.png)

3. 주택 예측과 레이블 비교
    - 주택 예측은 회귀 모델이라 MSE을 메트릭으로 선택하였습니다. 직관적으로 예측의 정확도를 확인하기 위해서 레이블과 예측치를 x, y축에 놓고 scatter plot를 하였습니다.
    - 둘 간에 우상향하는 선형 상관관계를 보이고 있습니다.
![image-2.png](attachment:image-2.png)


## Reuters 딥러닝 모델
```
Model: "sequential_3"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
dense_8 (Dense)              (None, 128)               1280128   
_________________________________________________________________
batch_normalization_1 (Batch (None, 128)               512       
_________________________________________________________________
activation_1 (Activation)    (None, 128)               0         
_________________________________________________________________
dense_9 (Dense)              (None, 128)               16512     
_________________________________________________________________
dropout_1 (Dropout)          (None, 128)               0         
_________________________________________________________________
dense_10 (Dense)             (None, 46)                5934      
=================================================================
Total params: 1,303,086
Trainable params: 1,302,830
Non-trainable params: 256
```
![image-3.png](attachment:image-3.png)

## CIFAR10 딥러닝 모델
```
Model: "sequential_4"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
dense_11 (Dense)             (None, 2048)              6293504   
_________________________________________________________________
batch_normalization_2 (Batch (None, 2048)              8192      
_________________________________________________________________
activation_2 (Activation)    (None, 2048)              0         
_________________________________________________________________
dropout_2 (Dropout)          (None, 2048)              0         
_________________________________________________________________
dense_12 (Dense)             (None, 1024)              2098176   
_________________________________________________________________
batch_normalization_3 (Batch (None, 1024)              4096      
_________________________________________________________________
activation_3 (Activation)    (None, 1024)              0         
_________________________________________________________________
dropout_3 (Dropout)          (None, 1024)              0         
_________________________________________________________________
dense_13 (Dense)             (None, 512)               524800    
_________________________________________________________________
batch_normalization_4 (Batch (None, 512)               2048      
_________________________________________________________________
activation_4 (Activation)    (None, 512)               0         
_________________________________________________________________
dropout_4 (Dropout)          (None, 512)               0         
_________________________________________________________________
dense_14 (Dense)             (None, 10)                5130      
=================================================================
Total params: 8,935,946
Trainable params: 8,928,778
Non-trainable params: 7,168
```
![image-4.png](attachment:image-4.png)

## CNN을 이용한 CIFAR10 분석
```
Model: "sequential_7"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d_6 (Conv2D)            (None, 30, 30, 64)        1792      
_________________________________________________________________
max_pooling2d_6 (MaxPooling2 (None, 15, 15, 64)        0         
_________________________________________________________________
conv2d_7 (Conv2D)            (None, 13, 13, 128)       73856     
_________________________________________________________________
max_pooling2d_7 (MaxPooling2 (None, 6, 6, 128)         0         
_________________________________________________________________
conv2d_8 (Conv2D)            (None, 4, 4, 128)         147584    
_________________________________________________________________
max_pooling2d_8 (MaxPooling2 (None, 2, 2, 128)         0         
_________________________________________________________________
flatten_2 (Flatten)          (None, 512)               0         
_________________________________________________________________
dense_21 (Dense)             (None, 128)               65664     
_________________________________________________________________
batch_normalization_7 (Batch (None, 128)               512       
_________________________________________________________________
activation_7 (Activation)    (None, 128)               0         
_________________________________________________________________
dense_22 (Dense)             (None, 64)                8256      
_________________________________________________________________
batch_normalization_8 (Batch (None, 64)                256       
_________________________________________________________________
activation_8 (Activation)    (None, 64)                0         
_________________________________________________________________
dropout_6 (Dropout)          (None, 64)                0         
_________________________________________________________________
dense_23 (Dense)             (None, 10)                650       
=================================================================
Total params: 298,570
Trainable params: 298,186
Non-trainable params: 384
```
![image-5.png](attachment:image-5.png)
