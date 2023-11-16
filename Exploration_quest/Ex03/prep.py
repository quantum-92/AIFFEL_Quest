# importlib: 사용자가 파이썬의 import 시스템과 상호작용하기 위한 API를 제공하는 내장 라이브러리
# 사용자는 import 함수의 구현 내용과, 실행에 필요한 하위 함수들을 이용해 필요에 맞는 임포터를 직접 구현하거나 임포트 관련 기능들을 자유롭게 사용할 수 있음
# importlib.metadata: 설치된 패키지 메타 데이터에 대한 접근을 제공하는 라이브러리.
# 해당 코드 블럭에서는 importlib.metadata 안에 있는 version() 함수를 이용하여 pixellib 라이브러리의 버전을 확인

import os
import numpy as np
from importlib.metadata import version
import cv2
import pixellib
from pixellib.semantic import semantic_segmentation
from matplotlib import pyplot as plt

print("cv2 version: ", cv2.__version__)
print("pixellib version: ", version('pixellib'))



# 이미지 읽어 오기
def get_image(img_name):
    #img_path = os.getenv('HOME')+'/aiffel/human_segmentation/images/yj3.jpg'  
    img_path = os.getenv('HOME')+'/aiffel/human_segmentation/images/' + img_name
    img_orig = cv2.imread(img_path) 

    return img_orig


def build_model():
    model_dir = os.getenv('HOME')+'/aiffel/human_segmentation/models' 
    model_file = os.path.join(model_dir, 'deeplabv3_xception_tf_dim_ordering_tf_kernels.h5') 

    model = semantic_segmentation() #PixelLib 라이브러리 에서 가져온 클래스를 가져와서 semantic segmentation을 수행하는 클래스 인스턴스를 만듬
    model.load_pascalvoc_model(model_file) # pascal voc에 대해 훈련된 예외 모델(model_file)을 로드하는 함수를 호출
    
    return model
    
# label name에 맞는 colormap을 찾아서 return 한다.
# (색상순서 변경)
def get_seg_color(label_name):
    #pascalvoc 데이터의 라벨종류
    LABEL_NAMES = [
        'background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
        'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
        'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tv'
    ]
    # colormap 생성
    colormap = make_colormap()
    # label의 index
    label_index = LABEL_NAMES.index(label_name)
    # segmentation color map = label index의 colormap (색상순서 변경 RGB -> BGR)
    seg_color = colormap[label_index][::-1]
    
    return seg_color

#컬러맵 만들기 
def make_colormap():
    colormap = np.zeros((256, 3), dtype = int)
    ind = np.arange(256, dtype=int)

    for shift in reversed(range(8)):
        for channel in range(3):
            colormap[:, channel] |= ((ind >> channel) & 1) << shift
        ind >>= 3
        
    return colormap

