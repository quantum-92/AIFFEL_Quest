## model function declaration
# import가 잘 안되서 여기로 옮겼음. 왜 안되지? ㅠㅠ

from keras.applications.vgg16 import VGG16, preprocess_input
from keras.preprocessing.image import img_to_array, load_img
from keras.models import Model
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import os
import pickle
import pandas as pd
import datetime

# fastapi 실행함수에 데이터를 전달하기 위한 클래스
from fastapi import FastAPI, Depends
from typing import Dict

def load_model_info():
    # VGG16 모델 로드 (마지막 분류 레이어 제외)
    model = VGG16(weights='imagenet', include_top=False)
    model = Model(inputs=model.input, outputs=model.output)

    # .npy 파일로부터 배열 로드
    feature_list = np.load('feature_list.npy')
    
    # .pkl 파일로부터 객체 로드
    with open('image_files.pkl', 'rb') as f:
        image_files = pickle.load(f)
    
    # CSV 파일에서 DataFrame 로드
    chart_df = pd.read_csv('chart_data.csv')
    
    # 'last_date' 컬럼을 리스트로 변환
    last_dates = chart_df['last_date'].tolist()
    
    # 'label' 컬럼을 리스트로 변환
    labels = chart_df['label'].tolist()
    
    return ({"model": model,
            "feature_list": feature_list,
            "image_files": image_files,
            "labels": labels,
            "last_dates": last_dates
            })

def extract_features(img_path, model):
    img = load_img(img_path, target_size=(224, 224))
    img_array = img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)
    features = model.predict(preprocessed_img)
    flattened_features = features.flatten()
    normalized_features = flattened_features / np.linalg.norm(flattened_features)
    return normalized_features    

def date_to_timestamp(query_date):
     return query_date + " 00:00:00-05:00"
 
def date_to_datetime(date_str):
    return datetime.strptime(date_str, '%Y-%m-%d')
   
import os
from typing import List, Dict

def get_similar_images_info_by_date(query_date, 
                                    feature_list, 
                                    image_files,  # 이미지 파일명 목록
                                    labels, 
                                    last_dates, 
                                    model, 
                                    top_n=5) -> List[Dict]:
    print("0...")
    image_dir = "images"
    # 날짜에 해당하는 이미지 파일 찾기
    try:
        img_index = last_dates.index(query_date)
        query_img_path = os.path.join(image_dir, image_files[img_index])
    except ValueError:
        print(f"No image found for date: {query_date}")
        return []

    print(len(features_list))
    # 유사도 계산 및 정렬
    query_features = extract_features(query_img_path, model)
    print("1...")
    similarities = cosine_similarity([query_features], feature_list)
    print("2...")
    similar_indices = np.argsort(similarities[0])[::-1][:top_n]
    print("3...")
    similar_images_info = []
    for idx in similar_indices:
        img_path = os.path.join(image_dir, image_files[idx])
        with open(img_path, "rb") as img_file:
            img_data = img_file.read()

        img_info = {
            'image': img_data,  # 이미지 파일 내용
            'last_date': last_dates[idx],
            'label': labels[idx]
        }
        similar_images_info.append(img_info)

    return similar_images_info

def get_best_similar_images_info_by_date(query_date, feature_list, image_files, labels, last_dates, model, top_n=5, days_diff=5):
    image_dir = "images"
    try:
        img_index = last_dates.index(query_date)
        query_img_path = os.path.join(image_dir, image_files[img_index])
    except ValueError:
        print(f"No image found for date: {query_date}")
        return []

    query_features = extract_features(query_img_path, model)
    similarities = cosine_similarity([query_features], feature_list)
    similar_indices = np.argsort(similarities[0])[::-1]

    similar_images_info = []
    query_date_dt = date_to_datetime(query_date)

    for idx in similar_indices:
        img_date = date_to_datetime(last_dates[idx])
        if abs((query_date_dt - img_date).days) >= days_diff:
            img_info = {
                'filename': image_files[idx],
                'last_date': last_dates[idx],
                'label': labels[idx]
            }
            similar_images_info.append(img_info)
            if len(similar_images_info) >= top_n:
                break

    return similar_images_info    

### fast api server ###
# server_fastapi_vgg16.py
import uvicorn   # pip install uvicorn 
from fastapi import FastAPI   # pip install fastapi 
from fastapi.middleware.cors import CORSMiddleware # 추가된부분 cors 문제 해결을 위한

from pydantic import BaseModel
import base64
from typing import List

class ImageInfo(BaseModel):
    image: str  # Base64 인코딩된 이미지
    last_date: str
    label: str


# 공유데이터 사용을 위한 클래스 정의
# 공유 데이터를 위한 클래스
class CommonData:
    def __init__(self, feature_list, image_files, labels, last_dates, model):
        self.feature_list = feature_list
        self.image_files = image_files
        self.labels = labels
        self.last_dates = last_dates
        self.model = model

# # 의존성 주입을 위한 함수
# def get_common_data():
#     return CommonData(feature_list, image_files, labels, last_dates, model)

# Create the FastAPI application
app = FastAPI()

# cors 이슈
origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# A simple example of a GET request
@app.get("/")
async def read_root():
    print("url was requested")
    return "차트 모양이 비슷한 날짜의 차트를 찾아서 돌려 주는 서비스입니다."

@app.get("/get-similar-images/{query_date}",
         response_model=List[ImageInfo])
async def get_similar_images(query_date: str,
             common_data: CommonData = Depends()):
    print(query_date)
    #print(len(common_data))
    similar_images_info = get_similar_images_info_by_date(
        query_date, 
        common_data.feature_list, 
        common_data.image_files, 
        common_data.labels, 
        common_data.last_dates, 
        common_data.model, 
        top_n=10)
    
    img_info = similar_images_info[3]
    base64_image = base64.b64encode(img_info['image']).decode()
    response = {
        'image': base64_image,
        'last_date': img_info['last_date'],
        'label': img_info['label']
    }

    return response

# async def get_similar_chart():
#     query_date = date_to_timestamp(query_date)
#     similar_images_info = get_similar_images_info_by_date(query_date, 
#                                         feature_list, 
#                                         image_files, 
#                                         labels, 
#                                         last_dates, 
#                                         model, 
#                                         top_n=10)
#     similar_images_info = similar_images_info 
#     return similar_images_info

@app.get('/chart_image/{query_date}')
async def get_chart_image():
    result = await vgg16_prediction_model.prediction_model()
    return result



#app = FastAPI()


# 예시 엔드포인트
@app.get("/example")
async def example_endpoint(common_data: CommonData = Depends()):
    # 여기에서 common_data를 사용합니다.
    pass
# Run the server

if __name__ == "__main__":
    
    model_info = load_model_info()

    feature_list = model_info["feature_list"]
    image_files = model_info["image_files"]
    labels = model_info["labels"]
    last_dates = model_info["last_dates"]
    model = model_info["model"]

    # 공유 데이터를 담은 인스턴스 생성
    common_data = CommonData(feature_list, image_files, labels, last_dates, model)

    # 의존성 주입 함수에 공유 데이터 인스턴스를 전달
    get_common_data = lambda: common_data

    uvicorn.run("server_fastapi:app",
            reload= True,   # Reload the server when code changes
            host="127.0.0.1",   # Listen on localhost 
            port=5000,   # Listen on port 5000 
            log_level="info"   # Log level
            )