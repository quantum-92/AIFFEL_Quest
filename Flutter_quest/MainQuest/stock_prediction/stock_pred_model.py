# vgg16_prediction_model.py

# 예측에 필요한 라이브러리
# from tensorflow.keras.applications.vgg16 import preprocess_input
# import tensorflow as tf
# from PIL import Image
# import numpy as np
# import matplotlib.pyplot as plt
# from tensorflow.keras.preprocessing import image
# from tensorflow.keras.applications.imagenet_utils import decode_predictions



from keras.applications.vgg16 import VGG16, preprocess_input
from keras.preprocessing.image import img_to_array, load_img
from keras.models import Model
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import os
import pickle
import pandas as pd

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

from datetime import datetime, timedelta

def date_to_datetime(date_str):
    return datetime.strptime(date_str, '%Y-%m-%d  00:00:00-05:00')

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
def get_similar_images_info_by_date(query_date, 
                                    feature_list, 
                                    image_files, 
                                    labels, 
                                    last_dates, 
                                    model, 
                                    top_n=5):
    image_dir = "images"
    # 날짜에 해당하는 이미지 파일 찾기
    try:
        #print(query_date)
        #print(last_dates[-5:])
        img_index = last_dates.index(query_date)
        #print(img_index)

        query_img_path = os.path.join(image_dir, image_files[img_index])
    except ValueError:
        print(f"No image found for date: {query_date}")
        return []

    # 나머지는 원래 함수와 동일
    query_features = extract_features(query_img_path, model)
    similarities = cosine_similarity([query_features], feature_list)
    similar_indices = np.argsort(similarities[0])[::-1][:top_n]

    similar_images_info = []
    for idx in similar_indices:
        img_info = {
            'filename': image_files[idx],
            'last_date': last_dates[idx],
            'label': labels[idx]
        }
        similar_images_info.append(img_info)

    return similar_images_info

if __name__ == "__main__":
    model_info = load_model_info()

    query_date = date_to_timestamp('2021-12-27')
    
    feature_list = model_info["feature_list"]
    image_files = model_info["image_files"]
    labels = model_info["labels"]
    last_dates = model_info["last_dates"]
    model = model_info["model"]

    similar_images_info = get_similar_images_info_by_date(query_date, 
                                        feature_list, 
                                        image_files, 
                                        labels, 
                                        last_dates, 
                                        model, 
                                        top_n=5)

    print(last_dates[:5])
    # 결과 출력 (예시)
    for img_info in similar_images_info:
        print(f"Filename: {img_info['filename']}, Last Date: {img_info['last_date']}, Label: {img_info['label']}")

