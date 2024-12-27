import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
import cv2
import matplotlib.pyplot as plt  # matplotlib 추가

# CIFAR-10 클래스 이름
class_names = ['비행기', '자동차', '새', '고양이', '사슴', '개', '개구리', '말', '배', '트럭']

# 모델 불러오기 (compile=False 추가)
model = load_model('vgg_cifar10_model.keras', compile=False)

# 이미지 전처리 함수
def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (32, 32))
    img = img.astype('float32') / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# 모델 재학습 함수
def retrain_with_label(image_path, label):
    # 이미지 전처리
    img = preprocess_image(image_path)

    # 사용자 라벨을 원핫 인코딩
    label_index = class_names.index(label)
    label_one_hot = to_categorical([label_index], num_classes=10)

    # 모델 컴파일 (옵티마이저와 손실 함수 설정)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # 모델 재학습
    model.fit(img, label_one_hot, epochs=5, batch_size=1)

    # 재학습된 모델 저장
    model.save('vgg_cifar10_model.keras')
    print("모델이 재학습되었고 저장되었습니다.")

# 메인 실행
if __name__ == "__main__":
    # 사용자로부터 파일 이름 입력받기
    print("static/uploads 폴더에 있는 이미지 파일 이름을 입력하세요.")
    file_name = input("이미지 파일 이름: ").strip()
    image_path = os.path.join("./static/uploads", file_name)

    # 파일 존재 여부 확인
    if not os.path.isfile(image_path):
        print(f"파일 '{file_name}'을 찾을 수 없습니다. 프로그램을 종료합니다.")
    else:
        print(f"이미지 경로: {image_path}")
        
        # 이미지 출력 (matplotlib 사용)
        img = cv2.imread(image_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # BGR에서 RGB로 변환
        
        # 이미지를 matplotlib으로 표시
        plt.imshow(img_rgb)
        plt.axis('off')  # 축을 표시하지 않음
        plt.show()

        # 사용자로부터 라벨 입력받기
        print(f"가능한 클래스: {class_names}")
        label = input("이미지의 실제 클래스를 입력하세요: ").strip()

        # 입력된 라벨 검증
        if label not in class_names:
            print("잘못된 라벨입니다. 프로그램을 종료합니다.")
        else:
            retrain_with_label(image_path, label)
