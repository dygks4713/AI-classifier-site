from flask import Flask, request, render_template, redirect
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import os

# Flask 애플리케이션 생성
app = Flask(__name__)

# CIFAR-10 모델 로드
model = load_model('vgg_cifar10_model.keras', compile=False)
class_names = ['비행기', '자동차', '새', '고양이', '사슴', '개', '개구리', '말', '배', '트럭']

# 업로드 폴더 설정
UPLOAD_FOLDER = 'static/uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# 이미지 전처리 함수
def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (32, 32))
    img = img.astype('float32') / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# 라우트: 메인 페이지
@app.route('/')
def index():
    return render_template('upload.html')  # 업로드 페이지 렌더링

# 라우트: 이미지 업로드 및 예측
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file uploaded", 400

    file = request.files['file']
    if file.filename == '':
        return "No file selected", 400

    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(file_path)


    # 이미지 전처리 및 모델 예측
    img = preprocess_image(file_path)
    predictions = model.predict(img)
    class_index = np.argmax(predictions)
    confidence = np.max(predictions) * 100

    return render_template(
        'result.html',
        image=f"uploads/{file.filename}",
        content=f"Image file: {file.filename}",
        prediction=class_names[class_index],
        probability=f"{confidence:.2f}"
    )

if __name__ == '__main__':
    app.run(debug=True)
