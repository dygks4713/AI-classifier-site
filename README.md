# AI-classifier-site
Train the model using the cifar-10 dataset and classify 10 types of images.

# file structure
```
AI-CLASSIFIER-SITE/
├── static/
│   ├── uploads/          # 업로드된 이미지 저장 폴더
│   └── style.css         # 스타일 시트
├── templates/            # HTML 템플릿 파일
│   ├── _formhelpers.html
│   ├── result.html
│   └── upload.html
├── venv/                 # 가상 환경 파일
├── .gitattributes        # Git 속성 파일
├── app.py                # Flask 애플리케이션 메인 스크립트
├── cifar_10_VGGnet.ipynb # CIFAR-10 모델 학습 Jupyter Notebook
├── README.md             # 프로젝트 설명 파일
├── requirements.txt      # 필요한 패키지 목록
├── retrain_model.py      # 모델 재학습 스크립트
├── sqlite.py             # SQLite 데이터베이스 관리 스크립트
└── vgg_cifar10_model.keras # 사전 학습된 모델 파일
```

# 필요한 라이브러리 설치
```bash
pip install -r requirements.txt
```

# 웹 애플리케이션 실행
- app.py 실행
```bash
python app.py
```
- 브라우저에서 http://127.0.0.1:5000 접속

# 사이트에서 확인
![이미지 분류](https://github.com/user-attachments/assets/158cb237-d6a0-40e1-be86-873a55762b98)

# 사용자가 업로드한 이미지 데이터를 재학습 시키기
- retrain_model.py 실행
![이미지 재학습](https://github.com/user-attachments/assets/eddaf3c8-d752-4ef9-81c5-05a7292ef825)
