import sqlite3

# 새로운 데이터베이스 파일 생성
db_name = "new_feedback.db"  # 원하는 이름으로 변경 가능
conn = sqlite3.connect(db_name)
cursor = conn.cursor()

# 테이블 생성
cursor.execute("""
CREATE TABLE feedback (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    image_path TEXT NOT NULL,
    predicted_label TEXT NOT NULL,
    actual_label TEXT,
    is_correct INTEGER
);
""")

# 데이터베이스 변경사항 저장
conn.commit()
conn.close()

print(f"'{db_name}' 데이터베이스가 생성되고, 'feedback' 테이블이 초기화되었습니다.")
