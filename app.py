import mysql.connector
from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv
import numpy as np
import os

app = Flask(__name__)
load_dotenv()

# MySQL 연결 정보
DB_CONFIG = {
    "host": os.getenv("DB_HOST"),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
    "database": os.getenv("DB_NAME")
}

class EmotionAnalyzer:
    _instance = None
    _is_initialized = False

    # 싱글톤 패턴
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(EmotionAnalyzer, cls).__new__(cls)
        return cls._instance

    # 초기화
    def __init__(self):
        # 이미 초기화된 경우 다시 하지 않음 (=DB 목록들 임베딩은 한번만)
        if EmotionAnalyzer._is_initialized:
            return

        # 감정 분석 모델
        self.model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')
        # DB에서 감정 목록 가져오기
        self.emotions = self._get_emotions_from_db()
        # 감정들의 임베딩 계산
        if self.emotions:
            self.emotion_embeddings = self.model.encode(self.emotions)
        else:
            self.emotion_embeddings = None
            
        EmotionAnalyzer._is_initialized = True

    # 데이터베이스에서 감정 목록 가져오기
    def _get_emotions_from_db(self):
        try:
            with mysql.connector.connect(**DB_CONFIG) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT name FROM keyword")
                return [row[0] for row in cursor.fetchall()]
        except Exception as e:
            print(f"DB 오류: {e}")
            return []

    # 입력된 텍스트와 가장 유사한 감정 찾기
    def find_similar_emotion(self, input_text, used_emotions):
        # 기본 체크
        if not input_text or not self.emotions or self.emotion_embeddings is None or len(self.emotion_embeddings) == 0:
            return None, 0

        try:
            # 입력 텍스트의 임베딩 계산
            text_embedding = self.model.encode([input_text])
            
            # 유사도 계산
            similarities = cosine_similarity(text_embedding, self.emotion_embeddings)[0]
            
            # NumPy array를 파이썬 리스트로 변환 (JSON 직렬화를 위해)
            similarities = similarities.tolist()
            
            # 이미 사용된(= 원래 DB 존재했던 or 이전 유사도 분석에서 선택된) 감정은 제외
            for i, emotion in enumerate(self.emotions):
                if emotion in used_emotions:
                    similarities[i] = -1
            
            # 가장 유사한 감정 찾기
            best_match_index = np.argmax(similarities)
            
            # float32를 일반 float으로 변환 (JSON 직렬화를 위해)
            similarity_score = float(similarities[best_match_index])
            
            return (
                self.emotions[best_match_index],
                round(similarity_score * 100, 2)
            )
            
        except Exception as e:
            print(f"유사도 분석 중 오류 발생: {e}")
            return None, 0

# 앱 시작 시, 감정 분석기 초기화
emotion_analyzer = EmotionAnalyzer()

@app.route('/analyze_emotions', methods=['POST'])
def analyze_emotions():
    # 감정 분석 API 엔드포인트
    try:
        # request
        data = request.get_json()
        if not data:
            return jsonify({"error": "데이터가 없습니다"}), 400

        # 입력 감정과 이미 사용된(= 원래 DB 존재했던) 감정 가져오기
        input_emotions = data.get('emotions', [])
        used_emotions = set(data.get('existingEmotions', []))

        # 입력 감정이 없으면 에러
        if not input_emotions:
            return jsonify({"error": "분석할 감정이 없습니다"}), 400

        results = []

        # 각 감정 유사도 분석
        for emotion in input_emotions:
            similar_emotion, score = emotion_analyzer.find_similar_emotion(
                emotion, used_emotions
            )
            
            if similar_emotion:
                results.append({
                    "inputEmotion": emotion,  # 입력 감정
                    "similarEmotion": similar_emotion,  # 유사한 감정
                    "similarityScore": score  # 유사도
                })
                used_emotions.add(similar_emotion)  # 유사도 분석이 여러 번인 경우에도 중복되지 않도록

        return jsonify({"analyzedEmotions": results})

    except Exception as e:
        print(f"Error occurred: {str(e)}")  # 디버깅을 위한 로그 추가
        return jsonify({"error": f"분석 중 오류 발생: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000)