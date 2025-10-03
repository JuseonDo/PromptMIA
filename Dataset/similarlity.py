import json
import numpy as np
from gensim.models import KeyedVectors
import gensim.downloader as api
from sklearn.metrics.pairwise import cosine_similarity

# Word2Vec 모델 로드 (Google의 사전학습 모델 사용)
print("Word2Vec 모델 로딩 중...")
model = api.load("word2vec-google-news-300")
print("모델 로딩 완료!")

def text_to_vector(text, model):
    """텍스트를 Word2Vec 벡터로 변환 (단어 벡터들의 평균)"""
    words = text.lower().split()
    word_vectors = []
    
    for word in words:
        try:
            word_vectors.append(model[word])
        except KeyError:
            # 모델에 없는 단어는 건너뜀
            continue
    
    # if len(word_vectors) == 0:
    #     # 모든 단어가 모델에 없는 경우 0 벡터 반환
    #     return np.zeros(model.vector_size)
    
    return np.mean(word_vectors, axis=0)

def calculate_similarity(text1, text2, model):
    """두 텍스트 간의 코사인 유사도 계산"""
    vec1 = text_to_vector(text1, model).reshape(1, -1)
    vec2 = text_to_vector(text2, model).reshape(1, -1)
    similarity = cosine_similarity(vec1, vec2)[0][0]

    return float(similarity)

def process_json_file(input_file, output_file):
    """JSON 파일을 읽어 유사도를 계산하고 결과를 저장"""
    
    # JSON 파일 로드
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    results = []
    
    # 각 항목에 대해 유사도 계산
    for item in data:
        item_id = item['id']
        answer = item['answer']
        options = item['options']
        
        similarities = {
            'id': item_id,
            'answer_vs_option0': calculate_similarity(answer, options[0], model),
            'answer_vs_option1': calculate_similarity(answer, options[1], model),
            'answer_vs_option2': calculate_similarity(answer, options[2], model),
            'option0_vs_option1': calculate_similarity(options[0], options[1], model),
            'option1_vs_option2': calculate_similarity(options[1], options[2], model),
            'option0_vs_option2': calculate_similarity(options[0], options[2], model)
        }
        
        results.append(similarities)
        print(f"ID {item_id} 처리 완료")
    
    # 결과를 JSON 파일로 저장
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    
    print(f"\n결과가 '{output_file}'에 저장되었습니다.")
    
    return results

# 사용 예시
if __name__ == "__main__":
    input_file = 'Dataset/falcon_dataset_refinedweb.json'  # 입력 JSON 파일 경로
    output_file = 'Dataset/similarity_results.json'  # 출력 파일 경로

    results = process_json_file(input_file, output_file)
    
    # 결과 미리보기 (첫 번째 항목)
    if results:
        print("\n첫 번째 항목 결과:")
        print(json.dumps(results[0], indent=4))