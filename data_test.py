import json
import re
from typing import List
from datasets import Dataset, load_dataset

# streaming으로 로드
dataset = load_dataset(
    "allenai/c4",
    "en",
    split="train",
    streaming=True
)

# # 처음 50개만 가져오기
# for i, example in enumerate(dataset):
#     if i >= 50:
#         break
#     print(example['content'])
#     print("\n---\n")


def collect_dataset(dataset, num_samples: int = 50, output_path: str = "untrained_dataset_collected.json"):
    """
    데이터셋에서 샘플 수집하고 저장
    """
    samples = []
    skipped = 0
    
    print(f"데이터셋에서 {num_samples}개 수집 중...")
    for i, example in enumerate(dataset):
        if len(samples) >= num_samples:
            break
        
        content = example['text'].strip()
        clean_paragraphs = extract_clean_paragraphs(content)
        
        # 깨끗한 문단이 없으면 건너뛰기
        if not clean_paragraphs:
            skipped += 1
            continue
        
        samples.append({
            "id": i,
            "content": content,
            "clean_paragraphs": clean_paragraphs
        })
    
    # JSON 저장
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(samples, f, ensure_ascii=False, indent=2)
    
    print(f"✓ {len(samples)}개 저장 완료 (건너뜀: {skipped}개): {output_path}")
    return samples


def extract_clean_paragraphs(text: str) -> List[str]:
    """MLM 실험에 적합한 깨끗한 문단 추출 (900-1000자)"""
    
    # 문단으로 분리 (빈 줄 기준)
    paragraphs = text.split('\n\n')
    
    clean = []
    for para in paragraphs:
        para = para.strip()
        
        # 문단 기준: 900-1000자, 최소 3문장, 최소 120단어
        if (900 <= len(para) <= 1000 and 
            para.count('.') >= 3 and
            para.count(' ') >= 120 and
            re.search(r'[a-zA-Z]', para) and
            not re.search(r'[@#$%^&*(){}\[\]]', para)):
            clean.append(para)  
    return clean


def get_all_paragraphs(samples: List[dict]) -> List[str]:
    """모든 샘플에서 깨끗한 문단들만 추출"""
    all_paragraphs = []
    for sample in samples:
        all_paragraphs.extend(sample['clean_paragraphs'])
    return all_paragraphs


def save_paragraphs(paragraphs: List[str], output_path: str = "clean_paragraphs.txt"):
    """문단 리스트를 텍스트 파일로 저장"""
    with open(output_path, 'w', encoding='utf-8') as f:
        for i, para in enumerate(paragraphs, 1):
            f.write(f"=== 문단 {i} ({len(para)} chars) ===\n")
            f.write(f"{para}\n\n")
    print(f"✓ {len(paragraphs)}개 문단 저장: {output_path}")


# 사용법
if __name__ == "__main__":
    # 1. 데이터셋 수집
    samples = collect_dataset(dataset, num_samples=50)
    
    # 2. 깨끗한 문단들 추출 (900-1000자)
    paragraphs = get_all_paragraphs(samples)
    
    # 3. 문단 저장
    save_paragraphs(paragraphs)
    
    # 4. MLM 실험용 문단 선택
    if len(paragraphs) > 0:
        import random
        experiment_text = random.choice(paragraphs)
        print(f"\n실험용 문단 ({len(experiment_text)} chars):")
        print(experiment_text[:200] + "...")