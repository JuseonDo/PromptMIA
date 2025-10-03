"""
Passage별 정답 위치 편향 종합 분석
같은 passage에 대해 정답을 A, B, C, D에 배치한 4개의 결과를 분석
+ 각 quiz별 A, B, C, D 선택 비율 추가
"""

import json
import math
from pathlib import Path
from typing import List, Dict, Tuple
from scipy.stats import fisher_exact
import pandas as pd
import numpy as np


# ============================================
# 1. 단일 Quiz 분석
# ============================================

def analyze_quiz(quiz_data: dict) -> dict:
    """단일 quiz 데이터 분석 + 위치별 선택 비율"""
    correct_pos = quiz_data['correct_position']
    predictions = quiz_data['predictions']
    total = len(predictions)
    
    # 각 위치별 선택 횟수 계산
    position_counts = {'A': 0, 'B': 0, 'C': 0, 'D': 0}
    for pred in predictions:
        if pred in position_counts:
            position_counts[pred] += 1
    
    # 비율로 변환
    position_rates = {pos: count / total * 100 for pos, count in position_counts.items()}
    
    correct_pos_count = predictions.count(correct_pos)
    other_pos_count = total - correct_pos_count
    
    return {
        'quiz_id': quiz_data['quiz_id'],
        'correct_position': correct_pos,
        'total_predictions': total,
        'correct_pos_selected': correct_pos_count,
        'other_pos_selected': other_pos_count,
        'correct_pos_rate': correct_pos_count / total,
        'position_selection_rates': position_rates,  # 새로 추가
        'position_selection_counts': position_counts  # 새로 추가
    }


# ============================================
# 2. Passage별 분석 (A, B, C, D 4개 quiz 통합)
# ============================================

def analyze_passage(quizzes: List[dict]) -> dict:
    """
    같은 passage에 대한 4개의 quiz 데이터 분석
    
    Args:
        quizzes: 같은 passage에 대한 4개의 quiz (정답이 A, B, C, D인 경우)
    
    Returns:
        분석 결과 딕셔너리
    """
    if len(quizzes) != 4:
        print(f"Warning: Expected 4 quizzes per passage, got {len(quizzes)}")
    
    passage_id = quizzes[0].get('passage_id', 'unknown')
    
    results = {
        'passage_id': passage_id,
        'quizzes': [],
        'position_analysis': {},
        'significant_positions': [],
        'overall_significant': False,
        'average_selection_rates': {}  # 새로 추가: passage 전체의 평균 선택 비율
    }
    
    positions = ['A', 'B', 'C', 'D']
    
    # 각 위치별 평균 선택 비율 계산을 위한 누적
    total_selection_rates = {'A': [], 'B': [], 'C': [], 'D': []}
    
    for position in positions:
        # 2x2 contingency table 구성
        correct_and_choose = 0      # 정답이 position이고 position 선택
        correct_and_not_choose = 0  # 정답이 position이고 position 선택 안함
        incorrect_and_choose = 0    # 정답이 position이 아니고 position 선택
        incorrect_and_not_choose = 0 # 정답이 position이 아니고 position 선택 안함
        
        for quiz in quizzes:
            is_correct = quiz['correct_position'] == position
            
            for pred in quiz['predictions']:
                chooses = pred == position
                
                if is_correct and chooses:
                    correct_and_choose += 1
                elif is_correct and not chooses:
                    correct_and_not_choose += 1
                elif not is_correct and chooses:
                    incorrect_and_choose += 1
                else:
                    incorrect_and_not_choose += 1
        
        # Fisher's Exact Test 수행
        contingency_table = [
            [correct_and_choose, incorrect_and_choose],
            [correct_and_not_choose, incorrect_and_not_choose]
        ]
        
        try:
            _, p_value = fisher_exact(contingency_table, alternative='two-sided')
        except:
            p_value = 1.0
        
        total_correct = correct_and_choose + correct_and_not_choose
        total_incorrect = incorrect_and_choose + incorrect_and_not_choose
        
        rate_when_correct = correct_and_choose / total_correct if total_correct > 0 else 0
        rate_when_incorrect = incorrect_and_choose / total_incorrect if total_incorrect > 0 else 0
        
        results['position_analysis'][position] = {
            'contingency_table': {
                'correct_and_choose': correct_and_choose,
                'correct_and_not_choose': correct_and_not_choose,
                'incorrect_and_choose': incorrect_and_choose,
                'incorrect_and_not_choose': incorrect_and_not_choose
            },
            'rate_when_correct': rate_when_correct,
            'rate_when_incorrect': rate_when_incorrect,
            'rate_difference': rate_when_correct - rate_when_incorrect,  # 추가: 비율 차이
            'count_when_correct': correct_and_choose,
            'total_when_correct': total_correct,
            'count_when_incorrect': incorrect_and_choose,
            'total_when_incorrect': total_incorrect,
            'p_value': p_value,
            'significant': p_value < 0.05
        }
        
        if p_value < 0.05:
            results['significant_positions'].append(position)
    
    # 개별 quiz 정보 저장 및 선택 비율 누적
    for quiz in quizzes:
        quiz_analysis = analyze_quiz(quiz)
        results['quizzes'].append(quiz_analysis)
        
        # 각 위치별 선택 비율 누적
        for pos in positions:
            total_selection_rates[pos].append(quiz_analysis['position_selection_rates'][pos])
    
    # passage 전체의 평균 선택 비율 계산
    for pos in positions:
        results['average_selection_rates'][pos] = np.mean(total_selection_rates[pos])
    
    # 전체 passage가 유의한지 판단 (절반 이상의 위치에서 유의)
    results['overall_significant'] = len(results['significant_positions']) >= 2
    
    return results


# ============================================
# 3. 전체 데이터셋 분석
# ============================================

def analyze_all_passages(passages: Dict[str, List[dict]]) -> dict:
    """
    모든 passage 분석
    
    Args:
        passages: {passage_id: [quiz1, quiz2, quiz3, quiz4], ...}
    
    Returns:
        전체 분석 결과
    """
    all_results = []
    total_passages = 0
    significant_passages = 0
    
    for passage_id, quizzes in passages.items():
        total_passages += 1
        result = analyze_passage(quizzes)
        all_results.append(result)
        
        if result['overall_significant']:
            significant_passages += 1
    
    # 요약 통계
    summary = {
        'total_passages': total_passages,
        'significant_passages': significant_passages,
        'significant_rate': significant_passages / total_passages if total_passages > 0 else 0,
        'position_summary': {
            pos: {'significant': 0, 'total': total_passages}
            for pos in ['A', 'B', 'C', 'D']
        },
        'overall_average_selection_rates': {}  # 새로 추가: 전체 평균 선택 비율
    }
    
    # 위치별 유의성 집계
    for result in all_results:
        for pos in ['A', 'B', 'C', 'D']:
            if result['position_analysis'][pos]['significant']:
                summary['position_summary'][pos]['significant'] += 1
    
    # 위치별 유의성 비율 계산
    for pos in ['A', 'B', 'C', 'D']:
        summary['position_summary'][pos]['rate'] = \
            summary['position_summary'][pos]['significant'] / total_passages if total_passages > 0 else 0
    
    # 전체 평균 선택 비율 계산
    for pos in ['A', 'B', 'C', 'D']:
        avg_rates = [result['average_selection_rates'][pos] for result in all_results]
        summary['overall_average_selection_rates'][pos] = np.mean(avg_rates)
    
    return {
        'summary': summary,
        'passages': all_results
    }


# ============================================
# 4. 데이터 로드 및 그룹화
# ============================================

def load_json_files(file_paths: List[str]) -> Dict[str, List[dict]]:
    """
    JSON 파일들을 읽어서 passage별로 그룹화
    
    Args:
        file_paths: JSON 파일 경로 리스트
    
    Returns:
        passage별로 그룹화된 quiz 데이터
    """
    all_quizzes = []
    
    for file_path in file_paths:
        try:
            if file_path.endswith('.json'):
                with open(file_path, 'r', encoding='utf-8') as f:
                    quiz_data = json.load(f)
                    
                    # passage_id가 없으면 quiz_id 기반으로 생성
                    if 'passage_id' not in quiz_data:
                        quiz_data['passage_id'] = f"passage_{(quiz_data['quiz_id'] - 1) // 4 + 1}"
                    
                    all_quizzes.append(quiz_data)
            elif file_path.endswith('.jsonl'):
                with open(file_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        quiz_data = json.loads(line.strip())
                        
                        # passage_id가 없으면 quiz_id 기반으로 생성
                        if 'passage_id' not in quiz_data:
                            quiz_data['passage_id'] = f"passage_{(quiz_data['quiz_id'] - 1) // 4 + 1}"
                        
                        all_quizzes.append(quiz_data)
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
    
    print(f"총 {len(all_quizzes)}개의 quiz 데이터를 로드했습니다.")
    
    # passage별로 그룹화
    passages = {}
    for quiz in all_quizzes:
        pid = quiz['passage_id']
        if pid not in passages:
            passages[pid] = []
        passages[pid].append(quiz)
    
    # 각 passage가 4개의 quiz를 가지는지 확인 및 정렬
    for pid, quizzes in passages.items():
        if len(quizzes) != 4:
            print(f"Warning: {pid} has {len(quizzes)} quizzes (expected 4)")
        
        # 정답 위치별로 정렬
        quizzes.sort(key=lambda x: x['correct_position'])
    
    print(f"총 {len(passages)}개의 passage를 발견했습니다.\n")
    
    return passages


def load_from_directory(directory_path: str, pattern: str = "*.jsonl") -> Dict[str, List[dict]]:
    """
    디렉토리에서 모든 JSON 파일 로드
    
    Args:
        directory_path: 디렉토리 경로
        pattern: 파일 패턴 (기본값: "*.json")
    
    Returns:
        passage별로 그룹화된 quiz 데이터
    """
    dir_path = Path(directory_path)
    file_paths = list(dir_path.glob(pattern))
    file_paths = [str(p) for p in file_paths]
    
    print(f"{len(file_paths)}개의 JSON 파일을 발견했습니다.")
    
    return load_json_files(file_paths)


# ============================================
# 5. 결과 출력
# ============================================

def print_results(analysis_result: dict):
    """분석 결과 출력"""
    summary = analysis_result['summary']
    passages = analysis_result['passages']
    
    print("=" * 70)
    print("전체 분석 결과 요약")
    print("=" * 70)
    
    print(f"\n총 Passage 수: {summary['total_passages']}")
    print(f"유의한 Passage 수: {summary['significant_passages']} "
          f"({summary['significant_rate'] * 100:.1f}%)")
    print(f"기준: 4개 위치 중 2개 이상에서 p < 0.05\n")
    
    print("위치별 유의성 통계:")
    for pos in ['A', 'B', 'C', 'D']:
        stat = summary['position_summary'][pos]
        print(f"  {pos}: {stat['significant']}/{stat['total']} "
              f"({stat['rate'] * 100:.1f}%) passages에서 유의")
    
    print("\n전체 평균 선택 비율:")
    for pos in ['A', 'B', 'C', 'D']:
        rate = summary['overall_average_selection_rates'][pos]
        print(f"  {pos}: {rate:.2f}%")
    
    print("\n" + "=" * 70)
    print("Passage별 상세 결과")
    print("=" * 70)
    
    for idx, result in enumerate(passages, 1):
        status = "✓ 유의함" if result['overall_significant'] else "✗ 유의하지 않음"
        print(f"\n[Passage {idx}] {status}")
        print(f"유의한 위치: {', '.join(result['significant_positions']) or '없음'}")
        
        print(f"평균 선택 비율: ", end="")
        for pos in ['A', 'B', 'C', 'D']:
            rate = result['average_selection_rates'][pos]
            print(f"{pos}={rate:.1f}% ", end="")
        print()
        
        for pos in ['A', 'B', 'C', 'D']:
            analysis = result['position_analysis'][pos]
            sig = "**" if analysis['significant'] else "  "
            diff = analysis['rate_difference'] * 100
            diff_sign = "+" if diff > 0 else ""
            print(f"  {sig}{pos}: p={analysis['p_value']:.6f} | "
                  f"정답일때 {analysis['count_when_correct']}/{analysis['total_when_correct']} ({analysis['rate_when_correct'] * 100:.1f}%) vs "
                  f"아닐때 {analysis['count_when_incorrect']}/{analysis['total_when_incorrect']} ({analysis['rate_when_incorrect'] * 100:.1f}%) "
                  f"[차이: {diff_sign}{diff:.1f}%p]")
        
        # # 각 quiz별 선택 비율 출력
        # print(f"  Quiz별 선택 비율:")
        # for quiz in result['quizzes']:
        #     rates = quiz['position_selection_rates']
        #     print(f"    Quiz {quiz['quiz_id']} (정답:{quiz['correct_position']}): "
        #           f"A={rates['A']:.1f}% B={rates['B']:.1f}% "
        #           f"C={rates['C']:.1f}% D={rates['D']:.1f}%")


def print_detailed_statistics(analysis_result: dict):
    """상세 통계 출력"""
    summary = analysis_result['summary']
    passages = analysis_result['passages']
    
    print("\n" + "=" * 70)
    print("상세 통계 분석")
    print("=" * 70)
    
    # 1. 유의한 위치 개수 분포
    print("\n[1] 유의한 위치 개수 분포:")
    sig_count_dist = [0] * 5  # 0개, 1개, 2개, 3개, 4개
    for p in passages:
        sig_count_dist[len(p['significant_positions'])] += 1
    
    for num_sig, count in enumerate(sig_count_dist):
        pct = count / len(passages) * 100 if passages else 0
        print(f"  {num_sig}개 위치 유의: {count} passages ({pct:.1f}%)")
    
    # 2. 위치별 p-value 분포
    print("\n[2] 위치별 p-value < 0.05인 passage 수:")
    for pos in ['A', 'B', 'C', 'D']:
        count = summary['position_summary'][pos]['significant']
        total = summary['total_passages']
        pct = count / total * 100 if total > 0 else 0
        print(f"  {pos}: {count}/{total} ({pct:.1f}%)")
    
    # 3. 위치별 평균 선택 비율
    print("\n[3] 위치별 평균 선택 비율:")
    for pos in ['A', 'B', 'C', 'D']:
        sum_correct = sum(p['position_analysis'][pos]['rate_when_correct'] for p in passages)
        sum_incorrect = sum(p['position_analysis'][pos]['rate_when_incorrect'] for p in passages)
        
        avg_correct = sum_correct / len(passages) * 100 if passages else 0
        avg_incorrect = sum_incorrect / len(passages) * 100 if passages else 0
        diff = avg_correct - avg_incorrect
        
        overall_avg = summary['overall_average_selection_rates'][pos]
        
        print(f"  {pos}: 정답일 때 {avg_correct:.2f}% vs 아닐 때 {avg_incorrect:.2f}% "
              f"(차이: {diff:.2f}%p) | 전체 평균: {overall_avg:.2f}%")
    
    # 4. 가장 편향이 심한 passages
    print("\n[4] 편향이 가장 심한 상위 5개 passages:")
    passages_with_bias = []
    for idx, p in enumerate(passages, 1):
        max_bias = 0
        for pos in ['A', 'B', 'C', 'D']:
            analysis = p['position_analysis'][pos]
            bias = abs(analysis['rate_when_correct'] - analysis['rate_when_incorrect'])
            max_bias = max(max_bias, bias)
        passages_with_bias.append((idx, max_bias, p))
    
    passages_with_bias.sort(key=lambda x: x[1], reverse=True)
    
    for idx, bias, p in passages_with_bias[:5]:
        print(f"  Passage {idx}: 최대 편향 {bias * 100:.2f}%p, "
              f"유의한 위치 {len(p['significant_positions'])}개")

# ============================================
# 7. 메인 실행 함수
# ============================================

def main():
    """메인 실행 함수 - 사용 예시"""
    
    # # 방법 1: 디렉토리에서 모든 JSON 파일 로드
    # passages = load_from_directory("/data2/PromptMIA/Falcon/results/batch_results_20251002_114022.jsonl")
    
    # 방법 2: 특정 파일들 로드
    file_list = ["/data2/PromptMIA/Falcon/results/batch_results_20251002_114022.jsonl"]
    passages = load_json_files(file_list)
    
    # 분석 실행
    result = analyze_all_passages(passages)
    
    # 결과 출력
    print_results(result)
    print_detailed_statistics(result)
        
    return result


if __name__ == "__main__":
    # 사용 예시
    result = main()