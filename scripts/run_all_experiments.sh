#!/bin/bash
# run_all_experiments.sh

# 퀴즈 JSONL 파일 경로 (이미 빈칸이 뚫린 문제들)
QUIZ_JSONL="/data2/PromptMIA/Dataset/quiz_dataset.jsonl"
SCRIPT_PATH="/data2/PromptMIA/ej_Falcon_test.py"

# GPU 설정
export CUDA_VISIBLE_DEVICES=6  

# JSONL 파일 존재 확인
if [ ! -f "$QUIZ_JSONL" ]; then
    echo "Error: Quiz JSONL file not found at $QUIZ_JSONL"
    exit 1
fi

# Python 스크립트 존재 확인
if [ ! -f "$SCRIPT_PATH" ]; then
    echo "Error: Python script not found at $SCRIPT_PATH"
    exit 1
fi

echo "========================================="
echo "Starting Quiz Experiments"
echo "========================================="
echo "Quiz JSONL: $QUIZ_JSONL"
echo ""

# Python 스크립트 실행
python "$SCRIPT_PATH" \
    "$QUIZ_JSONL" \
    --model_name="tiiuae/falcon-7b-instruct" \
    --samples=32 \
    --num_repeats=15 \
    --temperature=0.7 \
    --top_p=0.9 \
    --batch_size=16 \
    --max_new_tokens=5 \
    --repetition_penalty=1.2 \
    --test_number=1

# 실행 결과 확인
if [ $? -eq 0 ]; then
    echo ""
    echo "========================================="
    echo "✓ All experiments completed successfully!"
    echo "========================================="
    echo "Results saved in: /data2/PromptMIA/Falcon_ej/results/test_1"
else
    echo ""
    echo "========================================="
    echo "✗ Experiments failed"
    echo "========================================="
    exit 1
fi