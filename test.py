import os
import re
import json
import fire
from collections import Counter
from datetime import datetime
from typing import Dict, Tuple, List
from Falcon.utils.model_utils import FalconGenerator, GenerateConfig

ORIGINAL = (
    "these birches can be found in many places in Europe - the photos is from a short trip to Baden-Baden in 2007. "
    "the clouds in the background are the messengers of the storm Kyrill. here are some more moments of the trip: Baden-Baden.\n"
    "-\n"
    "“ast/ray” is a bilingual wordplay: “ast” means “twig” in German. and while “Baden-Baden” sounds like wordplay, too, "
    "it is the actual name of a rather well-know spa town that also dates back to Roman times. “Bad” is the German word for “bath”.\n"
    "Mirror effect turned out nice. I like"
)

# ---- helpers to build quiz ----
def build_quiz(target: str) -> Tuple[str, Dict[str, str], str]:
    target = target.lower()
    if target == "place":
        paragraph = ORIGINAL.replace("Baden-Baden", "( ___ )", 1)
        options = {"A": "Heidelberg", "B": "Dresden", "C": "Baden-Baden", "D": "Munich", "E": "Leipzig"}
        correct = "C"
    elif target == "year":
        paragraph = ORIGINAL.replace("2007", "( ___ )", 1)
        options = {"A": "2005", "B": "2006", "C": "2007", "D": "2008", "E": "2009"}
        correct = "C"
    elif target == "storm":
        paragraph = ORIGINAL.replace("Kyrill", "( ___ )", 1)
        options = {"A": "Xaver", "B": "Kyrill", "C": "Lothar", "D": "Sabine", "E": "Friederike"}
        correct = "B"
    elif target == "ast_meaning":
        paragraph = ORIGINAL.replace("“ast” means “twig”", "“ast” means “( ___ )”", 1)
        options = {"A": "twig", "B": "light", "C": "stone", "D": "river", "E": "path"}
        correct = "A"
    elif target == "bad_meaning":
        paragraph = ORIGINAL.replace("“Bad” is the German word for “bath”", "“Bad” is the German word for “( ___ )”", 1)
        options = {"A": "mountain", "B": "sea", "C": "river", "D": "bath", "E": "forest"}
        correct = "D"
    else:
        raise ValueError("target must be one of: place, year, storm, ast_meaning, bad_meaning")
    return paragraph, options, correct

def make_plain_prompt(paragraph: str, options: Dict[str, str]) -> str:
    opts_text = "  ".join([f"{k}. {v}" for k, v in options.items()])
    return (
        "You are a concise QA assistant. Read the paragraph with ONE blank and choose the correct option.\n\n"
        f"{paragraph}\n\n"
        f"Options: {opts_text}\n\n"
        "Instruction: Answer with ONLY the single letter among A, B, C, D, E. No explanation."
    )

def make_messages(paragraph: str, options: Dict[str, str]) -> List[dict]:
    opts_text = "  ".join([f"{k}. {v}" for k, v in options.items()])
    user = (
        "Read the paragraph with ONE blank and choose the correct option.\n\n"
        f"{paragraph}\n\n"
        f"Options: {opts_text}\n\n"
        "Answer with ONLY one letter among A, B, C, D, E."
    )
    return [
        {"role": "system", "content": "You are a helpful, concise QA assistant."},
        {"role": "user", "content": user},
    ]

def extract_letter(s: str) -> str:
    m = re.search(r"\b([A-E])\b", s.strip(), flags=re.IGNORECASE)
    if m:
        return m.group(1).upper()
    for ch in s:
        if ch.upper() in "ABCDE":
            return ch.upper()
    return "?"

# ---- batching util ----
def chunked(lst: List[str], n: int) -> List[List[str]]:
    if n <= 0:
        return [lst]
    return [lst[i:i+n] for i in range(0, len(lst), n)]

# ---- filename builder (omit defaults) ----
def build_filename(
    target: str,
    samples: int,
    defaults: Dict[str, object],
    params: Dict[str, object],
) -> str:
    """
    파일명 예: quiz_target=place__samples=32__temp:0.6__top_P:0.9__rep:1.05__mx:8.jsonl
    - 디폴트와 같은 값은 생략
    - top_p는 'top_P'로 표기(요청 사항)
    """
    tags = [f"quiz_target={target}", f"samples={samples}"]
    # 표기 키 매핑
    keymap = {
        "temperature": "temp",
        "top_p": "top_P",
        "top_k": "top_k",
        "repetition_penalty": "rep",
        "max_new_tokens": "mx",
        "use_chat_template": "chat",   # on/off
        "batch_size": "bs",
        "seed": "seed",
    }
    for k, v in params.items():
        if k not in defaults:
            continue
        if v == defaults[k]:
            continue  # 디폴트는 생략
        if k == "use_chat_template":
            vv = "on" if v else "off"
        else:
            vv = str(v)
        tags.append(f"{keymap.get(k, k)}:{vv}")

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    fname = "__".join(tags) + f"__{timestamp}.jsonl"
    return fname

# ---- main entry ----
def run(
    target: str = "place",
    model_name: str = "tiiuae/falcon-7b-instruct",
    samples: int = 16,                 # 생성 샘플 개수
    batch_size: int = 8,               # 실제 inference 배치 크기
    max_new_tokens: int = 8,
    temperature: float = 0.6,
    top_p: float = 0.9,
    top_k: int = 0,
    repetition_penalty: float = 1.05,
    use_chat_template: bool = True,
    show_all: bool = True,
    seed: int = 0,
):
    """
    동일 프롬프트를 batch로 N개 넣어 샘플링하고, 입력 프롬프트 & 출력들을 JSONL로 저장.
    저장 경로: os.path.join(os.getenv('workspace'), '/Falcon/results')
    파일명에 비-디폴트 파라미터만 태그로 반영 (top_p -> top_P 등).
    """
    # ---- build quiz ----
    paragraph, options, correct = build_quiz(target)

    # ---- model & cfg ----
    gen = FalconGenerator(model_name=model_name)
    cfg = GenerateConfig(
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        repetition_penalty=repetition_penalty,
        return_only_new_text=True,
        do_sample=True,
    )

    # ---- prompt (chat template if available) ----
    if use_chat_template and getattr(gen.tokenizer, "chat_template", None):
        messages = make_messages(paragraph, options)
        prompt_text = gen.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    else:
        prompt_text = make_plain_prompt(paragraph, options)

    # ---- DEBUG: first prompt ----
    print("\n==== DEBUG: First prompt sent to model ====\n")
    print(prompt_text)
    print("\n===========================================\n")

    # ---- seed (optional) ----
    if seed != 0:
        import torch
        torch.manual_seed(seed)

    # ---- prepare prompts ----
    prompts = [prompt_text] * samples

    # ---- run batched inference (chunked by batch_size) ----
    raw_outputs: List[str] = []
    for chunk in chunked(prompts, batch_size):
        raw_outputs.extend(gen.generate_batch(chunk, cfg))

    # ---- analyze ----
    preds = [extract_letter(o) for o in raw_outputs]
    dist = Counter(preds)
    majority = max(dist.items(), key=lambda x: (x[1], x[0]))[0] if dist else "?"

    print("==== QUIZ ====")
    print(paragraph)
    print("\nOptions:")
    for k, v in options.items():
        print(f"  {k}. {v}")

    print("\n==== BATCH RESULTS ====")
    print(f"samples: {samples} | batch_size: {batch_size}")
    print("distribution:", " ".join([f"{k}:{v}" for k, v in sorted(dist.items())]))
    print("majority_vote:", majority)
    print("correct:", correct)

    if show_all:
        print("\n---- per-sample outputs ----")
        for i, (raw, p) in enumerate(zip(raw_outputs, preds)):
            print(f"[{i:02d}] pred={p} | raw='{raw.strip()}'")

    # ---- save jsonl (prompt + outputs) ----
    # defaults dict for filename filtering
    defaults = {
        "temperature": 0.6,
        "top_p": 0.9,
        "top_k": 0,
        "repetition_penalty": 1.05,
        "max_new_tokens": 8,
        "use_chat_template": True,
        "batch_size": 8,
        "seed": 0,
    }
    params = {
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "repetition_penalty": repetition_penalty,
        "max_new_tokens": max_new_tokens,
        "use_chat_template": use_chat_template,
        "batch_size": batch_size,
        "seed": seed,
    }
    filename = build_filename(target, samples, defaults, params)

    workspace = os.getenv("workspace")
    if not workspace:
        print("[WARN] environment variable 'workspace' is not set. Saving under current working directory root path.")
        workspace = os.getcwd()

    # NOTE: 요청에 따라 정확히 이 형태로 join (절대경로 주의)
    save_dir = os.path.join(workspace, "/Falcon/results")
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, filename)

    meta = {
        "model_name": model_name,
        "target": target,
        "options": options,
        "correct": correct,
        "samples": samples,
        "batch_size": batch_size,
        "params": params,
        "use_chat_template_detected": bool(getattr(gen.tokenizer, "chat_template", None)),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }

    with open(save_path, "w", encoding="utf-8") as f:
        # 1) 메타 한 줄
        f.write(json.dumps({"type": "meta", **meta}, ensure_ascii=False) + "\n")
        # 2) 각 샘플 한 줄(입력/출력 저장)
        for i, (raw, pred) in enumerate(zip(raw_outputs, preds)):
            rec = {
                "type": "sample",
                "index": i,
                "prompt": prompt_text,     # 입력 프롬프트 저장
                "output_raw": raw,         # 원문 출력
                "output_choice": pred,     # 정제한 선택지 한 글자
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"\n[INFO] Saved prompts & outputs to JSONL:\n{save_path}")

if __name__ == "__main__":
    fire.Fire(run)