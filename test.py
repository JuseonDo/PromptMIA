import os
import re
import json
import fire
from dotenv import load_dotenv
from collections import Counter
from datetime import datetime
from typing import Dict, Tuple, List
from Falcon.utils.model_utils import FalconGenerator, GenerateConfig

# ORIGINAL = (
#     "these birches can be found in many places in Europe - the photos is from a short trip to Baden-Baden in 2007. "
#     "the clouds in the background are the messengers of the storm Kyrill. here are some more moments of the trip: Baden-Baden.\n"
#     "-\n"
#     "“ast/ray” is a bilingual wordplay: “ast” means “twig” in German. and while “Baden-Baden” sounds like wordplay, too, "
#     "it is the actual name of a rather well-know spa town that also dates back to Roman times. “Bad” is the German word for “bath”.\n"
#     "Mirror effect turned out nice. I like"
# )

# def build_quiz(target: str) -> Tuple[str, Dict[str, str], str]:
#     target = target.lower()
#     if target == "place":
#         paragraph = ORIGINAL.replace("Baden-Baden", "( ___ )", 1)
#         options = {"A": "Heidelberg", "B": "Dresden", "C": "Baden-Baden", "D": "Munich", "E": "Leipzig"}
#         correct = "C"
#     elif target == "year":
#         paragraph = ORIGINAL.replace("2007", "( ___ )", 1)
#         options = {"A": "2005", "B": "2006", "C": "2007", "D": "2008", "E": "2009"}
#         correct = "C"
#     elif target == "storm":
#         paragraph = ORIGINAL.replace("Kyrill", "( ___ )", 1)
#         options = {"A": "Xaver", "B": "Kyrill", "C": "Lothar", "D": "Sabine", "E": "Friederike"}
#         correct = "B"
#     elif target == "ast_meaning":
#         paragraph = ORIGINAL.replace("“ast” means “twig”", "“ast” means “( ___ )”", 1)
#         options = {"A": "twig", "B": "light", "C": "stone", "D": "river", "E": "path"}
#         correct = "A"
#     elif target == "bad_meaning":
#         paragraph = ORIGINAL.replace("“Bad” is the German word for “bath”", "“Bad” is the German word for “( ___ )”", 1)
#         options = {"A": "mountain", "B": "sea", "C": "river", "D": "bath", "E": "forest"}
#         correct = "D"
#     else:
#         raise ValueError("target must be one of: place, year, storm, ast_meaning, bad_meaning")
#     return paragraph, options, correct

ORIGINAL = (
    "Watch Survivor Redemption Island Season 22 Episode 11: A Mystery Package Online S22e11 Free Stream Megavideo\n\n"
    "Article by StreamThatSeries\n\n"
    "Horray!time for another dose of very exciting reality series with lots of twists.You must watch survivor redemption island season 22 episode 11 tonight "
    "with a title of 'Mystery Package' coz surely this will stir things up on the merge tribe murlonio. "
    "The show premiered on May 31, 2000 on CBS. It is hosted by veteran television personality Jeff Probst. "
    "The first U.S. season of Survivor followed the Swedish series, but subsequently the show has introduced several twists. "
    "Season 22 features the return of Rob Mariano and Russell Hantz. This is Russell's third time on the show and Rob's fourth."
)

def build_quiz(target: str):
    target = target.lower()
    if target == "season":
        paragraph = ORIGINAL.replace("Season 22", "Season ( ___ )", 1)
        options = {"A": "20", "B": "21", "C": "22", "D": "23"}
        correct = "C"
    elif target == "episode":
        paragraph = ORIGINAL.replace("Episode 11", "Episode ( ___ )", 1)
        options = {"A": "9", "B": "10", "C": "11", "D": "12"}
        correct = "C"
    elif target == "episode_title":
        paragraph = ORIGINAL.replace("'Mystery Package'", "'( ___ )'", 1)
        options = {"A": 'Mystery Package', "B": "Rice Wars", "C": "Redemption Island", "D": "Final Showdown"}
        correct = "A"
    elif target == "premiere_date":
        paragraph = ORIGINAL.replace("May 31, 2000", "( ___ )", 1)
        options = {"A": "May 28, 2000", "B": "May 31, 2000", "C": "June 1, 2000", "D": "June 15, 2000"}
        correct = "B"
    elif target == "host":
        paragraph = ORIGINAL.replace("Jeff Probst", "( ___ )", 1)
        options = {"A": "Phil Keoghan", "B": "Ryan Seacrest", "C": "Jeff Probst", "D": "Mark Burnett"}
        correct = "C"
    elif target == "rob_times":
        paragraph = ORIGINAL.replace("Rob's fourth", "Rob's ( ___ )", 1)
        options = {"A": "second", "B": "third", "C": "fourth", "D": "fifth"}
        correct = "C"
    elif target == "russell_times":
        paragraph = ORIGINAL.replace("Russell's third", "Russell's ( ___ )", 1)
        options = {"A": "second", "B": "third", "C": "fourth", "D": "fifth"}
        correct = "B"
    else:
        raise ValueError("target must be one of: season, episode, episode_title, premiere_date, host, rob_times, russell_times")
    return paragraph, options, correct

def make_plain_prompt(paragraph: str, options: Dict[str, str]) -> str:
    opts_text = "\n".join([f"{k}. {v}" for k, v in options.items()])
    return (
        "Question: Read the text and select the correct answer.\n\n"
        f"{paragraph}\n\n"
        f"{opts_text}\n\n"
        "Instructions:\n"
        "- Choose only ONE letter: A, B, C, D\n"
        "- Do not write explanations\n"
        "- Do not write full sentences\n\n"
        "Answer:"
    )

def make_messages(paragraph: str, options: Dict[str, str]):  
    opts_text = "\n".join([f"{k}. {v}" for k, v in options.items()])
    user = (
        f"{paragraph}\n\n"
        f"{opts_text}\n\n"
        "Choose one letter: A, B, C, D\n"
        "Do not explain. Answer:"
    )
    return [
        {"role": "system", "content": "You are a QA assistant. Answer with single letters only."},
        {"role": "user", "content": user},
    ]

def extract_letter(s: str) -> str:  # 정답 후보 추출 알고리즘
    m = re.search(r"\b([A-D])\b", s.strip(), flags=re.IGNORECASE)
    if m:
        return m.group(1).upper()
    for ch in s:
        if ch.upper() in "ABCD":
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
):
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
    target: str = "season",
    model_name: str = "tiiuae/falcon-7b-instruct",
    samples: int = 16,                 # 생성 샘플 개수
    batch_size: int = 8,               # 실제 inference 배치 크기
    max_new_tokens: int = 3,
    temperature: float = 0.1,
    top_p: float = 0.9,
    top_k: int = 0,
    repetition_penalty: float = 1.05,
    use_chat_template: bool = True,
    show_all: bool = True,
    seed: int = 0,
):

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

    load_dotenv()

    WORKSPACE = os.getenv("WORKSPACE")
    if not WORKSPACE:
        print("[WARN] environment variable 'WORKSPACE' is not set. Saving under current working directory root path.")
        WORKSPACE = os.getcwd()

    save_dir = os.getenv("RESULTS_DIR")
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

    final_filename = os.path.join("/data2/PromptMIA/Falcon/results", "final_results.jsonl")
    with open(final_filename, 'a') as f:
        data = {
            "predict": majority,
            "correct": correct  
        }
        f.write(json.dumps(data) + '\n')
    return 

if __name__ == "__main__":
    for i in range(5):
        fire.Fire(run)
