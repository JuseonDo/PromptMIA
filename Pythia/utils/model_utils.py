from typing import List, Optional, Dict, Any, Union
from dataclasses import dataclass

import os
import dotenv
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


@dataclass
class GenerateConfig:
    # sampling & length
    max_new_tokens: int = 256
    do_sample: bool = True
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 0
    prompt_type: str = "fewshot"
    repetition_penalty: float = 1.0
    # decoding control
    num_return_sequences: int = 1
    # return control
    return_only_new_text: bool = True
    # truncation for inputs
    truncation: bool = True
    max_input_tokens: Optional[int] = None  # None이면 토크나이저 기본값 사용


class PythiaGenerator:
    """
    Pythia 계열 모델 로드 + 싱글/배치 텍스트 생성 유틸.
    """
    def __init__(
        self,
        model_name: str = "EleutherAI/pythia-6.9b",
        cache_env_key: str = "model_cache_dir",
        torch_dtype: Optional[torch.dtype] = None,
        device_map: Union[str, Dict[str, int]] = "auto",
        trust_remote_code: bool = False,  # Pythia는 표준 아키텍처라 불필요
        tokenizer_kwargs: Optional[Dict[str, Any]] = None,
        model_kwargs: Optional[Dict[str, Any]] = None,
    ):
        """
        Args:
            model_name: 허깅페이스 모델 경로 (기본: EleutherAI/pythia-6.9b)
            cache_env_key: .env의 캐시 디렉토리 키 이름
            torch_dtype: None이면 GPU면 float16, 아니면 float32로 자동 설정
            device_map: "auto" 권장
            trust_remote_code: Pythia는 표준 아키텍처라 False
            tokenizer_kwargs, model_kwargs: 세부 옵션 오버라이드
        """
        dotenv.load_dotenv()
        cache_dir = os.getenv(cache_env_key, None)

        if torch_dtype is None:
            # Pythia는 float16 권장 (bfloat16도 가능하지만 float16이 더 일반적)
            torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        tokenizer_kwargs = tokenizer_kwargs or {}
        model_kwargs = model_kwargs or {}

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, cache_dir=cache_dir, **tokenizer_kwargs
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            torch_dtype=torch_dtype,
            device_map=device_map,
            trust_remote_code=trust_remote_code,
            **model_kwargs,
        )

        # Pythia 토크나이저는 기본적으로 pad_token이 설정되어 있지만 확인
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.pad_token_id = self.tokenizer.pad_token_id
        self.eos_token_id = self.tokenizer.eos_token_id

        # 편의 플래그
        self.device = next(self.model.parameters()).device

    def _tokenize(
        self,
        texts: List[str],
        truncation: bool = True,
        max_input_tokens: Optional[int] = None,
    ):
        return self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=truncation,
            max_length=max_input_tokens,
        ).to(self.device)

    def _postprocess(
        self,
        input_ids: torch.Tensor,
        generated_ids: torch.Tensor,
        return_only_new_text: bool = True,
    ) -> List[str]:
        """
        배치별로 입력 길이만큼을 잘라 새로 생성된 부분만 반환.
        """
        outputs: List[str] = []
        # 배치 차원 정렬
        if generated_ids.dim() == 1:
            generated_ids = generated_ids.unsqueeze(0)

        for i in range(generated_ids.size(0)):
            if return_only_new_text:
                # 각 샘플의 실제 입력 길이(패딩 제외)
                input_len = (input_ids[i] != self.pad_token_id).sum().item()
                new_tokens = generated_ids[i, input_len:]
                text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
            else:
                text = self.tokenizer.decode(generated_ids[i], skip_special_tokens=True)
            outputs.append(text)
        return outputs

    @torch.inference_mode()
    def generate_one(
        self,
        prompt: str,
        cfg: Optional[GenerateConfig] = None,
        **gen_kwargs
    ) -> str:
        """
        단일 프롬프트 생성.
        """
        cfg = cfg or GenerateConfig()
        enc = self._tokenize([prompt], truncation=cfg.truncation, max_input_tokens=cfg.max_input_tokens)

        # generate 파라미터 구성
        generate_params = dict(
            max_new_tokens=cfg.max_new_tokens,
            do_sample=cfg.do_sample,
            temperature=cfg.temperature,
            top_p=cfg.top_p,
            top_k=cfg.top_k,
            repetition_penalty=cfg.repetition_penalty,
            num_return_sequences=1,  # 단일
            pad_token_id=self.pad_token_id,
            eos_token_id=self.eos_token_id,
        )
        generate_params.update(gen_kwargs)

        out_ids = self.model.generate(**enc, **generate_params)
        out_text = self._postprocess(enc["input_ids"], out_ids, cfg.return_only_new_text)[0]
        return out_text

    @torch.inference_mode()
    def generate_batch(
        self,
        prompts: List[str],
        cfg: Optional[GenerateConfig] = None,
        **gen_kwargs
    ) -> List[str]:
        """
        배치 프롬프트 생성.
        """
        if not isinstance(prompts, list) or len(prompts) == 0:
            raise ValueError("prompts는 비어있지 않은 리스트여야 합니다.")

        cfg = cfg or GenerateConfig()
        print(prompts)
        enc = self._tokenize(prompts, truncation=cfg.truncation, max_input_tokens=cfg.max_input_tokens)

        generate_params = dict(
            max_new_tokens=cfg.max_new_tokens,
            do_sample=cfg.do_sample,
            temperature=cfg.temperature,
            top_p=cfg.top_p,
            top_k=cfg.top_k,
            repetition_penalty=cfg.repetition_penalty,
            num_return_sequences=1,
            pad_token_id=self.pad_token_id,
            eos_token_id=self.eos_token_id,
        )
        generate_params.update(gen_kwargs)

        out_ids = self.model.generate(
            input_ids=enc["input_ids"],
            attention_mask=enc["attention_mask"],
            **generate_params
        )
        out_texts = self._postprocess(enc["input_ids"], out_ids, cfg.return_only_new_text)
        return out_texts