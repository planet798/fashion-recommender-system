import json
import logging
import os

logger = logging.getLogger(__name__)


class LLMService:
    BACKEND_ZHIPU = "zhipu"
    BACKEND_LOCAL_QWEN = "local_qwen"

    def __init__(self, backend=None, api_key=None, local_model_path=None):
        self.backend = backend or os.environ.get("LLM_BACKEND", self.BACKEND_ZHIPU)
        self.api_key = api_key or os.environ.get("ZHIPU_API_KEY", "")
        self.local_model_path = local_model_path or os.environ.get(
            "LLM_LOCAL_MODEL_PATH", "Qwen/Qwen2.5-1.5B-Instruct"
        )
        self._zhipu_client = None
        self._local_tokenizer = None
        self._local_model = None
        self._local_device = None

    def _init_zhipu(self):
        if self._zhipu_client is not None:
            return
        try:
            from zhipuai import ZhipuAI
            self._zhipu_client = ZhipuAI(api_key=self.api_key)
            logger.info("ZhipuAI client initialized")
        except ImportError:
            raise ImportError(
                "zhipuai package not installed. Run: pip install zhipuai"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to initialize ZhipuAI client: {e}")

    def _init_local_model(self):
        if self._local_model is not None:
            return
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            self._local_device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"Loading local model from {self.local_model_path} on {self._local_device}...")

            self._local_tokenizer = AutoTokenizer.from_pretrained(
                self.local_model_path, trust_remote_code=True
            )
            self._local_model = AutoModelForCausalLM.from_pretrained(
                self.local_model_path,
                torch_dtype="auto",
                device_map="auto",
                trust_remote_code=True,
            )
            self._local_model.eval()
            logger.info("Local model loaded successfully")
        except ImportError:
            raise ImportError(
                "transformers/torch not installed. Run: pip install transformers accelerate torch"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load local model: {e}")

    def chat(self, messages, temperature=0.7, max_tokens=1024):
        if self.backend == self.BACKEND_ZHIPU:
            return self._chat_zhipu(messages, temperature, max_tokens)
        elif self.backend == self.BACKEND_LOCAL_QWEN:
            return self._chat_local(messages, temperature, max_tokens)
        else:
            raise ValueError(f"Unknown backend: {self.backend}")

    def _chat_zhipu(self, messages, temperature=0.7, max_tokens=1024):
        self._init_zhipu()
        try:
            response = self._zhipu_client.chat.completions.create(
                model="glm-4-flash",
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"ZhipuAI API call failed: {e}")
            raise

    def _chat_local(self, messages, temperature=0.7, max_tokens=1024):
        self._init_local_model()
        try:
            import torch

            text = self._local_tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = self._local_tokenizer([text], return_tensors="pt").to(
                self._local_device
            )
            with torch.no_grad():
                outputs = self._local_model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    do_sample=temperature > 0,
                    top_p=0.9,
                )
            generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
            return self._local_tokenizer.decode(
                generated_ids, skip_special_tokens=True
            ).strip()
        except Exception as e:
            logger.error(f"Local model inference failed: {e}")
            raise

    def extract_json(self, text):
        text = text.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            lines = [l for l in lines if not l.strip().startswith("```")]
            text = "\n".join(lines).strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            try:
                return json.loads(text[start:end])
            except json.JSONDecodeError:
                pass
        logger.warning(f"Failed to parse JSON from LLM response: {text[:200]}")
        return None

    def is_available(self):
        if self.backend == self.BACKEND_ZHIPU:
            return bool(self.api_key)
        elif self.backend == self.BACKEND_LOCAL_QWEN:
            return True
        return False
