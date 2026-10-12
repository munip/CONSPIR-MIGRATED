"""Pinned Prompt Guard 2 CPU scanner for the detector-only experiment."""
from __future__ import annotations

import hashlib
import math

MODEL_ID = 'meta-llama/Llama-Prompt-Guard-2-86M'
MODEL_REVISION = 'a8ded8e697ce7c355e395a0df51f94adb4a2fd27'


class PromptGuardScanner:
    def initialize(self):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_ID, revision=MODEL_REVISION, use_safetensors=True,
        ).to('cpu').eval()
        labels = {str(value).upper(): int(key) for key, value in self.model.config.id2label.items()}
        if labels == {'LABEL_0': 0, 'LABEL_1': 1} and self.model.config.num_labels == 2:
            # Meta's pinned inference helper uses probabilities[0, 1] for MALICIOUS:
            # llama-cookbook@3c106f3e6ee79d6df51ae706bc7ef2d734ec3ded,
            # getting-started/responsible_ai/prompt_guard/inference.py:get_jailbreak_score.
            labels = {'BENIGN': 0, 'MALICIOUS': 1}
        if set(labels) != {'BENIGN', 'MALICIOUS'}:
            raise ValueError(f'Unexpected Prompt Guard labels: {labels}')
        self.malicious_index = labels['MALICIOUS']
        self.window_size = min(512, self.tokenizer.model_max_length,
                               self.model.config.max_position_embeddings) - self.tokenizer.num_special_tokens_to_add(pair=False)
        self.stride = self.window_size // 2
        if self.stride < 1:
            raise ValueError('Prompt Guard has no usable token window')

    @property
    def effective_config(self):
        return dict(model_id=MODEL_ID, model_revision=MODEL_REVISION, device='cpu',
                    threshold=.5, malicious_class_index=self.malicious_index, temperature=1.0, score='maximum softmax MALICIOUS across all windows',
                    window_tokens=self.window_size, stride_tokens=self.stride,
                    full_input=True, preprocessing='Pinned model tokenizer on unchanged assistant text')

    def scan(self, text):
        tokens = self.tokenizer.encode(text, add_special_tokens=False, truncation=False)
        score = 0.0
        windows = 0
        with self.torch.inference_mode():
            for start in range(0, max(1, len(tokens)), self.stride):
                chunk = tokens[start:start + self.window_size]
                inputs = self.tokenizer.prepare_for_model(
                    chunk, return_tensors='pt', truncation=False, prepend_batch_axis=True,
                )
                logits = self.model(**inputs).logits
                if not self.torch.isfinite(logits).all():
                    raise RuntimeError('Prompt Guard returned non-finite logits')
                probability = float(self.torch.softmax(logits, dim=-1)[0, self.malicious_index])
                if not math.isfinite(probability):
                    raise RuntimeError('Prompt Guard returned a non-finite score')
                score = max(score, probability)
                windows += 1
                if start + self.window_size >= len(tokens):
                    break
        return dict(action='BLOCK' if score >= .5 else 'PASS', score=score,
                    token_count=len(tokens), window_count=windows,
                    input_hash=hashlib.sha256(text.encode('utf-8')).hexdigest(),
                    input_length=len(text), config=self.effective_config)
