"""Official Qwen prompt rewriting, loaded only while a job is being prepared."""
import gc
import json
import logging
import os
from pathlib import Path

LOG = logging.getLogger('qwen.enhancer')
T2I_MODEL = 'Qwen/Qwen-Image-2.1-PE-T2I'
EDIT_MODEL = 'Qwen/Qwen-Image-2.1-PE-I2I'


def enabled():
    value = os.getenv('PROMPT_ENHANCER_ENABLED', 'true').strip().lower()
    if value not in {'true', 'false', '1', '0'}:
        raise ValueError('PROMPT_ENHANCER_ENABLED deve ser true ou false.')
    return value in {'true', '1'}


def parse_rewrite(text):
    if '</think>' in text:
        text = text.split('</think>', 1)[1]
    elif '<think>' in text:
        raise ValueError('Prompt enhancer não terminou de gerar a resposta.')
    decoder = json.JSONDecoder()
    for index, char in enumerate(text):
        if char != '{':
            continue
        try:
            value, _ = decoder.raw_decode(text[index:])
        except ValueError:
            continue
        prompt = value.get('rewritten_prompt') if isinstance(value, dict) else None
        if isinstance(prompt, str) and prompt.strip():
            return prompt.strip()
    raise ValueError('Prompt enhancer não retornou um prompt válido.')


def enhance(prompt, images):
    import torch
    import psutil
    from huggingface_hub import hf_hub_download
    from transformers import AutoModelForCausalLM, AutoModelForImageTextToText, AutoProcessor, AutoTokenizer

    editing = bool(images)
    model_id = os.getenv('PROMPT_ENHANCER_EDIT_MODEL', EDIT_MODEL) if editing else os.getenv('PROMPT_ENHANCER_T2I_MODEL', T2I_MODEL)
    revision = os.getenv('PROMPT_ENHANCER_REVISION', 'main')
    path = hf_hub_download(model_id, 'system_prompt.txt', revision=revision)
    system_prompt = Path(path).read_text(encoding='utf-8').strip()
    # Leave capacity for activations and the diffusion pipeline. Accelerate will
    # place remaining layers in RAM; the model is released before diffusion.
    gpu_budget = os.getenv('PROMPT_ENHANCER_GPU_MEMORY', '10GiB')
    cpu_available = max(1, int(psutil.virtual_memory().available * 0.8 / (1024 ** 3)))
    max_memory = {0: gpu_budget, 'cpu': f'{cpu_available}GiB'}
    model = processor = inputs = output = None
    try:
        cls = AutoModelForImageTextToText if editing else AutoModelForCausalLM
        processor_cls = AutoProcessor if editing else AutoTokenizer
        processor = processor_cls.from_pretrained(model_id, revision=revision)
        model = cls.from_pretrained(model_id, revision=revision, dtype=torch.bfloat16, device_map='auto', max_memory=max_memory).eval()
        placement = getattr(model, 'hf_device_map', {})
        if 'disk' in placement.values():
            raise RuntimeError('Memória RAM insuficiente para carregar o prompt enhancer; desative-o ou aumente a RAM.')
        if editing:
            content = [{'type': 'image', 'image': image.convert('RGB')} for image in images]
            content.append({'type': 'text', 'text': prompt})
            messages = [{'role': 'system', 'content': [{'type': 'text', 'text': system_prompt}]}, {'role': 'user', 'content': content}]
            inputs = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors='pt', enable_thinking=True).to(model.device)
            tokenizer = processor.tokenizer
        else:
            messages = [{'role': 'system', 'content': system_prompt}, {'role': 'user', 'content': prompt}]
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=True)
            inputs = processor(text, return_tensors='pt').to(model.device)
            tokenizer = processor
        max_tokens = int(os.getenv('PROMPT_ENHANCER_MAX_TOKENS', '24000' if editing else '16256'))
        with torch.inference_mode():
            output = model.generate(**inputs, max_new_tokens=max_tokens, do_sample=True, temperature=1.0, top_p=0.95, top_k=20)
        text = tokenizer.decode(output[0, inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        return parse_rewrite(text)
    finally:
        del output, inputs, model, processor
        gc.collect()
        torch.cuda.empty_cache()
