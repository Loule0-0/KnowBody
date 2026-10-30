"""Local CPU text encoding only; no memory access or retrieval decisions."""
from __future__ import annotations

import contextlib
import hashlib
import importlib.metadata
import json
import sys
from pathlib import Path


def main():
    config_path = Path(__file__).with_name('retrieval_encoder.json')
    config = json.loads(config_path.read_text())
    signature = hashlib.sha256(config_path.read_bytes()+Path(__file__).read_bytes()).hexdigest()
    for package, version in config['packages'].items():
        if importlib.metadata.version(package) != version:
            raise ValueError('Pinned embedding dependency changed: ' + package)
    root = Path(config['model_path'])
    for relative, digest in config['files'].items():
        if hashlib.sha256((root / relative).read_bytes()).hexdigest() != digest:
            raise ValueError('Pinned embedding model changed: ' + relative)
    with contextlib.redirect_stdout(sys.stderr):
        import torch
        from sentence_transformers import SentenceTransformer
        torch.set_num_threads(config['threads'])
        model = SentenceTransformer(str(root), device='cpu',
            model_kwargs={'torch_dtype': torch.float32}, local_files_only=True)
        model.max_seq_length = config['max_tokens']
        if torch.cuda.is_initialized():
            raise ValueError('Embedding worker must not initialize CUDA')
    print(json.dumps({'status': 'READY', 'signature': signature,
                      'dimension': config['dimension']}), flush=True)
    for line in sys.stdin:
        try:
            request = json.loads(line)
            if set(request) != {'items'} or not isinstance(request['items'], list):
                raise ValueError('Expected items array')
            texts = []
            for item in request['items']:
                if set(item) != {'key', 'text'} or not isinstance(item['text'], str):
                    raise ValueError('Expected keyed text items')
                tokens = model.tokenizer(item['text'], truncation=False,
                    add_special_tokens=True)['input_ids']
                if len(tokens) > config['max_tokens']:
                    raise ValueError('Text exceeds encoder token limit: ' + str(item['key']))
                texts.append(item['text'])
            with contextlib.redirect_stdout(sys.stderr):
                vectors = model.encode(texts, batch_size=config['batch_size'],
                    normalize_embeddings=True, show_progress_bar=False).tolist() if texts else []
            answer = {'status': 'PASS', 'signature': signature, 'vectors': vectors}
        except Exception as error:
            answer = {'status': 'NEEDS_WORK', 'error': type(error).__name__, 'reason': str(error)}
        print(json.dumps(answer, allow_nan=False, separators=(',', ':')), flush=True)


if __name__ == '__main__':
    try:main()
    except Exception as error:
        print(json.dumps({'status':'NEEDS_WORK','error':type(error).__name__,'reason':str(error)}),flush=True)
        raise SystemExit(1)
