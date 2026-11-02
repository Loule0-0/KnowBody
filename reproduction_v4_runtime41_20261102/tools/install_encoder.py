"""Acquire pinned encoder files by content hash. Never run an encoder/model here."""
import argparse
import shutil
from pathlib import Path
from urllib.parse import quote
from common import BUNDLE, DATA, digest, fetch, load


def install(source=None):
    c=load(BUNDLE/'source/workspace/research/kyb/retrieval_encoder.json')
    target=Path(c['model_path'])
    if source:
        source=Path(source)
        for rel,expected in c['files'].items():
            if digest(source/rel)!=expected:raise ValueError('Encoder source differs: '+rel)
        for rel,expected in c['files'].items():
            out=target/rel
            if out.exists():
                if digest(out)!=expected:raise ValueError('Existing encoder differs: '+rel)
            else:
                out.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source/rel,out)
    else:
        for rel,expected in c['files'].items():
            url='https://modelscope.cn/api/v1/models/Qwen/Qwen3-Embedding-0.6B/repo?Revision=master&FilePath='+quote(rel,safe='')
            fetch(url,target/rel,sha256=expected)
    print('PASS: exact',len(c['files']),'encoder files; CPU worker not started')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path);install(p.parse_args().source)
