"""Small deployment helpers; no simulator, encoder or hosted-model imports."""
from pathlib import Path
import hashlib
import json
import os
import urllib.request

BUNDLE = Path(__file__).resolve().parents[1]
PROJECT = Path('/home/lzy/KnowYourBody_RoboDojo')
DATA = Path('/data/KnowYourBody_RoboDojo')
WORKSPACE = PROJECT/'workspaces/v4-runtime41-20261101'
BASE_MANIFEST = PROJECT/'releases/v4/runtime41_reproduction_20261102/BASE_HARNESS.json'
DATASET = 'RoboDojo-Benchmark/RoboDojo'
REVISION = '35efbc7dedfdbeeb6e95fb749bd885d73d483e41'


def load(path):
    return json.loads(Path(path).read_text())


def digest(path, kind='sha256'):
    h = hashlib.new(kind)
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024*1024), b''):h.update(chunk)
    return h.hexdigest()


def write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as f:json.dump(value, f, ensure_ascii=False, indent=2);f.write('\n')


def relative_path(value):
    p = Path(value)
    if p.is_absolute() or '..' in p.parts:raise ValueError('Expected a relative path: '+str(p))
    return p


def verify_sources():
    rows = load(BUNDLE/'SOURCE_MANIFEST.json')['files']
    for row in rows:
        if digest(BUNDLE/relative_path(row['path'])) != row['sha256']:
            raise ValueError('Package source differs: '+row['path'])
    return rows


def fetch(url, destination, *, sha256=None, git_oid=None, size=None):
    """Verified, atomic download; an existing mismatching file is never replaced."""
    destination = Path(destination)
    def valid(path):
        if size is not None and path.stat().st_size != size:return False
        if sha256:return digest(path) == sha256
        raw = path.read_bytes()
        return hashlib.sha1(f'blob {len(raw)}\0'.encode()+raw).hexdigest() == git_oid
    if destination.exists():
        if not valid(destination):raise ValueError('Existing artifact differs: '+str(destination))
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name+'.download')
    if temporary.exists():raise ValueError('Inspect/remove the incomplete download before retry: '+str(temporary))
    request = urllib.request.Request(url, headers={'User-Agent':'KnowBody-runtime41-reproduction'})
    with urllib.request.urlopen(request, timeout=120) as response, temporary.open('xb') as output:
        while chunk := response.read(1024*1024):output.write(chunk)
        output.flush();os.fsync(output.fileno())
    if not valid(temporary):raise ValueError('Artifact checksum mismatch: '+str(temporary))
    os.link(temporary,destination)
    temporary.unlink()


def dataset_url(path):
    from urllib.parse import quote
    return f'https://huggingface.co/datasets/{DATASET}/resolve/{REVISION}/'+quote(str(relative_path(path)),safe='/')


def fetch_metadata_row(row):
    fetch(dataset_url(row['path']), DATA/relative_path(row['path']),
          sha256=row.get('lfs',{}).get('oid'), git_oid=row.get('oid'), size=row['size'])


def base_files():
    base = load(BASE_MANIFEST)
    for name, expected in base['files'].items():
        if digest(name) != expected:raise ValueError('Installed source differs: '+name)
    return base
