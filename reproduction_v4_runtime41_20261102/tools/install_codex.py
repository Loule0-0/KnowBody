"""Install the exact official Codex binary. Does not log in or call a model."""
import argparse
import os
import shutil
import tarfile
import tempfile
from pathlib import Path
from common import BUNDLE, PROJECT, load, digest, fetch


def install(archive=None):
    pin=load(BUNDLE/'catalog/CODEX_BINARY.json');target=PROJECT/pin['target']
    if target.exists():
        if digest(target)!=pin['binary_sha256']:raise ValueError('Existing Codex differs; not replaced')
        print('PASS: exact Codex already installed');return
    archive=Path(archive) if archive else PROJECT/'downloads/codex-0.161.0-linux-x86_64.tar.gz'
    fetch(pin['archive_url'],archive,sha256=pin['archive_sha256'])
    with tarfile.open(archive,'r:gz') as package:
        member=package.getmember('codex-x86_64-unknown-linux-musl')
        if not member.isfile():raise ValueError('Expected regular Codex executable')
        target.parent.mkdir(parents=True,exist_ok=True)
        temporary=target.with_name('codex.installing')
        with package.extractfile(member) as src,temporary.open('xb') as dst:shutil.copyfileobj(src,dst)
    if digest(temporary)!=pin['binary_sha256']:raise ValueError('Official archive/baseline executable differ; inspect, do not substitute')
    temporary.chmod(0o755);os.link(temporary,target);temporary.unlink()
    print('PASS: exact official Codex installed; authentication remains a separate local step')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--archive',type=Path)
    install(p.parse_args().archive)
