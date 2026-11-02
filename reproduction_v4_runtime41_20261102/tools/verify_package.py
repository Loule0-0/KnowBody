"""Verify published source bytes/syntax without importing or executing the harness."""
import ast
import json
from pathlib import Path
from common import BUNDLE, digest, load, verify_sources


def verify():
    rows=verify_sources()
    package=load(BUNDLE/'PACKAGE_MANIFEST.json')
    for name,expected in package['files'].items():
        p=BUNDLE/name
        if digest(p)!=expected:raise ValueError('Published package changed: '+name)
        if p.suffix=='.py':ast.parse(p.read_bytes(),filename=name)
        if p.suffix=='.json':json.loads(p.read_text())
    print('PASS:',len(rows),'original source files;',len(package['files']),'package files; hashes, AST and JSON')


if __name__=='__main__':verify()
