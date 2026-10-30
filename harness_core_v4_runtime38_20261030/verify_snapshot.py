"""Verify copied source bytes and syntax, without executing harness code."""
from pathlib import Path
import ast,hashlib,json

root=Path(__file__).resolve().parent
manifest=json.loads((root/'SOURCE_MANIFEST.json').read_text())
for row in manifest['files']:
    file=root/row['path'];raw=file.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=row['sha256']:
        raise SystemExit('NEEDS_WORK: source hash mismatch: '+row['path'])
    if file.suffix=='.py':ast.parse(raw,filename=row['path'])
    elif file.suffix=='.json':json.loads(raw)
print('PASS:',manifest['snapshot'],len(manifest['files']),'source files; hashes, Python syntax and JSON valid.')
