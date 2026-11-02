"""Fetch only the frozen study's asset plan and exact material mirrors. No simulation."""
import argparse
import hashlib
import json
import os
from pathlib import Path
from common import BUNDLE, DATA, dataset_url, digest, fetch, load, relative_path, write_new


def valid(path,row,overrides):
    if not path.is_file():return False
    if row['path'] in overrides:return digest(path)==overrides[row['path']]['runtime_sha256']
    if path.stat().st_size!=row['size']:return False
    if row.get('lfs'):return digest(path)==row['lfs']['oid']
    raw=path.read_bytes();return hashlib.sha1(f'blob {len(raw)}\0'.encode()+raw).hexdigest()==row['oid']


def install(experiment):
    experiment=Path(experiment).resolve()
    rows={r['path']:r for entries in load(experiment/'TASK_ASSETS.json').values() for r in entries}
    overrides=load(experiment/'ASSET_OVERRIDES.json')
    patches={str(Path(p['asset']).relative_to(DATA)):p for p in load(BUNDLE/'catalog/ASSET_PATCHES.json')}
    for item in load(BUNDLE/'catalog/NVIDIA_MATERIALS.json')['files']:
        fetch(item['url'],DATA/relative_path(item['path']),sha256=item['sha256'],size=item['bytes'])
    for i,(name,row) in enumerate(sorted(rows.items()),1):
        target=DATA/relative_path(name)
        if valid(target,row,overrides):continue
        if target.exists():raise ValueError('Existing asset differs; not overwritten: '+name)
        if name in overrides:
            original=target.with_name(target.name+'.original')
            fetch(dataset_url(name),original,sha256=overrides[name]['original_sha256'],size=row['size'])
            raw=original.read_bytes()
            for change in patches[name]['changes']:
                old=change['original'].encode();new=change['local'].encode()
                if raw.count(old)!=1:raise ValueError('Audited material string does not appear exactly once')
                raw=raw.replace(old,new)
            if hashlib.sha256(raw).hexdigest()!=overrides[name]['runtime_sha256']:
                raise ValueError('Audited material-only patch differs')
            with target.open('xb') as out:out.write(raw)
        else:
            fetch(dataset_url(name),target,sha256=row.get('lfs',{}).get('oid'),git_oid=row.get('oid'),size=row['size'])
        if i%20==0:print(json.dumps({'verified':i,'total':len(rows)}),flush=True)
    assert all(valid(DATA/name,row,overrides) for name,row in rows.items())
    if not (experiment/'ASSETS_VERIFIED.json').exists():
        write_new(experiment/'ASSETS_VERIFIED.json',{'verdict':'PASS','files':len(rows),'scope':'Pinned assets only; no Kit/model calls'})
    print('PASS:',len(rows),'study assets and exact material mirrors verified')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('experiment',type=Path);install(p.parse_args().experiment)
