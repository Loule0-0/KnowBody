"""Copy exact licensed Kit bundles from a supplied installation/cache directory."""
import argparse
from pathlib import Path
import shutil
from common import BUNDLE, DATA, digest, load, write_new


def seed(source):
    source=Path(source).resolve();target=DATA/'kit_extensions'
    pins=load(BUNDLE/'catalog/KIT_EXTENSIONS.json')['files']
    # Validate the complete input before creating or changing the destination.
    for rel,expected in pins.items():
        if not (source/rel).is_file() or digest(source/rel)!=expected:
            raise ValueError('Exact vendor bundle missing/different: '+str(source/rel))
    if target.exists():
        if all((target/rel).is_file() and digest(target/rel)==value for rel,value in pins.items()):
            print('PASS: exact Kit seed already present');return
        raise ValueError('Existing Kit seed is incomplete or different; not overwritten')
    target.mkdir(parents=True)
    for rel in pins:
        out=target/rel;out.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source/rel,out)
    write_new(DATA/'kit_extensions_manifest.json',{'files':pins,'source':str(source)})
    print('PASS:',len(pins),'exact vendor files seeded; no Kit/model started')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('source',type=Path);seed(p.parse_args().source)
