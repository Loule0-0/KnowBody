"""Bind a pinned layout filename to the official full-catalog numeric index."""
from pathlib import Path
import re


def bind_layout(seed_manager,selected,inventory,data_root):
    parent=Path(selected['path']).parent
    pattern=re.compile(re.escape(selected['task'])+r'_(\d+)\.json')
    rows=[]
    for row in inventory:
        path=Path(row['path']);match=pattern.fullmatch(path.name)
        if row['type']=='file' and path.parent==parent and match:
            rows.append((int(match.group(1)),row['path']))
    rows.sort(key=lambda row:row[0])
    if [suffix for suffix,_ in rows]!=selected['candidate_layouts']:
        raise ValueError('Pinned layout catalog differs from frozen candidate list')
    index=selected['candidate_layouts'].index(selected['layout'])
    if rows[index][1]!=selected['path']:
        raise ValueError('Selected filename differs from frozen layout path')
    seed_manager.seed_info={rank:{'scene_layout':str(data_root/path)} for rank,(_,path) in enumerate(rows)}
    return index
