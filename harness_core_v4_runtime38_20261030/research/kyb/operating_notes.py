"""Render public SDK operating notes for the actual host perception surface."""
PROJECTION_CLAIM = ('The current body projections are available through read_body and do not prove\n'
    'object contact, clearance or retention.')

def for_perception(source_docs, perception):
    if source_docs.count(PROJECTION_CLAIM) != 1:
        raise ValueError('Pinned public body-query operating notes changed; review the active tool contract')
    if 'read_body' in perception.forbidden_tools:
        capability=('Current RGB and measured robot state are supplied directly. No body projection, '
            'depth or camera-calibration query is available. These observations do not establish '
            'object contact, clearance or retention.')
    else:
        capability=PROJECTION_CLAIM
    return source_docs.replace(PROJECTION_CLAIM, capability)
