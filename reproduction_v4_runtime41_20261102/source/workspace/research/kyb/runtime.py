"""Explicit experiment resources and exact task/layout selection."""
from pathlib import Path


def configuration(protocol):
    runtime=protocol['runtime']
    if set(runtime)!={'root','server_port','gpu'}:raise ValueError('runtime requires root, server_port and gpu')
    root=Path(runtime['root'])
    if not root.is_absolute():raise ValueError('Runtime root must be absolute')
    port,gpu=runtime['server_port'],runtime['gpu']
    if type(port) is not int or not 1024<=port<=65535:raise ValueError('Explicit unprivileged server port required')
    if type(gpu) is not int or gpu<0:raise ValueError('Explicit nonnegative GPU ordinal required')
    return root,port,gpu


def select(layouts,task,layout):
    matches=[row for row in layouts['tasks'] if (row['task'],row['layout'])==(task,layout)]
    if len(matches)!=1:raise ValueError('Task/layout must identify one frozen experiment entry')
    return matches[0]


def call_limits(protocol):
    limits={role:protocol[role]['calls_per_episode'] for role in ('primary','manager')}
    if any(type(value) is not int or value<1 for value in limits.values()):
        raise ValueError('Explicit positive integer limits for both model roles required')
    return limits
