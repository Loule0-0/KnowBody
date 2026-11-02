"""Host-only GPU episode leases shared by all three independent Kit studies."""
from contextlib import contextmanager
import fcntl,json,time,os
from pathlib import Path

# The measured first Kit used ~8.2GiB on the24GiB4090D. Three such instances
# leave no peak margin. Two independent instances may advance, the third waits
# without model calls or simulation and competes at the next episode boundary.
MAX_CONCURRENT_KIT = 2
LEASE_ROOT = Path('/data/KnowYourBody_RoboDojo/runtime/gpu_episode_leases')

@contextmanager
def episode_lease(gpu, experiment):
    root=LEASE_ROOT/f'gpu{gpu}';root.mkdir(parents=True,exist_ok=True)
    handles=[(root/f'slot{i}.lock').open('a+') for i in range(MAX_CONCURRENT_KIT)]
    chosen=None;announced=False
    try:
        while chosen is None:
            if (Path(experiment)/'STOP').exists():
                raise RuntimeError('Stopped while waiting for an episode GPU slot; no model or environment started')
            for handle in handles:
                try:fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
                except BlockingIOError:continue
                chosen=handle;break
            if chosen is None:
                if not announced:
                    print(json.dumps({'event':'waiting_gpu_episode_slot','gpu':gpu,'maximum_active':MAX_CONCURRENT_KIT,'experiment':str(experiment),'model_calls':0,'simulation_steps':0}),flush=True)
                    announced=True
                time.sleep(.5)
        print(json.dumps({'event':'gpu_episode_slot_acquired','gpu':gpu,'slot':str(chosen.name),'experiment':str(experiment)}),flush=True)
        yield chosen.fileno()
    finally:
        for handle in handles:handle.close()


def seal_kit_lease(fd):
    """Keep the live Kit lock, but do not pass it through exec to crash dialogs."""
    os.set_inheritable(fd,False)
