"""A real exec descendant must not retain a finished Kit's GPU slot."""
import fcntl
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch
from kyb import resources

class LeaseExecIsolationTests(unittest.TestCase):
    def test_live_sealed_worker_retains_lock_after_coordinator_closes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);release=root/'release'
            script="""import sys,time
from pathlib import Path
from kyb.resources import seal_kit_lease
seal_kit_lease(int(sys.argv[1]))
print('ready',flush=True)
while not Path(sys.argv[2]).exists():time.sleep(.03)
"""
            with patch.object(resources,'LEASE_ROOT',root/'leases'):
                with resources.episode_lease(0,root) as fd:
                    child=subprocess.Popen([sys.executable,'-c',script,str(fd),str(release)],pass_fds=(fd,),stdout=subprocess.PIPE,text=True)
                    self.assertEqual(child.stdout.readline().strip(),'ready')
                try:
                    with (root/'leases/gpu0/slot0.lock').open('a+') as slot:
                        with self.assertRaises(BlockingIOError):fcntl.flock(slot,fcntl.LOCK_EX|fcntl.LOCK_NB)
                finally:release.touch();child.wait(timeout=4);child.stdout.close()
                with (root/'leases/gpu0/slot0.lock').open('a+') as slot:fcntl.flock(slot,fcntl.LOCK_EX|fcntl.LOCK_NB)

    def test_exec_dialog_outlives_worker_but_cannot_keep_lease(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);release=root/'release'
            dialog="import sys,time;from pathlib import Path;\nwhile not Path(sys.argv[1]).exists():time.sleep(.03)"
            worker="""import subprocess,sys
from kyb.resources import seal_kit_lease
seal_kit_lease(int(sys.argv[1]))
p=subprocess.Popen([sys.executable,'-c',sys.argv[2],sys.argv[3]],close_fds=False,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
print(p.pid,flush=True)
"""
            with patch.object(resources,'LEASE_ROOT',root/'leases'):
                with resources.episode_lease(0,root) as fd:
                    child=subprocess.Popen([sys.executable,'-c',worker,str(fd),dialog,str(release)],pass_fds=(fd,),stdout=subprocess.PIPE,text=True)
                    pid=int(child.stdout.readline());child.wait(timeout=4);child.stdout.close()
                try:
                    os.kill(pid,0)  # It really remains alive after worker/coordinator release.
                    with (root/'leases/gpu0/slot0.lock').open('a+') as slot:fcntl.flock(slot,fcntl.LOCK_EX|fcntl.LOCK_NB)
                finally:
                    release.touch()
                    deadline=time.monotonic()+3
                    while (Path('/proc')/str(pid)).exists() and time.monotonic()<deadline:
                        if (Path('/proc')/str(pid)/'stat').read_text().split()[2]=='Z':break
                        time.sleep(.03)
