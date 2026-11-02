"""Real OS locks with a local child process; no simulator/model/network."""
import os,subprocess,sys,tempfile,time,unittest
from pathlib import Path
from unittest.mock import patch
from kyb import resources

class ResourcesTests(unittest.TestCase):
    def test_third_process_waits_then_enters_after_release(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);experiment=root/'experiment';experiment.mkdir();marker=root/'entered'
            with patch.object(resources,'LEASE_ROOT',root/'leases'):
                with resources.episode_lease(0,experiment):
                    with resources.episode_lease(0,experiment):
                        code="from pathlib import Path;from kyb import resources;resources.LEASE_ROOT=Path(%r);\nwith resources.episode_lease(0,Path(%r)): Path(%r).write_text('entered')"%(str(root/'leases'),str(experiment),str(marker))
                        child=subprocess.Popen([sys.executable,'-c',code],stdout=subprocess.PIPE,stderr=subprocess.PIPE)
                        try:
                            time.sleep(.7);self.assertIsNone(child.poll());self.assertFalse(marker.exists())
                        except BaseException:child.kill();child.communicate();raise
                    output,error=child.communicate(timeout=4)
                    self.assertEqual(child.returncode,0,error);self.assertTrue(marker.exists())
                    self.assertIn(b'waiting_gpu_episode_slot',output)
    def test_stop_before_acquisition_never_enters_episode(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'STOP').write_text('stop')
            with patch.object(resources,'LEASE_ROOT',root/'leases'):
                with self.assertRaisesRegex(RuntimeError,'Stopped while waiting'):
                    with resources.episode_lease(0,root):self.fail('Must not enter')
    def test_exception_releases_os_lock(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            with patch.object(resources,'LEASE_ROOT',root/'leases'),patch.object(resources,'MAX_CONCURRENT_KIT',1):
                with self.assertRaises(ValueError):
                    with resources.episode_lease(0,root):raise ValueError('fixture failure')
                with resources.episode_lease(0,root):pass

    def test_live_child_keeps_lease_after_coordinator_closes_handle(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);ready=root/'ready';release=root/'release'
            code="import time;from pathlib import Path;Path(%r).touch();\nwhile not Path(%r).exists(): time.sleep(.05)"%(str(ready),str(release))
            with patch.object(resources,'LEASE_ROOT',root/'leases'):
                with resources.episode_lease(0,root) as fd:
                    child=subprocess.Popen([sys.executable,'-c',code],pass_fds=(fd,))
                    limit=time.monotonic()+3
                    while not ready.exists() and time.monotonic()<limit:time.sleep(.05)
                    self.assertTrue(ready.exists())
                # The coordinator has closed its descriptor; the live child still
                # owns the same open file description, so the second slot is used.
                with resources.episode_lease(0,root) as fd:
                    import fcntl
                    with (root/'leases/gpu0/slot0.lock').open('a+') as slot:
                        with self.assertRaises(BlockingIOError):fcntl.flock(slot,fcntl.LOCK_EX|fcntl.LOCK_NB)
                release.touch();child.wait(timeout=3)
                with resources.episode_lease(0,root):pass
