"""Portable core regressions using fake CLI/manager, isolated leases and native state."""
import contextlib
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

B=Path(__file__).resolve().parents[1]
W=B/'source/workspace';R=B/'source/robodojo_overlay'
paths=list(map(str,[W/'research',W/'research/tests',R,R/'XPolicyLab']))
sys.path[:0]=paths;os.environ['PYTHONPATH']=os.pathsep.join(paths)
from kyb import resources,native_dispatch
from kyb.session import MemorySession

MODULES=['test_memory','test_hybrid_retrieval','test_source_scope','test_outline_size',
         'test_manager_continuity','test_case_witness','test_codex_cli','test_native_usage']


class RankingDouble:
    def rank(self,question,documents,**kwargs):return sorted(documents)[:20],{'encoder':'offline-test-double'}
    def close(self):pass


def main():
    native_dispatch.MIN_START_INTERVAL_SECONDS=.001
    with tempfile.TemporaryDirectory(prefix='kyb-package-offline-') as tmp:
        os.environ['CODEX_HOME']=str(Path(tmp)/'native');resources.LEASE_ROOT=Path(tmp)/'leases'
        suite=unittest.defaultTestLoader.loadTestsFromNames(MODULES)
        with patch.object(MemorySession,'get_retrieval_index',lambda self:self.retrieval_index or RankingDouble()):
            result=unittest.TextTestRunner(verbosity=2).run(suite)
        print('PASS' if result.wasSuccessful() else 'NEEDS_WORK','portable offline tests',result.testsRun)
        return 0 if result.wasSuccessful() else 1


if __name__=='__main__':raise SystemExit(main())
