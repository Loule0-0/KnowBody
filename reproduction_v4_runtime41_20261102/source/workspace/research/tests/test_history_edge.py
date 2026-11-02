"""A maintenance edge must not conceal a changed or unrelated memory chain."""
import json, tempfile, unittest
import experiment_monitor as monitor
from unittest.mock import patch
from pathlib import Path
from experiment_monitor import check_memory_edge, sha

class HistoryEdge(unittest.TestCase):
    def make(self, root):
        source, target, proof = [root/name for name in ('source.sqlite','target.sqlite','proof.json')]
        source.write_bytes(b'old frozen history');target.write_bytes(b'same history plus index')
        data={'verdict':'PASS','source':str(source),'target':str(target),'before_order':24,
            'source_sha256':sha(source),'target_sha256':sha(target),
            'authorization':'User2026-10-26: adapt previous own-chain experience to v4'}
        proof.write_text(json.dumps(data))
        edge={**{k:data[k] for k in ('source','target','source_sha256','target_sha256')},
            'before_order':24,'proof':str(proof),'proof_sha256':sha(proof)}
        return source,target,proof,edge
    def test_declared_edge_only_at_its_real_boundary(self):
        with tempfile.TemporaryDirectory() as folder:
            source,target,proof,edge=self.make(Path(folder))
            check_memory_edge(str(source),str(target),24,[edge])
            for order,edges in [(23,[edge]),(24,[]),(24,[edge,edge])]:
                with self.assertRaises(ValueError):check_memory_edge(str(source),str(target),order,edges)
    def test_current_ready_or_active_tail_is_checked(self):
        # Exercise monitor.main's real tail wiring with no completed rows/models.
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);source,target,proof,edge=self.make(root)
            (root/'PROTOCOL.json').write_text(json.dumps({'state_relative_path':'STATE.json',
                'coordinator_service':'test','primary':{'calls_per_episode':100},'manager':{'calls_per_episode':100}}))
            (root/'LAYOUTS.json').write_text(json.dumps({'tasks':[{'task':'test'}]}))
            state={'status':'READY','completed':[],'next_order':24,'memory':str(target),
                'initial_memory_snapshot':str(source),'active':None,'memory_adaptations':[]}
            (root/'STATE.json').write_text(json.dumps(state))
            import subprocess
            with patch.object(monitor.subprocess,'run',return_value=subprocess.CompletedProcess([],0,stdout='ActiveState=active')):
                monitor.main(root)
                result=json.loads((root/'monitor/LATEST.json').read_text())
                self.assertTrue(any('Current memory chain differs' in x for x in result['issues']))
                state['memory_adaptations']=[edge];(root/'STATE.json').write_text(json.dumps(state))
                monitor.main(root)
                result=json.loads((root/'monitor/LATEST.json').read_text())
                self.assertEqual(result['health'],'PASS')
    def test_changed_snapshot_and_changed_proof_are_detected(self):
        with tempfile.TemporaryDirectory() as folder:
            source,target,proof,edge=self.make(Path(folder))
            target.write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'snapshot changed'):
                check_memory_edge(str(source),str(target),24,[edge])
            target.write_bytes(b'same history plus index');proof.write_text('{}')
            with self.assertRaisesRegex(ValueError,'proof changed'):
                check_memory_edge(str(source),str(target),24,[edge])
    def test_relabelled_edge_without_authority_fails(self):
        with tempfile.TemporaryDirectory() as folder:
            source,target,proof,edge=self.make(Path(folder))
            data=json.loads(proof.read_text());data['authorization']='assumed'
            proof.write_text(json.dumps(data));edge['proof_sha256']=sha(proof)
            with self.assertRaisesRegex(ValueError,'authorization absent'):
                check_memory_edge(str(source),str(target),24,[edge])

if __name__=='__main__':unittest.main()
