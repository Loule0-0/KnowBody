"""Native failure keeps its caller alive; only an exact operator reply resumes."""
import json,os,tempfile,time,threading,unittest
from pathlib import Path
from test_codex_cli import FAKE
from kyb.codex_cli import CodexCliClient
from kyb.budget import CallBudget

class CapacityWait(unittest.TestCase):
    def test_real_local_transport_wait_has_no_requests_until_explicit_resume(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);exe=root/'codex';exe.write_text(FAKE);exe.chmod(0o700)
            budget=CallBudget(root/'budget.jsonl',3)
            client=CodexCliClient(root/'run',executable=str(exe),budget=budget)
            tools=[{'function':{'name':'move_eef','parameters':{}}}]
            result=[];errors=[]
            def call():
                try:result.append(client.complete([{'role':'user','content':'current observation'}],tools))
                except BaseException as error:errors.append(error)
            os.environ['KYB_FAKE_MODE']='capacity'
            worker=threading.Thread(target=call,daemon=True);worker.start()
            pending=client.root/'CAPACITY_WAIT.json'
            try:
                end=time.monotonic()+5
                while not pending.exists() and time.monotonic()<end:time.sleep(.01)
                self.assertTrue(pending.exists());state=json.loads(pending.read_text())
                self.assertEqual(state['status'],'WAITING_FOR_CAPACITY_RESUME')
                self.assertEqual(budget.view()['used'],1)
                time.sleep(.3);self.assertEqual(budget.view()['used'],1);self.assertTrue(worker.is_alive())
                control=Path(state['answer_path']);control.write_text(json.dumps({'failed_attempt':'wrong','action':'resume','provided_by':'operator'}))
                time.sleep(.3);self.assertEqual(budget.view()['used'],1)
                os.environ.pop('KYB_FAKE_MODE',None)
                control.write_text(json.dumps({'failed_attempt':state['failed_attempt'],'action':'resume','provided_by':'operator'}))
                worker.join(5);self.assertFalse(worker.is_alive());self.assertFalse(errors)
                self.assertEqual(len(result),1);self.assertEqual(budget.view()['used'],2)
                self.assertEqual(client.session_id,state['native_session_id'])
                self.assertEqual(client.summary()['outcome_unknown'],0)
                self.assertEqual(client.summary()['native_failed_turns'],1)
            finally:
                os.environ.pop('KYB_FAKE_MODE',None)
                if worker.is_alive() and pending.exists():
                    state=json.loads(pending.read_text());Path(state['answer_path']).write_text(json.dumps({'failed_attempt':state['failed_attempt'],'action':'stop','provided_by':'operator'}));worker.join(5)

if __name__=='__main__':unittest.main()
