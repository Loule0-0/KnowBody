"""Codex CLI policy transport; context and robot execution remain host-owned."""
from __future__ import annotations

import base64
import hashlib
import json
import os
import shutil
import signal
import subprocess
import time
import uuid
from pathlib import Path

from .native_dispatch import NativeDispatch
from .trace import append, write_once
from .context import latest_context
from .native_usage import read_rollout_usage, merge_usage_snapshots, transport_notices, summarize_role


class CodexCliFailure(RuntimeError):
    pass


class CodexCliClient:
    # One explicit native session per policy episode or manager transaction.
    disabled = ("shell_tool", "unified_exec", "apps", "plugins", "remote_plugin",
                "skill_search", "view_image", "multi_agent", "browser_use",
                "computer_use", "image_generation", "tool_suggest", "sleep_tool", "hooks",
                "memories", "code_mode", "code_mode_host", "workspace_dependencies", "goals",
                "browser_use_external", "in_app_browser")

    def __init__(self, root: Path, *, executable: str = "codex", timeout: float = 240,
                 role: str = "policy", budget=None):
        self.root = Path(root)
        self.executable = shutil.which(executable)
        if self.executable is None:
            raise CodexCliFailure("Codex CLI is not installed")
        self.timeout = timeout
        self.role = role
        self.model, self.effort = {"policy": ("gpt-6-astra", "medium"),
                                   "manager": ("gpt-6.1-sol", "xhigh")}[role]
        self.last_attempt_id = None
        self.budget = budget
        self.session_id = None
        self._messages_ref = None
        self._submitted = 0
        self._tools_signature = None
        self._failed = False
        self._last_native = None
        self.knowledge_provider = None
        self.operator_contract = None
        self._host_system = ""
        self.codex_home = Path(os.environ.get("CODEX_HOME", str(Path.home() / ".codex")))
        self.version = subprocess.run([self.executable, "--version"], check=True,
            capture_output=True, text=True).stdout.strip()
        self.workspace = self.root / f"{role}_workspace"
        self.workspace.mkdir(parents=True, exist_ok=True, mode=0o700)
        (self.workspace / "images").mkdir(exist_ok=True)

    def _state(self, attempt, state, **fields):
        append(self.root / "cli_calls.jsonl", {"attempt_id": attempt, "state": state,
            "role": self.role, "model": self.model, "effort": self.effort,
            "time": time.time(), **fields})

    def start_transaction(self):
        if self.role != 'manager':
            raise ValueError('Policy session is episode-scoped and cannot be reset')
        if self._failed:
            raise CodexCliFailure('Previous transaction failed; no automatic new session')
        self.session_id = None
        self._messages_ref = None
        self._submitted = 0
        self._tools_signature = None

    def _input(self, messages, directory):
        """Only newly supplied host messages/images; native history owns continuity."""
        start = self._submitted if messages is self._messages_ref else 0
        if start > len(messages):
            raise CodexCliFailure('Host conversation shortened; explicit continuity lost')
        selected = [(i,m) for i,m in enumerate(messages) if i>=start and m.get('role') not in ('assistant','system')
            and not (self.role=='policy' and isinstance(m.get('content'),str) and m['content'].startswith('<KYB_PREPARATION>'))]
        write_once(directory / 'context_selection.json', {
            'new_host_messages':len(selected), 'previously_submitted_messages':start,
            'native_session_id':self.session_id,
            'rule':'append_new_host_messages_only_no_reconstructed_native_history'})
        history, images, manifest, image_cache = [], [], [], {}
        for index,message in selected:
            copied={k:v for k,v in message.items() if k!='content'}
            content=message.get('content')
            if not isinstance(content,list):
                copied['content']=content
            else:
                parts=[]; reference={}
                for part in content:
                    if part.get('type')!='image_url':
                        copy=dict(part)
                        if self.role=='policy' and '\n<KYB_CONTEXT>' in copy.get('text',''):
                            copy['text']=copy['text'].split('\n<KYB_CONTEXT>',1)[0]
                        parts.append(copy)
                        if part.get('text','').startswith('RGB_REFERENCE '):
                            reference=json.loads(part['text'].removeprefix('RGB_REFERENCE '))
                        continue
                    uri=part['image_url']['url']
                    if not uri.startswith('data:image/jpeg;base64,'):
                        raise CodexCliFailure('Only explicitly supplied JPEG observations accepted')
                    raw=base64.b64decode(uri.split(',',1)[1],validate=True)
                    digest=hashlib.sha256(raw).hexdigest()
                    path=self.workspace/'images'/f'{digest}.jpg'
                    if not path.exists():path.write_bytes(raw)
                    elif hashlib.sha256(path.read_bytes()).hexdigest()!=digest:
                        raise CodexCliFailure('Immutable image cache changed')
                    if digest not in image_cache:
                        images.append(path);image_cache[digest]=len(images)
                    identity={**reference,'attachment':image_cache[digest],
                        'temporal_role':'explicit_evidence' if self.role=='manager' else 'new_observation'}
                    parts.append({'type':'supplied_image',**identity})
                    manifest.append({**identity,'sha256':digest,'path':str(path),'message_index':index})
                copied['content']=parts
            history.append(copied)
        write_once(directory/'image_manifest.json',manifest)
        return history,images

    def _archive(self, directory):
        events, malformed = [], 0
        path = directory / "events.jsonl"
        for line in path.read_text().splitlines() if path.exists() else []:
            try:
                events.append(json.loads(line))
            except ValueError:
                malformed += 1  # Interrupted trailing line remains in raw file.
        thread = next((e["thread_id"] for e in events if e["type"] == "thread.started"), None)
        if thread:
            rollouts = list((self.codex_home / "sessions").glob(f"*/*/*/*{thread}.jsonl"))
            if len(rollouts) == 1:
                shutil.copyfile(rollouts[0], directory / "native_rollout.jsonl")
        completed = [e for e in events if e["type"] == "turn.completed"]
        failed = [e for e in events if e["type"] == "turn.failed"]
        terminals = [e["type"] for e in events
                     if e["type"] in {"turn.completed", "turn.failed"}]
        items = [e["item"] for e in events if "item" in e]
        unexpected = [i.get("type") for i in items if i.get("type") not in {
            "agent_message", "reasoning", "error"}]
        result = {"thread_id": thread, "usage": completed[-1].get("usage") if completed else None,
            "usage_scope": "native_session_cumulative",
            "completed_turns": len(completed), "unexpected_item_types": unexpected,
            "failed_turns": len(failed),
            "explicit_failure": bool(failed and terminals[-1] == "turn.failed"
                and not completed and thread and not malformed and not unexpected
                and not (directory / "final.json").exists()),
            "terminal_failure": failed[-1].get("error") if failed else None,
            "diagnostics": [i for i in items if i.get("type") == "error"],
            "malformed_event_lines": malformed}
        rollout=directory / "native_rollout.jsonl"
        try:
            snapshot=read_rollout_usage(rollout) if rollout.exists() else {
                "issues":[{"code":"native_archive_unavailable"}]}
        except (OSError,ValueError,TypeError) as error:
            snapshot={"issues":[{"code":"native_archive_unreadable","error_type":type(error).__name__}]}
        result["native_response_usage"]=merge_usage_snapshots([snapshot])
        result["transport_notices"]=transport_notices(events)
        write_once(directory / "native_summary.json", result)
        return result

    def _capacity_resume(self, native):
        """Keep the caller/scene alive; never infer a retry authorization from time."""
        folder=self.root/'capacity_wait';folder.mkdir(exist_ok=True)
        control=folder/(self.last_attempt_id+'.answer.json')
        state={'status':'WAITING_FOR_CAPACITY_RESUME','failed_attempt':self.last_attempt_id,
            'role':self.role,'model':self.model,'effort':self.effort,
            'native_session_id':native['thread_id'],'answer_path':str(control),
            'reason':native['terminal_failure'],'budget':self.budget.view() if self.budget else None,
            'model_calls_while_waiting':0,'simulation_actions_while_waiting':0}
        pending=self.root/'CAPACITY_WAIT.json'
        def save():
            temporary=pending.with_suffix('.tmp');temporary.write_text(json.dumps(state,ensure_ascii=False));os.replace(temporary,pending)
        save()
        while True:
            if control.exists():
                try:
                    response=json.loads(control.read_text())
                    valid=(set(response)=={'failed_attempt','action','provided_by'} and
                        response['failed_attempt']==state['failed_attempt'] and
                        response['provided_by']=='operator' and response['action'] in ('resume','stop'))
                except (ValueError,TypeError):valid=False
                if valid:
                    state['status']='RESUME_AUTHORIZED' if response['action']=='resume' else 'STOP_AUTHORIZED'
                    state['answer_sha256']=hashlib.sha256(control.read_bytes()).hexdigest();save()
                    return response['action']=='resume'
            time.sleep(.25)

    def complete(self, messages, tools):
        while True:
            try:
                return self._complete_with_capacity_model(messages,tools)
            except CodexCliFailure:
                native=self._last_native or {}
                failure=native.get('terminal_failure')
                reason=failure.get('message','') if isinstance(failure,dict) else ''
                capacity=any(t in reason.lower() for t in ('at capacity','429 too many requests','overloaded','503 service unavailable'))
                if not (native.get('explicit_failure') and capacity and native.get('thread_id')
                        and (self.budget is None or self.budget.view()['remaining']>0)):
                    raise
                if self.session_id is not None and self.session_id!=native['thread_id']:raise
                if not self._capacity_resume(native):raise
                self.session_id=native['thread_id']
                self._failed=False

    def _complete_with_capacity_model(self, messages, tools):
        """One authorized model change for an explicit manager capacity rejection.

        The failed request and replacement both consume the same role allowance.
        The native transaction and host evidence permissions remain intact. No
        result was accepted from the failed turn; only its unaccepted input delta
        is resubmitted. Unknown, malformed or action-policy failures never switch.
        """
        try:
            return self._complete_once(messages, tools)
        except CodexCliFailure:
            native = self._last_native or {}
            failure = native.get('terminal_failure')
            reason = failure.get('message', '') if isinstance(failure, dict) else ''
            capacity = any(text in reason.lower() for text in (
                'at capacity', '429 too many requests', 'overloaded', '503 service unavailable'))
            if not (self.role == 'manager' and self.model == 'gpt-6.1-sol'
                    and native.get('explicit_failure') and capacity
                    and native.get('thread_id')
                    and (self.budget is None or self.budget.view()['remaining'] > 0)):
                raise
            if self.session_id is not None and self.session_id != native['thread_id']:
                raise
            append(self.root / 'model_switches.jsonl', {
                'at': time.time(), 'failed_attempt': self.last_attempt_id,
                'native_session_id': native['thread_id'], 'reason': reason,
                'from_model': self.model, 'from_effort': self.effort,
                'to_model': 'gpt-6-astra', 'to_effort': 'high',
                'budget': self.budget.view() if self.budget is not None else None,
                'scope': 'User-authorized capacity model change; same host transaction and native SID; only unaccepted input delta resubmitted; no host tool replay'})
            self.model, self.effort = 'gpt-6-astra', 'high'
            self.session_id = native['thread_id']
            self._failed = False
            return self._complete_once(messages, tools)

    def _complete_once(self, messages, tools):
        if self._failed:
            raise CodexCliFailure('Previous call failed; no automatic retry or new session')
        if self.budget is not None and self.budget.view()['remaining']==0:
            from .budget import BudgetExhausted
            raise BudgetExhausted('Assigned model allowance exhausted; no request sent')
        while ((self.root.parent if self.role=='manager' else self.root)/'PAUSE').exists():
            time.sleep(.25)
        self._last_native = None
        attempt = uuid.uuid4().hex
        self.last_attempt_id = attempt
        directory = self.root / "cli" / attempt
        directory.mkdir(parents=True, mode=0o700)
        inputs = directory / "input"
        inputs.mkdir()
        history, images = self._input(messages, inputs)
        names = [t["function"]["name"] for t in tools]
        schema = {"type": "object", "properties": {
            "tool": {"type": "string", "enum": names},
            "arguments_json": {"type": "string", "description":
                "A JSON object encoded as a string, matching the chosen robot tool's parameters."}},
            "required": ["tool", "arguments_json"], "additionalProperties": False}
        write_once(inputs / "schema.json", schema)
        instructions = (
            ("You are a visual robot policy. " if self.role == "policy" else
             "You manage conditional robot knowledge, not robot actions. ") +
            "Use supplied host messages, prior native session context and numbered RGB "
            "attachments as task evidence. The host role/action contract below defines the embodiment, "
            "and its user entries contain actual observations and conditional memory. Previous tool "
            "results describe planning; only explicit execution receipts prove execution. "
            "Do not use coding tools, external tools, skills, filesystem, or external information. "
            "Return exactly one supplied tool as the final structured JSON. The host alone "
            "validates it. Management tools cannot execute robot actions. "
            "Previously supplied images may be compacted; when exact historical pixels are unavailable "
            "do not pretend to inspect them now. Use exact observation IDs for memory evidence, never "
            "temporary attachment numbers. The tool calls described in the supplied history are "
            "host tools, not Codex tools.")
        host_system='\n'.join(m['content'] for m in messages if m.get('role')=='system' and isinstance(m.get('content'),str))
        if host_system:self._host_system=host_system
        instructions+='\nHOST ROLE AND CONTRACT:\n'+self._host_system
        preparation = self.knowledge_provider() if self.knowledge_provider is not None else None
        preparation_sha256=None
        if preparation is not None:
            write_once(inputs/'preparation.json',preparation)
            preparation_sha256=hashlib.sha256(json.dumps(preparation,ensure_ascii=False).encode()).hexdigest()
            instructions+='\nCURRENT STABLE PREPARATION (not current physical state):\n'+json.dumps(preparation,ensure_ascii=False)
        # The complete host contract is resident on every resume, not dependent on
        # an old user message surviving native compaction. Schemas have one source.
        contract=[tool['function'] for tool in tools]
        instructions+='\nAVAILABLE HOST TOOLS (name, description, complete parameter schema):\n'+json.dumps(contract,ensure_ascii=False,separators=(',',':'))
        (inputs / "instructions.txt").write_text(instructions)
        signature = json.dumps(tools,sort_keys=True)
        payload = {"conversation":history}
        if self.budget is not None:
            payload['call_budget'] = self.budget.view()
            payload['call_budget']['remaining_after_this_call']=max(0,payload['call_budget']['remaining']-1)
        if self.role == 'policy':
            context = latest_context(messages)
            if context: payload['current_readout'] = context
        prompt = json.dumps(payload,ensure_ascii=False)
        (inputs / "prompt.txt").write_text(prompt)
        call_tools={c['id']:c['function']['name'] for m in messages for c in (m.get('tool_calls') or [])}
        experience_bytes={}
        for message in history:
            name=call_tools.get(message.get('tool_call_id'))
            if message.get('role')=='tool' and name in ('query_experience','query_body_history','manage_memory'):
                experience_bytes[name]=experience_bytes.get(name,0)+len(json.dumps(message['content'],ensure_ascii=False).encode())
        write_once(inputs / 'context_metrics.json', {
            'role': self.role, 'native_session_id': self.session_id,
            'instructions_utf8_bytes': len(instructions.encode()),
            'new_prompt_utf8_bytes': len(prompt.encode()),
            'tools_utf8_bytes': len(json.dumps(contract,ensure_ascii=False,separators=(',',':')).encode()),
            'preparation_utf8_bytes': len(json.dumps(preparation,ensure_ascii=False).encode()) if preparation is not None else 0,
            'operator_contract_utf8_bytes': len(json.dumps(preparation.get('operator_clarifications',{}),ensure_ascii=False).encode()) if preparation is not None else
                len(json.dumps(self.operator_contract,ensure_ascii=False).encode()) if self.operator_contract is not None else 0,
            'experience_preparation_utf8_bytes': preparation.get('packing',{}).get('experience_serialized_bytes',0) if preparation is not None else 0,
            'images': len(images),
            'preparation_sha256':preparation_sha256,
            'new_experience_result_utf8_bytes_by_tool': experience_bytes,
            'scope': 'Actual host payload for this call; excludes native retained/compacted context and image tokens. Token usage is separately reported by CLI.'})
        command = [self.executable, "exec", "--ignore-user-config", "--strict-config",
            "--skip-git-repo-check", "--json", "--color", "never", "--sandbox", "read-only",
            "-C", str(self.workspace), "-m", self.model]
        settings = {"model_reasoning_effort": self.effort, "approval_policy": "never",
            "forced_login_method": "chatgpt",
            "project_doc_max_bytes": 0, "web_search": "disabled",
            "memories.use_memories": False, "memories.generate_memories": False,
            "features.unbounded_connection_retries": False,
            "model_instructions_file": str(inputs / "instructions.txt")}
        for key, value in settings.items():
            command += ["-c", f"{key}={json.dumps(value)}"]
        for feature in self.disabled:
            command += ["--disable", feature]
        command += ["--enable", "skip_host_skill_discovery"]
        if self.session_id:
            command += ["resume",self.session_id]
        command += [
            "--output-schema", str(inputs / "schema.json"),
            "--output-last-message", str(directory / "final.json")]
        for path in images:
            command += ["--image", str(path)]
        command += ["--", "-"]  # Keep the stdin sentinel out of --image's variadic values.
        # Nested Codex must not inherit this desktop thread's remote payload,
        # instructions, permissions, or thread identity. Authentication stays native.
        child_env = {k: v for k, v in os.environ.items() if not k.startswith("CODEX_")}
        child_env["CODEX_HOME"] = str(self.codex_home)
        write_once(directory / "invocation.json", {"argv": command, "version": self.version,
            "role": self.role, "model": self.model, "effort": self.effort,
            "context": "native_episode_or_transaction_session_incremental",
            "resumed_session_id":self.session_id,"new_prompt_bytes":len(prompt.encode()),
            "native_instruction_bytes":len(instructions.encode()),
            "image_count": len(images),
            "http_attempts": None, "internal_retries": "not_observable_from_exec_events"})
        append(self.root / 'native_dispatch_waits.jsonl', {
            'attempt_id': attempt, 'at': time.time(), 'state': 'waiting',
            'scope': 'Harness clients sharing CODEX_HOME; no model call reserved yet'})
        # Acquisition, including interrupted waiting, is outside the request's
        # UNKNOWN handling and timeout. The lock ends immediately after Popen.
        pause_path = (self.root.parent if self.role == 'manager' else self.root) / 'PAUSE'
        admitted = False
        try:
            with NativeDispatch(self.codex_home, pause_path=pause_path) as dispatch:
                admitted = True
                append(self.root / 'native_dispatch_waits.jsonl', {
                    'attempt_id': attempt, 'at': time.time(), 'state': 'admitted',
                    'wait_seconds': dispatch.wait_seconds,
                    'qualification': 'Total pre-dispatch wait, including lock/cooldown and any operator PAUSE; not exclusively pacing time'})
                self._state(attempt, "prepared")
                started = time.monotonic()
                process = None
                native = None
                try:
                    with (directory / "events.jsonl").open("wb") as stdout, (directory / "stderr.log").open("wb") as stderr:
                        if self.budget is not None:
                            self.budget.reserve(attempt,self.role)
                        dispatch.prepare_launch(attempt)
                        try:
                            process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=stdout,
                                stderr=stderr, env=child_env, start_new_session=True)
                        except OSError:
                            dispatch.spawn_rejected()
                            raise
                        dispatch.process_started()
                        self._state(attempt, "process_started", pid=process.pid,
                                    pre_dispatch_wait_seconds=dispatch.wait_seconds)
                        try:
                            process.communicate(prompt.encode(), timeout=self.timeout)
                        except subprocess.TimeoutExpired:
                            os.killpg(process.pid, signal.SIGTERM)
                            try:
                                process.wait(timeout=5)
                            except subprocess.TimeoutExpired:
                                os.killpg(process.pid, signal.SIGKILL)
                                process.wait()
                            raise CodexCliFailure("CLI timed out; outcome unknown; no automatic retry")
                    native = self._archive(directory)
                    thread, usage = native["thread_id"] or self.session_id, native["usage"]
                    if self.session_id is not None and thread != self.session_id:
                        raise CodexCliFailure('Resume returned a different session; no action accepted')
                    if thread is None:
                        raise CodexCliFailure('Native session identity missing')
                    if process.returncode or not native["completed_turns"] or native["unexpected_item_types"] or not (directory / "final.json").exists():
                        raise CodexCliFailure("CLI incomplete or used an unexpected tool; inspect local evidence")
                    result = json.loads((directory / "final.json").read_text())
                    arguments = json.loads(result["arguments_json"])
                    if result["tool"] not in names or not isinstance(arguments, dict):
                        raise CodexCliFailure("CLI returned an invalid robot tool object")
                    response = {"id": f"codex:{thread}:{attempt}", "model": self.model,
                        "choices": [{"message": {"role": "assistant", "content": None,
                            "tool_calls": [{"id": attempt, "type": "function", "function": {
                                "name": result["tool"], "arguments": json.dumps(arguments)}}]},
                            "finish_reason": "tool_calls"}], "usage": usage or {}}
                    write_once(directory / "visible_response.json", response)
                    self._state(attempt, "turn_complete", thread_id=thread, usage=usage,
                                native_completed_turns=native["completed_turns"],
                                wall_seconds=time.monotonic()-started)
                    self.session_id=thread
                    self._messages_ref=messages
                    self._submitted=len(messages)
                    self._tools_signature=signature
                    return response
                except BaseException as error:
                    self._failed=True
                    if process is not None and process.poll() is None:
                        os.killpg(process.pid, signal.SIGTERM)
                        try:
                            process.wait(timeout=5)
                        except subprocess.TimeoutExpired:
                            os.killpg(process.pid, signal.SIGKILL)
                            process.wait()
                    if native is None:
                        native = self._archive(directory)
                    self._last_native = native
                    self._state(attempt, "pre_dispatch_rejected" if process is None else "turn_rejected" if native["completed_turns"] else "turn_failed" if native["explicit_failure"] else "outcome_unknown",
                                error_type=type(error).__name__, thread_id=native["thread_id"],
                                usage=native["usage"], native_completed_turns=native["completed_turns"],
                                wall_seconds=time.monotonic()-started)
                    raise
        except BaseException as error:
            if not admitted:
                append(self.root / 'native_dispatch_waits.jsonl', {
                    'attempt_id': attempt, 'at': time.time(), 'state': 'aborted_before_dispatch',
                    'error_type': type(error).__name__, 'model_call_reserved': False})
            raise

    def summary(self):
        meter=summarize_role(self.root)
        invocations=[m for m in meter['models'] if m.get('cli_invocations')]
        return {**meter,"transport":"codex_cli","role":self.role,
            "model":self.model if len(invocations)<=1 else 'mixed',
            "effort":self.effort if len(invocations)<=1 else 'mixed',
            "action_responses_accepted":meter['structured_responses_accepted'] if self.role=='policy' else 0}
