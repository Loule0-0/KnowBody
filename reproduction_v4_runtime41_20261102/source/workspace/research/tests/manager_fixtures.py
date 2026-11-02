"""Offline manager responses. Never construct a real CLI or call a model."""
import copy
import json


def response(name, arguments):
    if name=='publish_materials':
        operations={k:arguments[k] for k in ('memory_patch','knowledge_ops','dispositions','processed') if k in arguments}
        if operations:arguments={k:v for k,v in arguments.items() if k not in operations}|{'groups':[operations]}
    return {'choices':[{'message':{'role':'assistant','content':None,'tool_calls':[
        {'id':'fake_'+name,'type':'function','function':{'name':name,'arguments':json.dumps(arguments)}}]}}]}


class PreparationClient:
    def __init__(self):self.inputs=[];self.budget=None
    def start_transaction(self):pass
    def complete(self,messages,tools):
        self.inputs.append(copy.deepcopy(messages))
        initial=json.loads(messages[1]['content'])
        return response('publish_materials',{'expected_version':initial['expected_version'],'answer':'Prepared offline fixture.',
            'default_knowledge':[]})


class ScriptedManager:
    def __init__(self,turns):self.turns=list(turns);self.inputs=[];self.budget=None
    def start_transaction(self):pass
    def complete(self,messages,tools):
        self.inputs.append(copy.deepcopy(messages))
        value=self.turns.pop(0)
        return value(messages) if callable(value) else response(*value)
