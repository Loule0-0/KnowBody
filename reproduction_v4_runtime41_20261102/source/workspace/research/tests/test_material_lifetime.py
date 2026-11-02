"""Knowledge validity follows its dependencies; original versions stay auditable."""
import unittest
import test_manager as fixtures
from kyb import library

class KnowledgeLifetimeTests(unittest.TestCase):
    setUp=fixtures.ManagerTests.setUp
    tearDown=fixtures.ManagerTests.tearDown
    observation=fixtures.ManagerTests.observation
    def test_grasp_change_invalidates_mount_but_not_intrinsic_geometry(self):
        library.initialize(self.store)
        results=[]
        with self.store.db:
            for family,keys in [('geometry',['body']),('tool_mount',['body','attachment:left'])]:
                op={'op':'add','domain':'body','family':family,'text':'Known source-specific relation','question':'Which source contract applies?','use':'Apply this source contract.',
                    'conditions':{},'strength':'configuration','depends_on':keys,
                    'refs':[{'type':'evidence','id':self.prior['id']}]}
                results.append(library.publish_knowledge(self.store,op,allowed={self.prior['id']},visible_cases={},
                    visible_knowledge={},dependencies=self.deps,frontier=self.session.source_frontier()))
        changed={**self.deps,'attachment:left':'new_grasp'}
        cards=library.read_knowledge(self.store,'',context={},dependencies=changed,ids=[r['id'] for r in results])
        by_family={c['family']:c for c in cards}
        self.assertTrue(by_family['geometry']['default_eligible'])
        self.assertFalse(by_family['tool_mount']['default_eligible'])
    def test_revision_requires_read_version_and_preserves_old_record(self):
        library.initialize(self.store)
        op={'op':'add','domain':'body','family':'geometry','text':'Known relation','question':'Which source contract applies?','use':'Apply this source contract.',
            'conditions':{},'strength':'configuration','refs':[{'type':'evidence','id':self.prior['id']}]}
        with self.store.db:
            first=library.publish_knowledge(self.store,op,allowed={self.prior['id']},visible_cases={},
                visible_knowledge={},dependencies=self.deps,frontier=self.session.source_frontier())
        retire={'op':'retire','id':first['id'],'reason':'Source contract superseded'}
        with self.assertRaises(ValueError):library.publish_knowledge(self.store,retire,allowed=set(),visible_cases={},
            visible_knowledge={},dependencies=self.deps,frontier=self.session.source_frontier())
        with self.store.db:
            library.publish_knowledge(self.store,retire,allowed=set(),visible_cases={},
                visible_knowledge={first['id']:first['revision']},dependencies=self.deps,frontier=self.session.source_frontier())
        self.assertEqual(self.store.db.execute('SELECT status FROM reusable_knowledge WHERE revision=?',(first['revision'],)).fetchone()[0],'active')
        card=library.read_knowledge(self.store,'',context={},dependencies=self.deps,ids=[first['id']])[0]
        self.assertFalse(card['default_eligible']);self.assertEqual(card['status'],'retired')
