"""Deterministic routing fixtures only; genuine inference is verified separately."""
import copy
import unittest
from .research_decision_engine import compute_research_verdict as decide, ResearchVerdict
from .research_inference import validate_research_response, ABSTENTION, contract_descriptor
from .test_research_integration import StubEngine,png


def model(label,score=.8):
    others=[x for x in ('Normal','Benign','Malignant') if x!=label]
    return dict(status='ok',prediction=label,score=score,probabilities={label:score,**{x:(1-score)/2 for x in others}})


class DecisionTests(unittest.TestCase):
    def setUp(self):self.models={'UCLM':model('Normal',.7),'BUS-BRA':model('Malignant',.99),'BUSI':model('Malignant',.999),'BUS-BRA BI-RADS':dict(status='ok',prediction='5',score=1,probabilities={'5':1})}
    def test_uclm_three_classes(self):
        for label in ('Normal','Benign','Malignant'):
            self.models['UCLM']=model(label);r=decide(self.models,'UCLM');self.assertEqual(r.prediction,label);self.assertEqual(r.primary_model_id,'uclm');self.assertEqual(r.training_domain_relation,'native_research')
    def test_bra_binary_routing(self):
        for label in ('Benign','Malignant'):
            self.models['BUS-BRA']=model(label);r=decide(self.models,'BUS-BRA');self.assertEqual(r.prediction,label);self.assertEqual(r.primary_model_id,'bra')
    def test_bra_cannot_normal(self):
        self.models['BUS-BRA']=model('Normal');r=decide(self.models,'BUS-BRA');self.assertEqual(r.status,'unavailable');self.assertIsNone(r.prediction);self.assertNotIn('Normal',r.supported_primary_classes)
    def test_bra_warning(self):self.assertTrue(any('cannot classify Normal' in t for t in decide(self.models,'BUS-BRA').applicability_limitations))
    def test_busi_historical_only(self):
        r=decide(self.models,'BUSI');self.assertEqual(r.primary_model_id,'uclm');self.assertEqual(r.prediction,'Normal');self.assertTrue(r.experimental_fallback);self.assertEqual(r.secondary_model_observations[0].role,'historical_comparison');self.assertNotIn('busi',r.agreement.disagreeing_model_ids)
    def test_unknown_fallback_warning(self):
        r=decide(self.models,'unknown');self.assertTrue(r.experimental_fallback);self.assertEqual(r.training_domain_relation,'unknown');self.assertIn('not a reliable negative',' '.join(r.applicability_limitations))
    def test_missing_primary(self):
        del self.models['UCLM'];r=decide(self.models,'UCLM');self.assertEqual(r.status,'unavailable');self.assertIsNone(r.primary_model_score)
    def test_primary_missing_status(self):
        self.models['UCLM']={'status':'missing'};self.assertEqual(decide(self.models,'unknown').error_code,'primary_model_missing')
    def test_primary_error_no_secondary_substitution(self):
        self.models['UCLM']={'status':'error'};r=decide(self.models,'UCLM');self.assertIsNone(r.prediction);self.assertEqual(r.primary_model_id,'uclm')
    def test_primary_not_run(self):
        self.models['BUS-BRA']={'status':'not_run'};self.assertEqual(decide(self.models,'BUS-BRA').status,'unavailable')
    def test_complete_failure(self):
        for domain in ('unknown','UCLM','BUSI','BUS-BRA'):
            r=decide({},domain);self.assertEqual(r.status,'unavailable');self.assertIsNone(r.prediction)
    def test_secondary_failure_keeps_primary(self):
        self.models['BUS-BRA']={'status':'error'};r=decide(self.models,'UCLM');self.assertEqual(r.prediction,'Normal');self.assertEqual(r.agreement.indicator,'unavailable')
    def test_disagreement_does_not_vote(self):
        self.models['UCLM']=model('Benign',.6);r=decide(self.models,'UCLM');self.assertEqual(r.agreement.indicator,'disagreement');self.assertEqual(r.prediction,'Benign');self.assertEqual(r.primary_model_score,.6)
    def test_agreement_never_increases_score(self):
        self.models['UCLM']=model('Malignant',.6);r=decide(self.models,'UCLM');self.assertEqual(r.agreement.indicator,'agreement');self.assertEqual(r.primary_model_score,.6)
    def test_normal_vs_binary_not_comparable(self):self.assertEqual(decide(self.models,'UCLM').agreement.indicator,'not_comparable')
    def test_high_score_remains_uncalibrated(self):
        self.models['UCLM']=model('Normal',.999999);r=decide(self.models,'unknown');self.assertEqual(r.score_semantics,'uncalibrated_softmax');self.assertIn('not a reliable negative',' '.join(r.applicability_limitations))
    def test_no_borrowed_higher_score(self):self.assertEqual(decide(self.models,'UCLM').primary_model_score,.7)
    def test_inconsistent_primary_score_rejected(self):
        self.models['UCLM']['score']=.123
        with self.assertRaises(ValueError):decide(self.models,'UCLM')
    def test_birads_never_in_observations(self):self.assertNotIn('birads',[r.model_id for r in decide(self.models,'UCLM').secondary_model_observations])
    def test_no_weighted_vote_fields(self):
        r=decide(self.models,'UCLM').model_dump();self.assertNotIn('weights',r);self.assertNotIn('confidence',r);self.assertEqual(r['primary_model_score'],.7)
    def test_deterministic_and_inputs_immutable(self):
        before=copy.deepcopy(self.models);a=decide(self.models,'UCLM');b=decide(self.models,'UCLM');self.assertEqual(a,b);self.assertEqual(self.models,before)
    def test_unknown_domain_rejected(self):
        with self.assertRaises(ValueError):decide(self.models,'native-verified')
    def test_unknown_version_rejected(self):
        with self.assertRaises(ValueError):validate_research_response({'contract_version':'9.0.0'})
    def test_old_contract_10_no_verdict(self):
        r=StubEngine().analyze(png(),attribution=False);validate_research_response(r.model_dump());self.assertNotIn('research_verdict',r.model_dump())
    def test_old_contract_11_no_verdict(self):
        r=StubEngine().analyze(png(),attribution=False,contract_version='1.1.0');validate_research_response(r.model_dump());self.assertNotIn('research_verdict',r.model_dump())
    def test_new_contract_separate_consensus(self):
        r=StubEngine().analyze(png(),attribution=False,contract_version='1.2.0');self.assertEqual(r.consensus.prediction,ABSTENTION);self.assertIsNone(r.consensus.score);self.assertEqual(r.research_verdict.primary_model_id,'uclm');self.assertEqual(len(r.models),4);validate_research_response(r.model_dump())
    def test_tampered_verdict_rejected(self):
        r=StubEngine().analyze(png(),attribution=False,contract_version='1.2.0').model_dump();r['research_verdict']['primary_model_score']=.123
        with self.assertRaises(ValueError):validate_research_response(r)
    def test_all_versions_discoverable_default_10(self):
        self.assertEqual(contract_descriptor()['contract_version'],'1.0.0')
        for v in ['1.0.0','1.1.0','1.2.0']:self.assertEqual(contract_descriptor(v)['contract_version'],v)


if __name__=='__main__':unittest.main(verbosity=2)
