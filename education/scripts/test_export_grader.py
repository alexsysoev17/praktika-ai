import copy
import unittest
from agent_lab import REFERENCE_SUITE, evaluate, schema_errors
from grade_agent_export import grade

class ExportChecks(unittest.TestCase):
    def rows(self):
        return [{'caseId':c['id'],'mode':'adapted','output':c['expected']} for c in REFERENCE_SUITE['cases']]
    def test_good_answers_still_external_unverified(self):
        r=grade(REFERENCE_SUITE,self.rows());self.assertTrue(r['allContractChecksPassed']);self.assertIsNone(r['humanLevel']);self.assertEqual(r['source'],'external_artifact_unverified')
    def test_missing_answer_remains_in_denominator(self):
        r=grade(REFERENCE_SUITE,self.rows()[:1]);self.assertEqual(r['passRate'],.5);self.assertFalse(r['allContractChecksPassed']);self.assertEqual(len(r['missing']),1)
    def test_wrong_answer_fails(self):
        rows=self.rows();rows[1]=copy.deepcopy(rows[1]);rows[1]['output']['missing']=[]
        self.assertFalse(grade(REFERENCE_SUITE,rows)['allContractChecksPassed'])
    def test_duplicates_do_not_inflate_score(self):
        with self.assertRaises(ValueError):grade(REFERENCE_SUITE,self.rows()*2)
    def test_nested_bool_does_not_equal_number(self):
        case={'checks':[{'type':'exact','expected':{'allowed':[1]}}]}
        self.assertEqual(evaluate({'allowed':[True]},case,{})['status'],'failed')
        self.assertEqual(evaluate({'allowed':[1]},case,{})['status'],'passed')
        self.assertTrue(schema_errors({'allowed':[True]},{'enum':[{'allowed':[1]}]}))
    def test_unknown_case_rejected(self):
        with self.assertRaises(ValueError):grade(REFERENCE_SUITE,[{'caseId':'invented','output':{}}])
    def test_repeated_trials_must_all_be_present(self):
        r=grade(REFERENCE_SUITE,self.rows(),repetitions=2);self.assertEqual(r['planned'],4);self.assertFalse(r['allContractChecksPassed'])
if __name__=='__main__':unittest.main()
