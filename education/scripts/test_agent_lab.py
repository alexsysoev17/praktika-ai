"""Offline, mock-HTTP and integrity tests; never calls a real paid provider."""
import copy
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import socket
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

from agent_lab import (AgentLab, LabError, REFERENCE_SUITE, TERMINAL, digest, evaluate,
                       http_json, normalize_endpoint, provider_payload, provider_result,
                       resolve_endpoint, validate_suite)


class MockHandler(BaseHTTPRequestHandler):
    def log_message(self, *args): pass
    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        self.server.requests.append({'path': self.path, 'body': body, 'headers': dict(self.headers)})
        if self.path == '/redirect':
            self.send_response(302); self.send_header('Location', 'http://169.254.169.254/latest/meta-data/'); self.end_headers(); return
        if self.path == '/slow': time.sleep(1.5)
        if self.path == '/failure':
            self.send_response(500); self.end_headers(); self.wfile.write(b'sensitive upstream error is not exposed'); return
        if self.path == '/invalid': raw = b'{broken'
        else:
            input_ = body.get('input', '')
            if not input_ and body.get('messages'): input_ = body['messages'][-1]['content']
            output = {'status': 'ready', 'missing': []} if '2026-12-01' in input_ else {'status': 'needs_input', 'missing': ['deadline']}
            output_text = json.dumps(output)
            if self.path == '/v1/chat/completions': data = {'model': body['model'], 'choices': [{'message': {'content': output_text}, 'finish_reason': 'stop'}], 'usage': {'prompt_tokens': 50, 'completion_tokens': 20}}
            elif self.path == '/api/chat': data = {'model': body['model'], 'message': {'content': output_text}, 'done': True, 'done_reason': 'stop', 'prompt_eval_count': 50, 'eval_count': 20}
            else: data = {'model': body['model'], 'output': output, 'usage': {'inputTokens': 50, 'outputTokens': 20}}
            raw = json.dumps(data).encode()
        self.send_response(200); self.send_header('Content-Type', 'application/json'); self.send_header('Content-Length', str(len(raw))); self.end_headers()
        try: self.wfile.write(raw)
        except (BrokenPipeError, ConnectionResetError): pass


class LabTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = ThreadingHTTPServer(('127.0.0.1', 0), MockHandler)
        cls.server.daemon_threads = True; cls.server.requests = []
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True); cls.thread.start()
        cls.base = f'http://127.0.0.1:{cls.server.server_port}'

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown(); cls.server.server_close(); cls.thread.join()

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.root = Path(self.tmp.name); self.lab = AgentLab(self.root)

    def tearDown(self):
        self.lab.close(); self.tmp.cleanup()

    def request(self, route, data, status=201):
        code, value = self.lab.post('/api/lab/' + route, data)
        self.assertEqual(code, status, value); return value

    def connection(self, kind='custom_http', path='/agent', **extras):
        return self.request('connections', {'name': 'Mock model', 'kind': kind, 'endpoint': self.base + path, 'model': 'mock-same-model', **extras})

    def agent(self, conn, **extras):
        return self.request('agents', {'name': 'Adapted test', 'connectionId': conn['id'], 'instructions': 'SKILL_UNIQUE_MARKER', **extras})

    def wait(self, id_):
        for _ in range(400):
            code, run = self.lab.get('/api/lab/runs/' + id_)
            self.assertEqual(code, 200, run)
            if run['status'] in TERMINAL: return run
            time.sleep(.015)
        self.fail('Run did not finish')

    def run_agent(self, agent, **extras):
        run = self.request('runs', {'agentId': agent['id'], 'suiteId': 'mechanics-smoke', **extras}, 202)
        return self.wait(run['id'])

    def test_fixture_is_explicit_not_llm_and_no_uplift(self):
        run = self.run_agent({'id': 'fixture-agent'})
        self.assertEqual(run['status'], 'completed'); self.assertEqual(len(run['results']), 4)
        self.assertTrue(run['summary']['fixture']); self.assertIsNone(run['summary']['passRateDelta'])
        self.assertIsNone(run['summary']['humanLevel']); self.assertIsNone(run['summary']['cost'])
        self.assertFalse(run['summary']['gate']['automaticPassed'])
        self.assertTrue(all(run['integrity'].values()))

    def test_custom_http_ab_same_model_and_no_answer_leak(self):
        conn = self.connection(pricing={'inputPerMillion': 1, 'outputPerMillion': 2})
        agent = self.agent(conn, skills=[{'name': 'Contract', 'version': '2', 'content': 'SKILL_BODY'}], context='CONTEXT_MARKER')
        run = self.run_agent(agent)
        self.assertEqual(run['status'], 'completed', run)
        self.assertEqual(run['summary']['arms']['baseline']['passed'], 2)
        self.assertEqual(run['summary']['arms']['adapted']['passed'], 2)
        self.assertEqual(run['calls'], 4); self.assertAlmostEqual(run['summary']['cost'], .00036)
        for row in run['results']:
            trace = row['stages'][0]
            self.assertEqual(trace['providerModel'], 'mock-same-model')
            self.assertNotIn('checks', trace['input']); self.assertNotIn('expected', trace['input'])
            self.assertEqual('SKILL_BODY' in trace['instructions'], row['mode'] == 'adapted')
            self.assertEqual('CONTEXT_MARKER' in trace['instructions'], row['mode'] == 'adapted')
        self.assertFalse(run['summary']['gate']['automaticPassed'])
        self.assertTrue(run['summary']['mechanicsOnly']); self.assertIsNone(run['summary']['passRateDelta'])

    def test_compatible_adapter_actual_http_and_usage(self):
        run = self.run_agent(self.agent(self.connection('openai_compatible', '/v1/chat/completions')), mode='adapted')
        self.assertEqual(run['summary']['arms']['adapted']['passed'], 2)
        self.assertEqual(run['results'][0]['inputTokens'], 50)
        body = self.server.requests[-1]['body']
        self.assertFalse(body['stream']); self.assertIn('max_tokens', body); self.assertNotIn('tools', body)

    def test_ollama_adapter_actual_http_and_usage(self):
        run = self.run_agent(self.agent(self.connection('ollama', '/api/chat')), mode='baseline')
        self.assertEqual(run['summary']['arms']['baseline']['passed'], 2)
        self.assertEqual(run['results'][0]['outputTokens'], 20)
        self.assertFalse(self.server.requests[-1]['body']['stream'])
        self.assertIn('num_predict', self.server.requests[-1]['body']['options'])

    def test_openai_serialization_and_response_parser(self):
        c = {'kind': 'openai', 'model': 'user-selected-model', 'maxOutputTokens': 1200}
        payload = provider_payload(c, 'system', 'input')
        self.assertEqual(payload, {'model': 'user-selected-model', 'instructions': 'system', 'input': 'input', 'max_output_tokens': 1200, 'store': False, 'stream': False})
        parsed = provider_result(c, {'status': 'completed', 'id': 'resp_example', 'model': 'user-selected-model', 'output': [{'type': 'reasoning', 'summary': []}, {'type': 'message', 'content': [{'type': 'output_text', 'text': 'one'}, {'type': 'output_text', 'text': 'two'}]}], 'usage': {'input_tokens': 9, 'output_tokens': 4}})
        self.assertEqual(parsed['output'], 'one\ntwo'); self.assertEqual(parsed['inputTokens'], 9)
        self.assertTrue(parsed['complete'])
        parsed = provider_result(c, {'status': 'incomplete', 'output': [{'type': 'message', 'content': [{'type': 'output_text', 'text': 'partial'}]}]})
        self.assertFalse(parsed['complete'])

    def test_keys_never_stored_or_exposed(self):
        self.request('connections', {'kind': 'openai', 'name': 'x', 'model': 'x', 'apiKey': 'do-not-store'}, 400)
        with patch.dict('os.environ', {'UNIT_API_KEY': 'secret-never-in-state'}):
            conn = self.connection(apiKeyEnv='UNIT_API_KEY')
            run = self.run_agent(self.agent(conn), mode='adapted')
            state = self.lab.get('/api/lab/state')[1]
            self.assertNotIn('secret-never-in-state', json.dumps(state)); self.assertNotIn('secret-never-in-state', json.dumps(run))
            self.assertEqual(self.server.requests[-1]['headers']['Authorization'], 'Bearer secret-never-in-state')
        self.assertNotIn(b'secret-never-in-state', (self.root/'data/agent-lab/lab.sqlite3').read_bytes())

    def test_missing_credentials_stop_before_network(self):
        conn = self.connection(apiKeyEnv='ABSENT_TEST_API_KEY'); agent = self.agent(conn)
        before = len(self.server.requests)
        self.request('runs', {'agentId': agent['id'], 'suiteId': 'mechanics-smoke'}, 409)
        self.assertEqual(len(self.server.requests), before)

    def test_pipeline_budget_stops_without_silent_extra_calls(self):
        agent = self.agent(self.connection(), pipeline=[{'name': 'draft', 'instructions': 'first'}, {'name': 'verify', 'instructions': 'second'}])
        run = self.run_agent(agent, mode='adapted', budget={'maxCalls': 1, 'maxSeconds': 30})
        self.assertEqual(run['status'], 'budget_exhausted', run)
        self.assertEqual(run['calls'], 1); self.assertEqual(run['snapshot']['plannedCalls'], 4)
        self.assertEqual(run['summary']['arms']['adapted']['planned'], 2)
        self.assertFalse(run['summary']['gate']['automaticPassed'])

    def test_unknown_price_rejects_money_budget(self):
        agent = self.agent(self.connection())
        self.request('runs', {'agentId': agent['id'], 'suiteId': 'mechanics-smoke', 'budget': {'maxCost': 1}}, 409)

    def test_small_money_budget_prevents_first_call(self):
        agent = self.agent(self.connection(pricing={'inputPerMillion': 10, 'outputPerMillion': 30}))
        before = len(self.server.requests)
        run = self.run_agent(agent, budget={'maxCost': .000001})
        self.assertEqual(run['status'], 'budget_exhausted', run); self.assertEqual(run['calls'], 0)
        self.assertEqual(len(self.server.requests), before)

    def test_cancel_inflight_preserves_incurred_metrics(self):
        agent = self.agent(self.connection(path='/slow'))
        first = self.request('runs', {'agentId': agent['id'], 'suiteId': 'mechanics-smoke'}, 202)
        for _ in range(100):
            if self.lab.get('/api/lab/runs/' + first['id'])[1]['status'] == 'running': break
            time.sleep(.01)
        time.sleep(.1)
        cancel = self.request('runs/' + first['id'] + '/cancel', {}, 200)
        self.assertTrue(cancel['cancelRequested'])
        run = self.wait(first['id'])
        self.assertEqual(run['status'], 'cancelled'); self.assertLessEqual(run['calls'], 1)
        self.assertFalse(run['summary']['gate']['automaticPassed']); self.assertTrue(run['integrity']['resultsMatch'])

    def test_timeout_does_not_pass_or_claim_zero_cost(self):
        agent = self.agent(self.connection(path='/slow', timeoutSeconds=1))
        run = self.run_agent(agent, mode='adapted', budget={'maxCalls': 1, 'maxSeconds': 2})
        self.assertEqual(run['results'][0]['status'], 'error')
        self.assertIsNone(run['summary']['cost']); self.assertGreaterEqual(run['unknownCostCalls'], 1)
        self.assertEqual(run['summary']['arms']['adapted']['passRate'], 0)

    def test_redirect_never_followed_and_error_body_hidden(self):
        for path, code in [('/redirect', 'redirect_blocked'), ('/failure', 'provider_http_500'), ('/invalid', 'invalid_provider_json')]:
            with self.assertRaises(LabError) as raised: http_json(self.base + path, {'model': 'x'}, {}, 3)
            self.assertEqual(raised.exception.code, code)
            self.assertNotIn('sensitive', str(raised.exception))

    def test_endpoint_policy_and_dns_rebinding(self):
        for url in ['http://169.254.169.254/agent', 'http://example.com/agent', 'https://user:secret@example.com/agent', 'https://example.com/agent?token=x', 'http://127.0.0.1:8001/api/lab/runs']:
            with self.assertRaises(LabError): normalize_endpoint('custom_http', url)
        answer = [(socket.AF_INET, socket.SOCK_STREAM, 6, '', ('127.0.0.1', 443))]
        with patch('agent_lab.socket.getaddrinfo', return_value=answer):
            with self.assertRaises(LabError): resolve_endpoint('https://public.example/agent')
        answer = [(socket.AF_INET, socket.SOCK_STREAM, 6, '', ('10.1.1.1', 443))]
        with patch('agent_lab.socket.getaddrinfo', return_value=answer), patch.dict('os.environ', {'AGENT_LAB_ALLOWED_HOSTS': ''}):
            with self.assertRaises(LabError): resolve_endpoint('https://service.example/agent')
        with patch('agent_lab.socket.getaddrinfo', return_value=answer), patch.dict('os.environ', {'AGENT_LAB_ALLOWED_HOSTS': 'service.example'}):
            self.assertEqual(resolve_endpoint('https://service.example/agent')[0].hostname, 'service.example')

    def test_immutable_versions_and_snapshots(self):
        conn = self.connection(); first = self.agent(conn); second = self.agent(conn, parentId=first['id'], instructions='Different')
        self.assertEqual(second['version'], 2); self.assertNotEqual(first['id'], second['id'])
        run = self.run_agent(first, mode='adapted')
        self.assertEqual(run['snapshot']['agent']['instructions'], 'SKILL_UNIQUE_MARKER')
        self.assertEqual(run['snapshotHash'], digest(run['snapshot']))
        self.assertNotEqual(first['hash'], second['hash'])

    def test_import_metrics_unknown_and_duplicate_rejected(self):
        payload = {'name': 'Custom GPT exported', 'suiteId': 'mechanics-smoke', 'provenance': 'Manual export from own GPT; unverified model/settings', 'results': [{'caseId': 'ready', 'mode': 'adapted', 'output': {'status': 'ready', 'missing': []}}]}
        imported = self.request('imports', payload)
        self.assertIsNone(imported['cost']); self.assertFalse(imported['gate']['automaticPassed']); self.assertIsNone(imported['passRateDelta'])
        run = self.lab.get('/api/lab/runs/' + imported['id'])[1]
        self.assertEqual(run['results'][0]['metricsSource'], 'self_reported_unverified')
        payload['results'] *= 2; self.request('imports', payload, 400)

    def test_human_gate_required_and_decision_not_deployment(self):
        suite = copy.deepcopy(REFERENCE_SUITE); suite['id'] = 'human-gate-test'; suite['fixtureOnly'] = False; suite['humanRubric'] = ['Check actual output']
        self.request('suites', suite)
        agent = self.agent(self.connection()); run = self.run_agent(agent, mode='adapted', suiteId=suite['id'])
        decision = {'decision': 'approve_pilot', 'reviewer': 'Expert', 'reason': 'Reviewed actual outputs'}
        self.request('runs/' + run['id'] + '/decisions', decision, 409)
        for row in run['results']:
            self.request('runs/' + run['id'] + '/reviews', {'resultId': row['id'], 'reviewer': 'Expert', 'verdict': 'accepted', 'notes': 'Verified task contract and source content.'})
        saved = self.request('runs/' + run['id'] + '/decisions', decision)
        self.assertFalse(saved['deploymentChanged']); self.assertFalse(saved['identityVerified'])
        after = self.lab.get('/api/lab/runs/' + run['id'])[1]
        self.assertEqual(after['resultHash'], run['resultHash']); self.assertEqual(len(after['decisions']), 1)
        self.assertEqual(after['summary']['arms']['adapted']['humanQuality'], 'accepted_by_local_review')

    def test_evaluator_negative_cases_and_numeric_boolean(self):
        suite = validate_suite(REFERENCE_SUITE); case = suite['cases'][0]
        for output in ('not json', '{}', '{"status":"ready","missing":true}', '{"status":"invented","missing":[]}'):
            self.assertEqual(evaluate(output, case, suite)['status'], 'failed')
        c = {'checks': [{'type': 'numeric', 'path': 'amount', 'expected': 4, 'tolerance': .1}]}
        self.assertEqual(evaluate('{"amount":4.05}', c, {})['status'], 'passed')
        self.assertEqual(evaluate('{"amount":true}', c, {})['status'], 'failed')
        self.assertEqual(evaluate('{"amount":NaN}', c, {})['status'], 'failed')
        self.assertEqual(evaluate('whatever', {'checks': []}, {})['status'], 'unscored')

    def test_unknown_schema_feature_is_not_silently_ignored(self):
        suite = copy.deepcopy(REFERENCE_SUITE); suite['outputSchema']['oneOf'] = [{}]
        with self.assertRaises(LabError): validate_suite(suite)

    def test_restart_marks_unfinished_interrupted(self):
        run = self.run_agent({'id': 'fixture-agent'})
        self.lab._put('runs', {k: v for k, v in {**run, 'status': 'running'}.items() if k not in ('summary', 'integrity', 'events')})
        self.lab.close(); self.lab = AgentLab(self.root)
        read = self.lab.get('/api/lab/runs/' + run['id'])[1]
        self.assertEqual(read['status'], 'interrupted'); self.assertTrue(read['integrity']['resultsMatch'])

    def test_second_process_cannot_interrupt_live_database(self):
        with self.assertRaises(RuntimeError): AgentLab(self.root)
        self.assertEqual(self.lab.get('/api/lab/state')[0], 200)

    def test_critical_failure_blocks_even_zero_threshold(self):
        suite = copy.deepcopy(REFERENCE_SUITE); suite['id'] = 'critical-safety-test'
        suite['fixtureOnly'] = False
        suite['cases'][0]['critical'] = True
        suite['cases'][0]['checks'] = [{'type': 'exact', 'path': 'status', 'expected': 'needs_input'}]
        self.request('suites', suite)
        agent = self.agent(self.connection())
        started = self.request('runs', {'agentId': agent['id'], 'suiteId': suite['id'], 'mode': 'adapted', 'threshold': 0}, 202)
        run = self.wait(started['id'])
        self.assertEqual(run['status'], 'completed'); self.assertFalse(run['summary']['gate']['automaticPassed'])
        self.assertEqual(run['summary']['gate']['criticalFailed'], [{'caseId': 'ready', 'repetition': 1}])
        for row in run['results']:
            self.request('runs/' + run['id'] + '/reviews', {'resultId': row['id'], 'reviewer': 'Expert', 'verdict': 'accepted', 'notes': 'Cannot override critical automatic fail'})
        self.request('runs/' + run['id'] + '/decisions', {'decision': 'approve_pilot', 'reviewer': 'Expert', 'reason': 'Cannot bypass gate'}, 409)


class BundledSuiteTests(unittest.TestCase):
    def test_neutral_example_contract_and_broken_outputs(self):
        path = Path(__file__).resolve().parents[1]/'content/agent-benchmarks.json'
        suites = json.loads(path.read_text())['suites']
        self.assertEqual(len(suites), 1)
        for raw in suites:
            suite = validate_suite(raw)
            for case in suite['cases']:
                with self.subTest(suite=suite['id'], case=case['id']):
                    self.assertEqual(evaluate(case['expected'], case, suite)['status'], 'passed')
                    self.assertEqual(evaluate('{}', case, suite)['status'], 'failed')
                    self.assertEqual(evaluate('unstructured response', case, suite)['status'], 'failed')
                    if suite.get('humanRubric'): self.assertEqual(evaluate(case['expected'], case, suite)['humanQuality'], 'pending')


if __name__ == '__main__': unittest.main(verbosity=2)
