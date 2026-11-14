import base64,copy,io,json,shutil,struct,tempfile,time,unittest
from pathlib import Path
from unittest.mock import patch
from PIL import Image
import pipeline_lab as p

SOURCE=Path(__file__).resolve().parents[1]
class FakeAgents:
    def _all(self,kind):return [self._get(kind,'test-agent')]
    def _get(self,kind,id_):
        if kind=='agents':return {'id':id_,'name':id_,'connectionId':'local-test','instructions':'Role test','skills':[],'context':''}
        return {'id':'local-test','kind':'custom_http','model':'test-model','endpoint':'http://127.0.0.1:9999/runner','apiKeyEnv':'','maxOutputTokens':128,'timeoutSeconds':10,'pricing':{'inputPerMillion':1,'outputPerMillion':1,'currency':'USD'}}

def png(color=(120,20,220,255),transparent=False):
    im=Image.new('RGBA',(512,512),(0,0,0,0))
    if not transparent:im.paste(color,(100,100,410,410))
    b=io.BytesIO();im.save(b,format='PNG');return b.getvalue()

def rewrite_glb(raw,change):
    n=struct.unpack_from('<I',raw,12)[0];doc=json.loads(raw[20:20+n]);change(doc);js=json.dumps(doc).encode();js+=b' '*((-len(js))%4);rest=raw[20+n:]
    return struct.pack('<4sII',b'glTF',2,20+len(js)+len(rest))+struct.pack('<II',len(js),0x4E4F534A)+js+rest

class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
        (self.root/'content').mkdir();shutil.copyfile(SOURCE/'content/pipeline-artifacts.json',self.root/'content/pipeline-artifacts.json')
        shutil.copytree(SOURCE/'assets/pipeline',self.root/'assets/pipeline')
        self.lab=p.PipelineLab(self.root,FakeAgents())
    def tearDown(self):self.lab.close();self.tmp.cleanup()
    def call(self,path,payload):
        code,d=self.lab.post('/api/pipeline/'+path,payload)
        self.assertLess(code,300,d);return d
    def blueprint(self,live=False,**limits):
        s=p.template('mesh_3d')['spec'];s['limits'].update(limits)
        if live:
            for n in s['nodes']:
                if n['role'] in p.MODEL_ROLES:n['agentId']='agent-'+n['role']
        return self.call('blueprints',{'name':'Test blueprint','spec':s})
    def wait(self,id_):
        for i in range(200):
            r=self.lab.get_item('runs',id_)
            if r['status']!='running' and id_ not in self.lab.events:return r
            time.sleep(.01)
        self.fail('Run did not stop')
    def start(self,b=None,mode='walkthrough',**kw):
        b=b or self.blueprint();return self.wait(self.call('runs',{'blueprintId':b['id'],'mode':mode,'brief':'Six objects in one visual style',**kw})['id'])
    def review(self,r,decision='approve'):return self.call('runs/'+r['id']+'/review',{'decision':decision,'reviewer':'Test reviewer','notes':'Synthetic test only'})
    def test_default_templates_and_bad_contract(self):
        state=self.lab.get('/api/pipeline/state')[1];self.assertEqual(state['agents'][0]['id'],'test-agent');self.assertNotIn('instructions',state['agents'][0])
        for kind in p.DEFAULT_CONTRACTS:self.assertTrue(p.validate_spec(p.template(kind)['spec'])['valid'])
        s=p.template('mesh_3d')['spec'];s['qualityContract']['maxTriangles']=True;self.assertFalse(p.validate_spec(s)['valid'])
        s=p.template('mesh_3d')['spec'];s['qualityContract']['execute']='shell';self.assertFalse(p.validate_spec(s)['valid'])
    def test_cycle_missing_input_and_human_gate(self):
        for mutate in [lambda s:s['nodes'][0]['dependsOn'].append('final_check'),lambda s:s['nodes'][2]['inputs'].append('not-produced'),lambda s:s.update(humanGate={'after':'planner','before':'batch_spec'})]:
            s=p.template('mesh_3d')['spec'];mutate(s);self.assertFalse(p.validate_spec(s)['valid'])
    def test_same_agent_cannot_review_itself(self):
        s=p.template('mesh_3d')['spec'];s['nodes'][2]['agentId']='x';s['nodes'][4]['agentId']='x';self.assertFalse(p.validate_spec(s)['valid'])
    def test_walkthrough_requires_two_human_gates(self):
        r=self.start();self.assertEqual(r['status'],'awaiting_review');self.assertEqual(r['phase'],'pilot');self.assertEqual(r['calls'],0)
        self.assertEqual(self.lab.post('/api/pipeline/runs/'+r['id']+'/resume',{})[0],409)
        after=self.review(r);self.assertEqual(after['phase'],'batch');self.assertEqual(after['status'],'awaiting_artifacts')
        self.call('runs/'+r['id']+'/resume',{});r=self.wait(r['id']);self.assertEqual(r['status'],'awaiting_review');self.assertEqual(r['phase'],'batch')
        self.assertEqual(self.review(r)['status'],'completed')
        self.assertTrue(all(x['passed'] for x in r['artifactChecks']['pilot']+r['artifactChecks']['batch']))
        self.assertTrue(self.lab.get('/api/pipeline/runs/'+r['id'])[1]['snapshotIntegrity'])
    def test_real_bad_palette_blocks_and_repair_is_bounded(self):
        r=self.start(artifactIds=['mesh-cloud','mesh-folder','mesh-bad-palette']);self.assertEqual(r['status'],'needs_revision')
        self.assertEqual(self.lab.post('/api/pipeline/runs/'+r['id']+'/review',{'decision':'approve','reviewer':'X','notes':'bad'})[0],409)
        self.call('runs/'+r['id']+'/artifacts',{'phase':'pilot','artifactIds':['mesh-cloud','mesh-folder','mesh-lock']});r=self.wait(r['id']);self.assertEqual(r['status'],'awaiting_review')
        self.review(r,'request_changes')
        code,d=self.lab.post('/api/pipeline/runs/'+r['id']+'/artifacts',{'phase':'pilot','artifactIds':['mesh-cloud','mesh-folder','mesh-chart']});self.assertEqual(code,409)
    def test_batch_cannot_repeat_pilot_bytes(self):
        r=self.start();self.review(r)
        code,d=self.lab.post('/api/pipeline/runs/'+r['id']+'/artifacts',{'phase':'batch','artifactIds':r['artifactIds']['pilot']});self.assertEqual(code,400)
    def test_png_bytes_dimensions_alpha_and_upload(self):
        data=png();self.assertTrue(p.inspect_bytes(data,'raster_3d',p.DEFAULT_CONTRACTS['raster_3d'])['passed'])
        self.assertFalse(p.inspect_bytes(png(transparent=True),'raster_3d',p.DEFAULT_CONTRACTS['raster_3d'])['passed'])
        a=self.call('artifacts',{'name':'input.png','type':'raster_3d','dataBase64':base64.b64encode(data).decode()})
        content=self.lab.get('/api/pipeline/artifacts/'+a['id']+'/content')[1];self.assertEqual(base64.b64decode(content['dataBase64']),data)
        bad=bytearray(data);bad[16:20]=struct.pack('>I',1000000);self.assertFalse(p.inspect_bytes(bytes(bad),'raster_3d',p.DEFAULT_CONTRACTS['raster_3d'])['passed'])
    def test_glb_checks_actual_bytes_and_transforms(self):
        raw=(self.root/'assets/pipeline/cloud.glb').read_bytes() if (self.root/'assets/pipeline/cloud.glb').exists() else self.lab.artifact('mesh-cloud')[1]
        valid=p.inspect_bytes(raw,'mesh_3d',p.DEFAULT_CONTRACTS['mesh_3d']);self.assertTrue(valid['passed'])
        changed=rewrite_glb(raw,lambda d:d['nodes'][0].update(scale=[100,100,100]));self.assertFalse(p.inspect_bytes(changed,'mesh_3d',p.DEFAULT_CONTRACTS['mesh_3d'])['passed'])
        bad=rewrite_glb(raw,lambda d:d['buffers'][0].update(uri='http://localhost/secret'));self.assertFalse(p.inspect_bytes(bad,'mesh_3d',p.DEFAULT_CONTRACTS['mesh_3d'])['passed'])
        bad=rewrite_glb(raw,lambda d:d['accessors'][0].update(count=200000));self.assertFalse(p.inspect_bytes(bad,'mesh_3d',p.DEFAULT_CONTRACTS['mesh_3d'])['passed'])
    def test_per_blueprint_quality_contract_frozen(self):
        s=p.template('mesh_3d')['spec'];s['qualityContract']['allowedBaseColors']=['#A3E635','#EF4444','#F1EDFF']
        b=self.call('blueprints',{'name':'Red palette','spec':s});r=self.start(b,artifactIds=['mesh-bad-palette','mesh-folder','mesh-lock'])
        self.assertEqual(r['snapshot']['contracts']['mesh_3d']['allowedBaseColors'],s['qualityContract']['allowedBaseColors']);self.assertEqual(r['status'],'needs_revision')
        check=r['artifactChecks']['pilot'][0];self.assertTrue(next(c['passed'] for c in check['checks'] if c['id']=='palette'))
    def test_artifact_change_after_checks_blocks_approval(self):
        r=self.start();a,raw=self.lab.artifact('mesh-cloud');(self.root/a['path']).write_bytes(raw+b'changed')
        self.assertEqual(self.lab.post('/api/pipeline/runs/'+r['id']+'/review',{'decision':'approve','reviewer':'X','notes':'accept'})[0],409)
    def test_live_roles_do_not_generate_fake_artifacts(self):
        response={'output':'A text plan, no image','complete':True,'inputTokens':20,'outputTokens':10,'reportedCost':None,'providerModel':'test'}
        with patch.object(p.al,'invoke',return_value=response) as invoke:
            r=self.start(self.blueprint(live=True),mode='live');self.assertEqual(r['status'],'awaiting_artifacts');self.assertEqual(invoke.call_count,3);self.assertEqual(r['artifactIds']['pilot'],[])
            self.call('runs/'+r['id']+'/artifacts',{'phase':'pilot','artifactIds':['mesh-cloud','mesh-folder','mesh-lock']});r=self.wait(r['id']);self.assertEqual(r['status'],'awaiting_review');self.assertEqual(invoke.call_count,4)
            self.assertTrue(all(x['source']=='live_text_role' for x in r['nodeResults']));self.assertFalse(self.lab.summary(r)['generatedByLab'])
    def test_live_budget_stops_next_call(self):
        response={'output':'Plan','complete':True,'inputTokens':20,'outputTokens':10,'reportedCost':None}
        with patch.object(p.al,'invoke',return_value=response) as invoke:
            r=self.start(self.blueprint(live=True,maxCalls=1),mode='live');self.assertEqual(r['status'],'budget_exhausted');self.assertEqual(invoke.call_count,1)
    def test_cancel_waiting_does_not_resume(self):
        r=self.start();self.assertEqual(self.call('runs/'+r['id']+'/cancel',{})['status'],'cancelled')
        self.assertEqual(self.lab.post('/api/pipeline/runs/'+r['id']+'/resume',{})[0],409)
    def test_unknown_file_and_path_not_read(self):
        reg=self.lab.registry();reg['artifacts'].append({'id':'outside','type':'mesh_3d','path':'content/pipeline-artifacts.json','source':'authored_fixture'})
        (self.root/'content/pipeline-artifacts.json').write_text(json.dumps(reg))
        self.assertEqual(self.lab.get('/api/pipeline/artifacts/outside/content')[0],404)
    def test_revision_new_id_and_old_snapshot(self):
        b=self.blueprint();r=self.start(b);s=copy.deepcopy(b['spec']);s['pilotCount']=2
        b2=self.call('blueprints',{'name':'Version two','parentId':b['id'],'spec':s});self.assertEqual(b2['version'],2);self.assertNotEqual(b2['hash'],b['hash']);self.assertEqual(self.lab.get_item('runs',r['id'])['snapshot']['blueprint']['spec']['pilotCount'],3)
    def test_partial_contract_freezes_defaults_when_blueprint_saved(self):
        s=p.template('mesh_3d')['spec'];s['qualityContract']={'maxTriangles':19000}
        b=self.call('blueprints',{'name':'Frozen defaults','spec':s})
        reg=self.lab.registry();reg['contracts']['mesh_3d']['roughnessMin']=.99;(self.root/'content/pipeline-artifacts.json').write_text(json.dumps(reg))
        r=self.start(b);self.assertEqual(r['snapshot']['contracts']['mesh_3d']['roughnessMin'],.6);self.assertEqual(r['status'],'awaiting_review')
    def test_human_gate_persists_across_restart_and_running_is_interrupted(self):
        r=self.start();self.lab.close();self.lab=p.PipelineLab(self.root,FakeAgents())
        self.assertEqual(self.lab.get_item('runs',r['id'])['status'],'awaiting_review')
        r['status']='running';self.lab.put('runs',r);self.lab.close();self.lab=p.PipelineLab(self.root,FakeAgents())
        self.assertEqual(self.lab.get_item('runs',r['id'])['status'],'interrupted')
    def test_real_roughness_and_corrupted_png_cannot_be_approved(self):
        r=self.start(artifactIds=['mesh-cloud','mesh-folder','mesh-bad-roughness']);self.assertEqual(r['status'],'needs_revision')
        self.assertFalse(next(c['passed'] for c in r['artifactChecks']['pilot'][-1]['checks'] if c['id']=='roughness'))
        bad=bytearray(png());bad[29]^=255
        self.assertEqual(self.lab.post('/api/pipeline/artifacts',{'name':'broken.png','type':'raster_3d','dataBase64':base64.b64encode(bad).decode()})[0],400)
    def test_cost_reservation_blocks_before_paid_call(self):
        with patch.object(p.al,'invoke') as invoke:
            r=self.start(self.blueprint(live=True,maxCost=.000001),mode='live');self.assertEqual(r['status'],'budget_exhausted');self.assertEqual(invoke.call_count,0)
    def test_inflight_cancel_and_concurrent_attachment(self):
        response={'output':'Completed during cancellation','complete':True,'inputTokens':20,'outputTokens':10,'reportedCost':None}
        def invoke(c,instructions,prompt,timeout,cancel):cancel.wait(1);return response
        with patch.object(p.al,'invoke',side_effect=invoke):
            id_=self.call('runs',{'blueprintId':self.blueprint(live=True)['id'],'mode':'live','brief':'Bound cancellation'})['id']
            for _ in range(100):
                if self.lab.get_item('runs',id_)['calls']:break
                time.sleep(.01)
            self.assertEqual(self.lab.post('/api/pipeline/runs/'+id_+'/artifacts',{'phase':'pilot','artifactIds':['mesh-cloud','mesh-folder','mesh-lock']})[0],409)
            self.call('runs/'+id_+'/cancel',{});r=self.wait(id_);self.assertEqual(r['status'],'cancelled');self.assertEqual(r['calls'],1)
    def test_batch_attempts_use_batch_role_limit(self):
        s=p.template('mesh_3d')['spec']
        next(n for n in s['nodes'] if n['role']=='batch_spec')['maxAttempts']=1
        b=self.call('blueprints',{'name':'No batch repair','spec':s});r=self.start(b);self.review(r)
        self.call('runs/'+r['id']+'/resume',{});r=self.wait(r['id']);self.review(r,'request_changes')
        self.assertEqual(self.lab.post('/api/pipeline/runs/'+r['id']+'/artifacts',{'phase':'batch','artifactIds':['mesh-chart','mesh-stack','mesh-bad-roughness']})[0],409)
    def test_upload_preserves_unverified_provenance_as_data(self):
        marker=self.root/'provenance-must-not-execute'
        origin='External Tool v2; prompt/ref saved separately; $(touch '+str(marker)+'); <script>never_execute()</script>'
        a=self.call('artifacts',{'name':'Owned input','type':'raster_3d','dataBase64':base64.b64encode(png()).decode(),'metadata':{'origin':origin,'originalName':'my rendered icon.png','provenance':'verified-by-provider','path':'/tmp/overwrite'}})
        self.assertEqual(a['metadata']['origin'],origin);self.assertEqual(a['metadata']['originalName'],'my rendered icon.png');self.assertFalse(a['metadata']['identityVerified']);self.assertIn('unverified',a['metadata']['provenance']);self.assertNotIn('path',a['metadata']);self.assertFalse(marker.exists())
        self.assertEqual(self.lab.get_item('artifacts',a['id'])['metadata']['origin'],origin)
    def test_upload_metadata_rejects_paths_objects_and_unbounded_notes(self):
        before=list(self.lab.dir.glob('artifact_*'))
        for metadata in [{'origin':{}},{'origin':'x'*3001},{'originalName':['file.png']},{'originalName':'../file.png'},{'originalName':'C:\\files\\file.png'},{'originalName':'bad\x00name.png'},[]]:
            status,_=self.lab.post('/api/pipeline/artifacts',{'name':'Input','type':'raster_3d','dataBase64':base64.b64encode(png()).decode(),'metadata':metadata});self.assertEqual(status,400,metadata)
        self.assertEqual(list(self.lab.dir.glob('artifact_*')),before)
    def test_authored_rendered_png_fixtures(self):
        for name in ['cloud','folder','lock','chart','stack','target']:
            checked=p.inspect_bytes((self.root/'assets/pipeline'/f'{name}.png').read_bytes(),'raster_3d',p.DEFAULT_CONTRACTS['raster_3d']);self.assertTrue(checked['passed'],(name,checked))
    def test_attempt_history_retains_failed_bytes_and_retry_once(self):
        r=self.start(artifactIds=['mesh-cloud','mesh-folder','mesh-bad-palette']);first=copy.deepcopy(r['attemptHistory'][0]);self.assertEqual(first['attempt'],1)
        self.assertFalse(first['checks'][-1]['passed']);self.assertEqual(first['artifacts'][-1]['id'],'mesh-bad-palette')
        self.call('runs/'+r['id']+'/artifacts',{'phase':'pilot','artifactIds':['mesh-cloud','mesh-folder','mesh-lock']});r=self.wait(r['id'])
        self.assertEqual(len(r['attemptHistory']),2);self.assertEqual(r['attemptHistory'][0],first);self.assertEqual(r['attemptHistory'][1]['attempt'],2);self.assertTrue(all(c['passed'] for c in r['attemptHistory'][1]['checks']))
        self.assertNotEqual(first['artifacts'][-1]['sha256'],r['attemptHistory'][1]['artifacts'][-1]['sha256'])
        self.review(r);self.assertEqual(len(self.lab.get_item('runs',r['id'])['attemptHistory']),2)
        self.call('runs/'+r['id']+'/resume',{});r=self.wait(r['id']);self.assertEqual(len(r['attemptHistory']),3);self.assertEqual(r['attemptHistory'][-1]['phase'],'batch');self.assertEqual(r['attemptHistory'][-1]['attempt'],1)
    def test_legacy_history_recovers_only_current_known_evidence(self):
        r=self.start(artifactIds=['mesh-cloud','mesh-folder','mesh-bad-palette']);r.pop('attemptHistory');r.pop('attemptHistoryComplete');self.lab.put('runs',r)
        returned=self.lab.get('/api/pipeline/runs/'+r['id'])[1];self.assertEqual(returned['attemptHistory'],[]);self.assertFalse(returned['attemptHistoryComplete'])
        self.call('runs/'+r['id']+'/artifacts',{'phase':'pilot','artifactIds':['mesh-cloud','mesh-folder','mesh-lock']});r=self.wait(r['id'])
        self.assertEqual(len(r['attemptHistory']),2);self.assertFalse(r['attemptHistoryComplete']);self.assertEqual(r['attemptHistory'][0]['source'],'legacy_current_snapshot');self.assertIsNone(r['attemptHistory'][0]['attempt']);self.assertIsNone(r['attemptHistory'][0]['checkedAt']);self.assertFalse(r['attemptHistory'][0]['checks'][-1]['passed'])
if __name__=='__main__':unittest.main()
