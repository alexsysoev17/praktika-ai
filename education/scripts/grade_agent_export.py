"""Offline grading of externally produced answers; never calls a model or executes submissions."""
import argparse
import json
from pathlib import Path
from agent_lab import validate_suite, evaluate, digest, LabError

def read(path):
    p=Path(path)
    if p.stat().st_size>2_000_000:raise ValueError('JSON exceeds 2 MB')
    return json.loads(p.read_text(),parse_constant=lambda _: (_ for _ in ()).throw(ValueError('Nonfinite JSON')))

def grade(suite,results,mode='adapted',repetitions=1):
    suite=validate_suite(suite)
    if not isinstance(results,list):raise ValueError('Results must be an array')
    cases={c['id']:c for c in suite['cases']};seen=set();rows=[]
    for item in results:
        if not isinstance(item,dict):raise ValueError('Result must be an object')
        if item.get('mode','adapted')!=mode:continue
        cid=item.get('caseId');rep=item.get('repetition',1)
        if cid not in cases:raise ValueError('Unknown caseId')
        if type(rep) is not int or not 1<=rep<=repetitions:raise ValueError('Unexpected repetition')
        key=(cid,rep)
        if key in seen:raise ValueError('Duplicate result')
        seen.add(key)
        if 'output' not in item:raise ValueError('Missing output')
        evaluation=evaluate(item['output'],cases[cid],suite)
        rows.append({'caseId':cid,'repetition':rep,'evaluation':evaluation,'critical':bool(cases[cid].get('critical'))})
    expected=len(cases)*repetitions;passed=sum(r['evaluation']['status']=='passed' for r in rows)
    missing=[{'caseId':cid,'repetition':rep} for cid in cases for rep in range(1,repetitions+1) if (cid,rep) not in seen]
    return {'schemaVersion':'1','source':'external_artifact_unverified','suiteId':suite['id'],'suiteHash':suite['hash'],'resultHash':digest(results),'mode':mode,'planned':expected,'received':len(rows),'passed':passed,'passRate':passed/expected,'missing':missing,'allContractChecksPassed':passed==expected,'humanQuality':'pending','humanLevel':None,'results':rows,'limitations':['JSON contracts only; no proof of tool execution, visual quality, model identity or work benefit.','CI success does not grant a skill level or deploy an agent.']}

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--suite',required=True);parser.add_argument('--suite-id');parser.add_argument('--results',required=True);parser.add_argument('--output',required=True);parser.add_argument('--mode',choices=['baseline','adapted'],default='adapted');parser.add_argument('--repetitions',type=int,choices=range(1,6),default=1);args=parser.parse_args()
    try:
        source=read(args.suite)
        if 'suites' in source:
            if not args.suite_id:raise ValueError('--suite-id required for suite collection')
            suite=next((s for s in source['suites'] if s['id']==args.suite_id),None)
            if suite is None:raise ValueError('Unknown suite ID')
        else:suite=source
        results=read(args.results);results=results.get('results') if isinstance(results,dict) else results
        report=grade(suite,results,args.mode,args.repetitions)
        Path(args.output).write_text(json.dumps(report,ensure_ascii=False,indent=2))
        print(json.dumps({k:report[k] for k in ['planned','received','passed','allContractChecksPassed']}));return 0 if report['allContractChecksPassed'] else 1
    except (ValueError,KeyError,TypeError,LabError,OSError) as e:
        print('Input error:',str(e));return 2
if __name__=='__main__':raise SystemExit(main())
