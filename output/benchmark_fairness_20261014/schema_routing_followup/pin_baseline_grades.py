#!/usr/bin/env python3
"""Pin the exact existing baseline grades; never call a model or change a grade."""
from pathlib import Path
import argparse
import importlib.util
import json

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('schema_routing_control',HERE/'run_followup.py')
driver=importlib.util.module_from_spec(spec);spec.loader.exec_module(driver)
DEFAULT=HERE/'baseline_grade_manifest.json'


def collect(out):
    driver.verify(out)
    cells=[];hashes={}
    missing=[]
    for row in driver.read(out/'frozen/reused_one_shots.json')['candidates']:
        for judge in driver.primary.JUDGES:
            item=driver.primary.accepted(driver.PRIMARY,judge,row)
            if item is None:
                missing.append({'candidate_id':row['candidate_id'],'judge':judge});continue
            path,response=item
            files={}
            for name in ('status.json','parsed_response.json','prompt.json','raw_response.json','telemetry.json'):
                p=path.with_name(name)
                assert p.is_file(),str(p)
                key=str(p.resolve());hashes[key]=driver.sha(p);files[name]=key
            cells.append({'candidate_id':row['candidate_id'],'judge':judge,'source_files':files,'response_sha256':files and driver.sha(path.with_name('parsed_response.json'))})
    if missing:raise RuntimeError('Wait for all 9 original baseline valid grades: '+json.dumps(missing))
    assert len(cells)==9
    return {'created_at':driver.now(),'policy':'Exact original strict-validator accepted one-shot grades. No new inference, score alteration or grade selection. Referenced immutable primary artifacts pinned before final comparison.','source_campaign':str(driver.PRIMARY),'followup':str(out),'cells':cells,'hashes':hashes,'helper_sha256':driver.sha(Path(__file__))}


def verify_manifest(path):
    record=driver.read(path)
    assert len(record['cells'])==9
    for source,digest in record['hashes'].items():assert driver.sha(Path(source))==digest,source
    for cell in record['cells']:
        row={'candidate_id':cell['candidate_id']}
        item=driver.primary.accepted(driver.PRIMARY,cell['judge'],row)
        assert item and str(item[0].resolve())==cell['source_files']['status.json']
        assert driver.sha(item[0].with_name('parsed_response.json'))==cell['response_sha256']
    return record


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=driver.DEFAULT);p.add_argument('--manifest',type=Path,default=DEFAULT);p.add_argument('--verify',action='store_true');args=p.parse_args()
    driver.configure()
    if args.manifest.exists() or args.verify:
        record=verify_manifest(args.manifest)
    else:
        record=collect(args.output.resolve());driver.write(args.manifest,record);verify_manifest(args.manifest)
    print(json.dumps({'baseline_grades':len(record['cells']),'manifest':str(args.manifest),'verified':True}),flush=True)

if __name__=='__main__':main()
