"""Independently recompute reported metrics from saved per-turn predictions."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile

import numpy as np

from data.generate_data import Conversation
from eval.evaluate import full_report


def verify(root):
    read=lambda p:json.loads(p.read_text(encoding="utf-8"))
    manifest=read(root/"manifest.json")
    raw=read(root/"dataset.json")
    assert hashlib.sha256(json.dumps(raw,sort_keys=True).encode()).hexdigest()==manifest["dataset_hash"]
    with zipfile.ZipFile(root/"source.zip") as archive:
        for name,digest in manifest["source_hashes"].items():
            assert hashlib.sha256(archive.read(name.replace("\\","/"))).hexdigest()==digest,name
    data=[Conversation(**dict(c,conv_id=str(c["conv_id"]))) for c in raw]
    by={s:[c for c in data if c.split==s] for s in {c.split for c in data}}
    feats={c.conv_id:None for c in data}
    count=0
    for file in root.glob("**/seed-*/metrics.json"):
        saved=read(file);scores=np.load(file.parent/"predictions.npz")
        assert set(scores.files)=={c.conv_id for c in data}
        for c in data:
            assert len(scores[c.conv_id])==len(c.turns)
            assert np.isfinite(scores[c.conv_id]).all()
        for alpha,metrics in saved.items():
            actual,_=full_report(by,feats,lambda c,f:scores[c.conv_id],float(alpha))
            for split,r in actual.items():
                assert r["calibration"]["fpr"]<=float(alpha)+1e-12,(file,alpha)
                for section in ("classification","earliness","calibration","false_positives"):
                    for key,value in r[section].items():
                        expected=metrics[split][section][key]
                        if value is None or isinstance(value,list): assert value==expected,(file,section,key)
                        elif isinstance(value,(int,float,np.number)):
                            assert abs(float(value)-float(expected))<1e-10,(file,section,key)
                        else: assert value==expected
        count+=1
    print(f"Verified {count} experiment/seed artifacts in {root}; all calibration rates respect targets.")
    return count


def main():
    parser=argparse.ArgumentParser();parser.add_argument("runs",nargs="+",type=Path)
    args=parser.parse_args()
    for root in args.runs:verify(root)


if __name__=="__main__":main()
