"""Verify the exact four-NFE checkpoint and official model contract."""
import argparse,hashlib,json,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT))
from model import Model
import model,distillation.submission,teacher.model,teacher.diffusion,src.utils
for mod in (model,distillation.submission,teacher.model,teacher.diffusion,src.utils):assert Path(mod.__file__).resolve().is_relative_to(ROOT),mod.__file__
p=argparse.ArgumentParser();p.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu');a=p.parse_args()
path=ROOT/'checkpoints/few_nfe.ckpt';h=hashlib.sha256()
with path.open('rb') as f:
 for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
assert h.hexdigest()=='36a457612d241a0a462c37d2a73a33a73f29257fb4d297dbd296851a2b0c1b66'
m=Model.load_checkpoint(str(path),'few_nfe',device=a.device).eval();assert m.count_parameters()==48883107
calls=[];hook=m.backbone.register_forward_pre_hook(lambda *_:calls.append(1));y=torch.arange(32,device=a.device)
with torch.inference_mode():
 x=m.sample((32,3,64,64),device=a.device,category=y,seed=42);n=len(calls);calls.clear();second=m.sample((32,3,64,64),device=a.device,category=y,seed=42)
hook.remove();assert n==4 and len(calls)==4 and x.shape==(32,3,64,64) and x.dtype==torch.float32 and x.isfinite().all() and x.min()>=-1 and x.max()<=1 and torch.equal(x,second)
print(json.dumps({'sha256':h.hexdigest(),'strict_load':True,'parameters':m.count_parameters(),'backbone_calls':n,'batch':32,'shape':list(x.shape),'dtype':str(x.dtype),'min':float(x.min()),'max':float(x.max()),'deterministic':True,'device':a.device}))
