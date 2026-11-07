"""Generate a class-conditioned grid with the frozen four-NFE student."""
import argparse
from pathlib import Path
import torch
from torchvision.utils import save_image
from model import Model
ROOT=Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu');p.add_argument('--labels',nargs='+',type=int,default=[0]);p.add_argument('--seed',type=int,default=42);p.add_argument('--output',default='samples.png');a=p.parse_args()
assert all(0<=v<151 for v in a.labels),'class IDs must be in [0,150]'
out=Path(a.output);assert not out.exists(),f'output already exists: {out}'
m=Model.load_checkpoint(str(ROOT/'checkpoints/few_nfe.ckpt'),'few_nfe',device=a.device)
y=torch.tensor(a.labels,dtype=torch.long,device=a.device)
with torch.inference_mode(): x=m.sample((len(y),3,64,64),device=a.device,category=y,seed=a.seed)
save_image(((x.float().cpu()+1)/2).clamp(0,1),str(out),nrow=min(8,len(y)))
print(f'Saved {len(y)} image(s) to {out} (four backbone calls).')
