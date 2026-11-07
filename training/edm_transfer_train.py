"""Matched VP-DMD teacher substitution. Experimental checkpoints are tmpfs only."""
import argparse,copy,hashlib,json,random,shutil,sys,time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from model import Model
from distillation.core import coefficients
from distillation.submission import submission_backbone
from teacher.cfg import apply_cfg
from research_campaign.legacy_helpers import Evaluator,rollout,digest
from research_campaign.external_distill_20261104.model.edm import load_model,denoise
from research_campaign.phase4_few_20261104.trajectory_train import state_at
from research_campaign.phase4_few_20261104.energy_train import final
from dataset import category_from_filename

BASE=ROOT/'research_campaign/phase4_few_20261104/dmd_full_pilot/selected_001000.ckpt'
VP=Path('/root/DMD/experiments/night_run/fresh_teacher/final.pt')
EDM=Path('/run/user/0/fastgen_teammate_edm_20261105/objects/81/66/816638eab17241bcabf07eadcbf49f3ee2f0c03b8f76e8d83d811ee15cf437cc')
FLOOR=int(2.25*1024**3)
def main():
 p=argparse.ArgumentParser();p.add_argument('--arm',choices=('vp','edm'),required=True);p.add_argument('--name',required=True);p.add_argument('--steps',type=int,default=200);p.add_argument('--seed',type=int,default=680101);p.add_argument('--batch',type=int,default=16);p.add_argument('--micro',type=int,default=4);p.add_argument('--warmup',type=int,default=100);p.add_argument('--smoke',action='store_true');a=p.parse_args()
 assert a.batch%a.micro==0 and a.steps>0
 for path,expected in [(BASE,'053195fe1da3c4f884cb34feaaeb878030744403768f3a7fc7a4fe5ea3c99d5a'),(VP,'b5c09c58954352f71291a2c4421c5f542ac6abb392acda4074c1f118ab9c3a27'),(EDM,'816638eab17241bcabf07eadcbf49f3ee2f0c03b8f76e8d83d811ee15cf437cc')]:assert digest(path)==expected
 assert min(shutil.disk_usage(ROOT).free,shutil.disk_usage('/dev/shm').free)>FLOOR+8*1024**2
 tmp=Path('/run/user/0/fastgen_edm_score_transfer_20261105');tmp.mkdir(exist_ok=True)
 assert shutil.disk_usage(tmp).free>1024**3
 out=ROOT/'research_campaign/edm_score_transfer_20261105'/a.name;out.mkdir(exist_ok=False)
 torch.set_num_threads(8)
 ck=torch.load(VP,map_location='cpu',weights_only=False); lines=(ROOT/'overnight/train.txt').read_text().splitlines();assert ck['train_paths_sha256']==hashlib.sha256('\n'.join(lines).encode()).hexdigest()
 real=submission_backbone().cuda().eval().requires_grad_(False);real.load_state_dict(ck['ema'],strict=True);del ck
 fake=copy.deepcopy(real).train().requires_grad_(True);edm=load_model('cuda',checkpoint=EDM).eval().requires_grad_(False)
 g=Model.load_checkpoint(str(BASE),'few_nfe',device='cuda');anchor=copy.deepcopy(g).eval().requires_grad_(False);ema=copy.deepcopy(g).eval().requires_grad_(False)
 assert sum(p.numel() for p in g.parameters())==48883107 and g.sampling_times.tolist()==[999,741,483,226]
 t0=torch.arange(50,951,device='cuda');aa,ss=coefficients(g.alpha_bar,t0);q=(ss/aa).flatten();eligible=t0[(q>=np.exp(-4.2))&(q<=np.exp(1.8))];assert eligible[0]==50 and eligible[-1]==596 and len(eligible)==547
 mapping=json.loads((ROOT/'data/pokemon-generation-one-22k/category_to_id.json').read_text());pool=torch.tensor([mapping[category_from_filename(s)] for s in lines],device='cuda');assert len(pool)==16324
 opt=torch.optim.AdamW(g.parameters(),lr=1e-6,weight_decay=0);fopt=torch.optim.AdamW(fake.parameters(),lr=1e-4,weight_decay=0)
 # Reset after every initialization, identically in both arms.
 torch.manual_seed(a.seed);np.random.seed(a.seed);random.seed(a.seed);finger=hashlib.sha256()
 cfg={'arm':a.arm,'seed':a.seed,'steps':a.steps,'batch':a.batch,'micro':a.micro,'warmup':a.warmup,'fake_updates':2,'generator_lr':1e-6,'fake_lr':1e-4,'EMA':.99,'anchor_to_initialized_DMD':1.,'CFG':2.,'timestep_uniform':[50,596],'class_distribution':'empirical image frequencies of overnight/train.txt','common_initialization_sha256':digest(BASE),'edm_sha256':digest(EDM),'vp_sha256':digest(VP),'teacher_exposed_dev_reference':True,'precision':'BF16 forward FP32 math','checkpoint_policy':'temporary /run only; no optimizer saved','training_source':'new controlled VP-DMD score-transfer experiment'}
 (out/'config.json').write_text(json.dumps(cfg,indent=2))
 def draw():
  y=pool[torch.randint(len(pool),(a.micro,),device='cuda')];z=torch.randn((a.micro,3,64,64),device='cuda');finger.update(y.cpu().numpy().tobytes());finger.update(z.flatten()[:8].cpu().numpy().tobytes());return y,z
 def fake_step():
  fake.train().requires_grad_(True);fopt.zero_grad(set_to_none=True);vl=[]
  for _ in range(a.batch//a.micro):
   y,z=draw()
   with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16): im=final(g,state_at(g,z,y,3),y)
   t=torch.randint(50,597,(len(y),),device='cuda');finger.update(t.cpu().numpy().tobytes());aa,ss=coefficients(g.alpha_bar,t);noise=torch.randn_like(im)
   with torch.autocast('cuda',dtype=torch.bfloat16):loss=F.mse_loss(fake(aa*im+ss*noise,t,y).float(),noise)
   assert loss.isfinite();(loss/(a.batch//a.micro)).backward();vl.append(float(loss.detach()))
  grad=float(torch.nn.utils.clip_grad_norm_(fake.parameters(),1,error_if_nonfinite=True));assert grad>0;fopt.step();fopt.zero_grad(set_to_none=True);return float(np.mean(vl)),grad
 def gen_step(probe=False):
  fake.eval().requires_grad_(False);g.train();opt.zero_grad(set_to_none=True);vals=[];flow=[]
  for mi in range(a.batch//a.micro):
   y,z=draw()
   with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):old=rollout(anchor,z,y)
   outputs=[]
   def hk(_m,_a,o):o.retain_grad();outputs.append(o)
   hook=g.backbone.register_forward_hook(hk) if probe and mi==0 else None
   with torch.autocast('cuda',dtype=torch.bfloat16):im=rollout(g,z,y)
   if hook:hook.remove()
   t=torch.randint(50,597,(len(y),),device='cuda');finger.update(t.cpu().numpy().tobytes());aa,ss=coefficients(g.alpha_bar,t);noise=torch.randn_like(im);x=aa*im.detach()+ss*noise
   with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
    fe=fake(x,t,y).float()
    if a.arm=='vp':
     u=real(x,t,torch.full_like(y,151));c=real(x,t,y);eps=apply_cfg(u,c,2.).float();rx=(x-ss*eps)/aa
    else:
     q=(ss/aa).flatten();ys=x/aa
     u=denoise(edm,ys,q,torch.full_like(y,151));c=denoise(edm,ys,q,y);rx=apply_cfg(u,c,2.).float()
     eps=(ys-rx)/(ss/aa)
     assert torch.allclose((x-ss*eps)/aa,rx,atol=.0001,rtol=.0001)
    fx=(x-ss*fe)/aa;norm=(im.detach()-rx).abs().mean((1,2,3),keepdim=True).clamp_min(.001);grad=(fx-rx)/norm
   assert grad.isfinite().all() and rx.isfinite().all();dm=.5*F.mse_loss(im,im.detach()-grad);reg=F.mse_loss(im,old);loss=dm+reg;assert loss.isfinite();(loss/(a.batch//a.micro)).backward()
   if probe and mi==0:flow=[float(o.grad.float().norm()) for o in outputs];assert len(flow)==4 and all(v>0 and np.isfinite(v) for v in flow)
   vals.append([float(dm.detach()),float(reg.detach()),float(grad.square().mean()),float(rx.square().mean()),float(eps.square().mean())])
  gn=float(torch.nn.utils.clip_grad_norm_(g.parameters(),1,error_if_nonfinite=True));assert 0<gn<1000;opt.step();opt.zero_grad(set_to_none=True)
  return np.array(vals).mean(0).tolist(),gn,flow
 start=time.monotonic();torch.cuda.reset_peak_memory_stats()
 for i in range(2 if a.smoke else a.warmup):fake_step()
 for step in range(1,a.steps+1):
  assert min(shutil.disk_usage(ROOT).free,shutil.disk_usage('/dev/shm').free)>=FLOOR
  fl,fg=fake_step();fl,fg=fake_step();v,gn,flow=gen_step(probe=(step==1))
  with torch.no_grad():
   for ep,gp in zip(ema.parameters(),g.parameters()):ep.lerp_(gp,.01)
  row={'step':step,'dmd_loss':v[0],'anchor_mse':v[1],'dmd_grad_mse':v[2],'teacher_x0_rms':v[3]**.5,'teacher_epsilon_rms':v[4]**.5,'fake_loss':fl,'fake_grad':fg,'generator_grad':gn,'four_nfe_output_grads':flow,'input_rng_sha256':finger.hexdigest(),'seconds':time.monotonic()-start,'peak_gpu_bytes':torch.cuda.max_memory_allocated()}
  with (out/'train.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
  if step==1 or step%25==0:print(json.dumps(row),flush=True)
 if not a.smoke:
  path=tmp/(a.name+'_final.ckpt');assert not path.exists() and shutil.disk_usage(tmp).free>300*1024**2
  with path.open('xb') as f:torch.save({'state_dict':{k:v.detach().cpu() for k,v in ema.state_dict().items()},'config':cfg,'step':a.steps},f)
  checked=Model.load_checkpoint(str(path),'few_nfe',device='cuda');assert checked.sampling_times.tolist()==[999,741,483,226]
  print(json.dumps({'temporary_checkpoint':str(path),'sha256':digest(path),'warning':'tmpfs, not persistent'}),flush=True)
  del g,anchor,fake,real,edm,ema;torch.cuda.empty_cache()
  ev=Evaluator()
  for seed in (42,1042,2042):
   result,_=ev.evaluate(checked,seed);result.update(arm=a.arm,checkpoint_sha256=digest(path),reference_limitation='teacher trained on development reference, not independent')
   (out/f'dev_seed_{seed}.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
 (out/'result.json').write_text(json.dumps({'steps_completed':a.steps,'input_rng_sha256':finger.hexdigest(),'runtime_s':time.monotonic()-start,'peak_gpu_bytes':torch.cuda.max_memory_allocated()},indent=2))
 print(json.dumps({'completed':a.steps,'input_rng_sha256':finger.hexdigest()}),flush=True)
if __name__=='__main__':main()
