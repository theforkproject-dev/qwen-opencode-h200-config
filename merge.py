"""Bounded shardwise complete vanilla LoRA merge matching PEFT 0.21.2 safe_merge.
Inputs must be verified read-only packages. Output is an ephemeral derivation,
not a separately dm-verity-mounted artifact. No remote model code is loaded.
"""
import gc,hashlib,json,os,pathlib,shutil,time

def sha(path):
 h=hashlib.sha256()
 with open(path,'rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()

def merge(base,adapter,out,expected_pairs=256,hashes=None):
 import torch
 from safetensors import safe_open
 from safetensors.torch import load_file,save_file
 torch.set_num_threads(8);torch.set_num_interop_threads(1)
 base,adapter,out=map(pathlib.Path,(base,adapter,out));out.mkdir(exist_ok=True)
 if any(out.iterdir()):raise ValueError('output must be empty')
 c=json.loads((adapter/'adapter_config.json').read_text())
 assert c['peft_type']=='LORA' and c['r']==32 and c['lora_alpha']==64 and c['bias']=='none'
 for k in ['use_dora','use_rslora','use_qalora','lora_bias','fan_in_fan_out','modules_to_save','rank_pattern','alpha_pattern','target_parameters','layer_replication','alora_invocation_tokens']:
  assert not c.get(k),f'unsupported adapter variant {k}'
 index=json.loads((base/'model.safetensors.index.json').read_text());wm=index['weight_map']
 inputs={}
 for group,path in [('base',base),('adapter',adapter)]:
  for name,h in (hashes or {}).get(group,{}).items():
   assert '/' not in name and name not in {'.','..'}
   got=sha(path/name);assert got==h,f'input hash mismatch {group}/{name}';inputs[group+'/'+name]=got
 a=load_file(adapter/'adapter_model.safetensors',device='cpu');pairs={}
 for k,v in a.items():
  assert v.dtype==torch.float32 and torch.isfinite(v).all()
  prefix='base_model.model.'
  assert k.startswith(prefix) and (k.endswith('.lora_A.weight') or k.endswith('.lora_B.weight')),k
  name=k[len(prefix):].rsplit('.lora_',1)[0]+'.weight';which=k.rsplit('.lora_',1)[1][0]
  assert name in wm and which not in pairs.setdefault(name,{})
  pairs[name][which]=v
 assert len(a)==2*expected_pairs and len(pairs)==expected_pairs
 for k,v in pairs.items():assert set(v)=={'A','B'} and v['A'].shape[0]==32 and v['B'].shape[1]==32
 report={'recipe':'PEFT0.21.2 vanilla safe_merge: base_BF16 + BF16(2 * (B_FP32 @ A_FP32))','input_hashes':inputs,'versions':{'torch':torch.__version__,'safetensors':__import__('safetensors').__version__},'updates':[],'output_hashes':{},'untargeted_equal':0,'start':time.time()}
 seen=set()
 for shard in sorted(set(wm.values())):
  assert '/' not in shard and shard.endswith('.safetensors')
  path=base/shard
  with safe_open(path,framework='pt',device='cpu') as f:
   keys=list(f.keys());assert set(keys)=={k for k,v in wm.items() if v==shard}
   tensors={}
   for k in keys:
    v=f.get_tensor(k)
    if k in pairs:
     A=pairs[k]['A'];B=pairs[k]['B'];assert v.dtype==torch.bfloat16 and tuple(v.shape)==(B.shape[0],A.shape[1])
     delta=(B@A)*2.0
     assert torch.isfinite(delta).all()
     # Independently recompute selected delta coordinates using float64 dots.
     max_error=0.0
     for r,col in [(0,0),(v.shape[0]//2,v.shape[1]//2),(v.shape[0]-1,v.shape[1]-1)]:
      ref=float(torch.dot(B[r].double(),A[:,col].double())*2);val=float(delta[r,col]);err=abs(ref-val);assert err<=1e-6+1e-5*abs(ref);max_error=max(max_error,err)
     merged=v.clone();merged+=delta.to(v.dtype);del delta
     assert torch.isfinite(merged).all()
     changed=int(torch.count_nonzero(merged!=v));assert changed>0,f'no applied update {k}'
     tensors[k]=merged;seen.add(k)
     report['updates'].append({'name':k,'shape':list(v.shape),'changed':changed,'independent_delta_max_abs_error':max_error})
    else:tensors[k]=v
   save_file(tensors,str(out/shard),metadata={'format':'pt'})
   # Reload all saved tensors. The untouched base values must be exactly equal.
   with safe_open(out/shard,framework='pt',device='cpu') as g:
    assert list(g.keys())==keys
    for k in keys:
     assert torch.equal(g.get_tensor(k),tensors[k]),f'save mismatch {k}'
     if k not in pairs:
      assert torch.equal(g.get_tensor(k),f.get_tensor(k));report['untargeted_equal']+=1
   del tensors
  gc.collect();report['output_hashes'][shard]=sha(out/shard)
  print(json.dumps({'stage':'merged-shard','shard':shard,'updates_so_far':len(seen)}),flush=True)
 assert seen==set(pairs)
 for name in ['config.json','generation_config.json','model.safetensors.index.json','LICENSE']:
  shutil.copyfile(base/name,out/name)
 for name in ['tokenizer.json','tokenizer_config.json','chat_template.jinja']:
  shutil.copyfile(adapter/name,out/name)
 for p in out.iterdir():
  if p.is_file() and p.name not in report['output_hashes']:report['output_hashes'][p.name]=sha(p)
 assert not any('lora_' in k for k in wm)
 report['elapsed_seconds']=time.time()-report['start'];report['all_updates']=len(seen)
 (out/'MERGE-RECEIPT.json').write_text(json.dumps(report,sort_keys=True))
 print(json.dumps({'stage':'merge-complete','all_updates':len(seen),'untargeted_equal':report['untargeted_equal'],'receipt_sha256':sha(out/'MERGE-RECEIPT.json'),'elapsed_seconds':report['elapsed_seconds']}),flush=True)
 return report

if __name__=='__main__':
 merge('/tinfoil/models/qwen-base','/tinfoil/models/qwen-adapter','/merged',hashes=json.loads(os.environ['QWEN_INPUT_HASHES']))
