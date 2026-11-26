import pathlib,tempfile,json,torch,hashlib,subprocess,sys
from safetensors.torch import save_file,load_file
import merge
with tempfile.TemporaryDirectory() as d:
 d=pathlib.Path(d);b=d/'base';a=d/'adapter';o=d/'out';b.mkdir();a.mkdir()
 torch.manual_seed(2631)
 base={};adapter={};expected={}
 for i in range(2):
  for name in ['q_proj','k_proj','v_proj','o_proj','up_proj','gate_proj','down_proj']:
   k=f'model.language_model.layers.{i}.{name}.weight';w=torch.randn(64,48,dtype=torch.bfloat16);A=torch.randn(32,48)*0.025;B=torch.randn(64,32)*0.025
   base[k]=w;adapter['base_model.model.'+k[:-7]+'.lora_A.weight']=A;adapter['base_model.model.'+k[:-7]+'.lora_B.weight']=B
   # Distinct implementation: explicit rank accumulation, FP64 reference then cast.
   delta=torch.zeros(64,48,dtype=torch.float64)
   for r in range(32):delta+=B[:,r].double().unsqueeze(1)*A[r,:].double().unsqueeze(0)*2
   expected[k]=w+delta.to(torch.bfloat16)
 base['untouched']=torch.randn(10,10,dtype=torch.bfloat16)
 save_file(base,str(b/'model-00001-of-00001.safetensors'));save_file(adapter,str(a/'adapter_model.safetensors'))
 (b/'model.safetensors.index.json').write_text(json.dumps({'weight_map':{k:'model-00001-of-00001.safetensors' for k in base}}))
 for n in ['config.json','generation_config.json']: (b/n).write_text('{}')
 (b/'LICENSE').write_text('synthetic')
 for n in ['tokenizer.json','tokenizer_config.json']:(a/n).write_text('{}')
 (a/'chat_template.jinja').write_text('{{ messages }}')
 (a/'adapter_config.json').write_text(json.dumps({'peft_type':'LORA','r':32,'lora_alpha':64,'bias':'none'}))
 report=merge.merge(b,a,o,expected_pairs=14,hashes={'adapter':{'adapter_model.safetensors':merge.sha(a/'adapter_model.safetensors')}})
 got=load_file(o/'model-00001-of-00001.safetensors');assert report['all_updates']==14 and report['untargeted_equal']==1
 for k in expected:assert torch.equal(got[k],expected[k]),k
 assert torch.equal(got['untouched'],base['untouched'])
 # Fail closed on incomplete adapter. Fresh process avoids torch global thread reinit.
 child='import merge;merge.merge('+repr(str(b))+','+repr(str(a))+','+repr(str(d/'bad'))+',expected_pairs=15)'
 p=subprocess.run([sys.executable,'-c',child],capture_output=True);assert p.returncode!=0
 print(json.dumps({'passed':True,'target_types':7,'layers':2,'updated_matrices':14,'reference':'independent explicit FP64 rank accumulation','untargeted_equal':True,'incomplete_adapter_rejected':True,'torch':torch.__version__}))
