"""Internal content-free qualification surface; not a model/admin proxy."""
import json,pathlib,subprocess,importlib.metadata
from http.server import BaseHTTPRequestHandler,HTTPServer
class H(BaseHTTPRequestHandler):
 def log_message(self,*a):pass
 def do_GET(self):
  if self.path!='/receipt':self.send_error(404);return
  p=pathlib.Path('/merged/MERGE-RECEIPT.json')
  result={'merge':json.loads(p.read_text()) if p.exists() else None,'packages':{}}
  for n in ['vllm','torch','transformers','tokenizers','safetensors','peft']:
   try:result['packages'][n]=importlib.metadata.version(n)
   except importlib.metadata.PackageNotFoundError:result['packages'][n]=None
  try:
   r=subprocess.run(['nvidia-smi','--query-gpu=name,uuid,driver_version,memory.total,memory.used,memory.free','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=5)
   result['nvidia_smi']={'exit':r.returncode,'csv':r.stdout.strip()[:2000],'memory_units':'MiB from nvidia-smi, not CUDA free-byte measurement'}
  except Exception:result['nvidia_smi']={'error':'unavailable'}
  b=json.dumps(result).encode();self.send_response(200);self.send_header('Content-Type','application/json');self.send_header('Content-Length',str(len(b)));self.end_headers();self.wfile.write(b)
HTTPServer(('0.0.0.0',8002),H).serve_forever()
