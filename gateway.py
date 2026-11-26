"""Exact-route, text-only, two-generation gateway. No content/credential logging."""
import hashlib,hmac,http.client,json,os,socket,threading,time
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
MODEL='qwen3.8-opencode'
MAX_BODY=4*1024*1024
MAX_OUTPUT=16384
KEY_HASH=os.environ['QWEN_KEY_SHA256']
BACKEND=os.environ.get('QWEN_BACKEND','model:8000')
assert len(KEY_HASH)==64 and all(c in '0123456789abcdef' for c in KEY_HASH)
active=threading.BoundedSemaphore(2)

def unique(pairs):
 d={}
 for k,v in pairs:
  if k in d:raise ValueError('duplicate key')
  d[k]=v
 return d

def validate(b):
 if not isinstance(b,dict):raise ValueError('object required')
 fields={'model','messages','tools','tool_choice','parallel_tool_calls','stream','stream_options','temperature','top_p','top_k','max_tokens','max_completion_tokens','stop','seed','presence_penalty','frequency_penalty','repetition_penalty','chat_template_kwargs','response_format','reasoning_effort'}
 if set(b)-fields:raise ValueError('unsupported field')
 if b.get('model')!=MODEL:raise ValueError('unknown model')
 if type(b.get('stream',False)) is not bool:raise ValueError('stream must be boolean')
 msgs=b.get('messages')
 if not isinstance(msgs,list) or not 1<=len(msgs)<=512:raise ValueError('messages required')
 for m in msgs:
  if not isinstance(m,dict) or set(m)-{'role','content','name','tool_calls','tool_call_id'}:raise ValueError('invalid message')
  if m.get('role') not in {'system','user','assistant','tool','developer'}:raise ValueError('invalid role')
  c=m.get('content')
  if c is not None and not isinstance(c,str):
   if not isinstance(c,list) or any(not isinstance(v,dict) or set(v)!={'type','text'} or v.get('type')!='text' or not isinstance(v.get('text'),str) for v in c):raise ValueError('text only')
  if 'tool_calls' in m:
   if not isinstance(m['tool_calls'],list):raise ValueError('invalid tool calls')
   for tc in m['tool_calls']:
    if not isinstance(tc,dict) or tc.get('type')!='function' or not isinstance(tc.get('function'),dict):raise ValueError('function only')
 tools=b.get('tools',[])
 if not isinstance(tools,list) or len(tools)>64:raise ValueError('too many tools')
 for t in tools:
  if not isinstance(t,dict) or set(t)-{'type','function'} or t.get('type')!='function' or not isinstance(t.get('function'),dict):raise ValueError('function tools only')
 if b.get('reasoning_effort') not in (None,'none','low'):raise ValueError('unsupported reasoning effort under thinking-off profile')
 if b.get('chat_template_kwargs',{'enable_thinking':False})!={'enable_thinking':False}:raise ValueError('thinking must remain off')
 if 'max_tokens' in b and 'max_completion_tokens' in b:raise ValueError('conflicting output limits')
 n=b.pop('max_completion_tokens',b.get('max_tokens',MAX_OUTPUT))
 if type(n) is not int or not 1<=n<=MAX_OUTPUT:raise ValueError('output limit')
 for k in ['temperature','top_p','presence_penalty','frequency_penalty','repetition_penalty']:
  if k in b and (type(b[k]) not in (int,float) or not __import__('math').isfinite(b[k])):raise ValueError('finite sampling required')
 b['max_tokens']=n;b['chat_template_kwargs']={'enable_thinking':False}
 b.setdefault('temperature',0.8);b.setdefault('top_p',1.0);b.setdefault('top_k',20)
 return b

class Handler(BaseHTTPRequestHandler):
 protocol_version='HTTP/1.0'
 server_version='QwenGateway/1'
 def log_message(self,*a):pass
 def setup(self):
  super().setup();self.connection.settimeout(30)
 def reply(self,status,obj):
  data=json.dumps(obj,separators=(',',':')).encode();self.send_response(status);self.send_header('Content-Type','application/json');self.send_header('Content-Length',str(len(data)));self.send_header('Cache-Control','no-store');self.end_headers();self.wfile.write(data)
 def auth(self):
  vals=self.headers.get_all('Authorization',[])
  if len(vals)!=1 or not vals[0].startswith('Bearer '):return False
  return hmac.compare_digest(hashlib.sha256(vals[0][7:].encode()).hexdigest(),KEY_HASH)
 def do_GET(self):
  if self.path=='/health':
   c=http.client.HTTPConnection(BACKEND,timeout=3)
   try:c.request('GET','/health');r=c.getresponse();r.read(4096);self.reply(200 if r.status==200 else 503,{'ready':r.status==200})
   except Exception:self.reply(503,{'ready':False})
   finally:c.close()
   return
  if self.path=='/v1/qualification' and self.requestline.split(' ')[1]=='/v1/qualification':
   if not self.auth():self.reply(401,{'error':'unauthorized'});return
   c=http.client.HTTPConnection(os.environ.get('QWEN_QUALIFIER','model:8002'),timeout=10)
   try:
    c.request('GET','/receipt');r=c.getresponse();data=r.read(2*1024*1024);self.reply(r.status,json.loads(data))
   except Exception:self.reply(503,{'error':'qualification unavailable'})
   finally:c.close()
   return
  if self.path!='/v1/models' or self.requestline.split(' ')[1]!='/v1/models':self.reply(404,{'error':'route not allowed'});return
  if not self.auth():self.reply(401,{'error':'unauthorized'});return
  self.reply(200,{'object':'list','data':[{'id':MODEL,'object':'model','owned_by':'amotivv-evaluation'}]})
 def do_POST(self):
  if self.path!='/v1/chat/completions' or self.requestline.split(' ')[1]!='/v1/chat/completions':self.reply(404,{'error':'route not allowed'});return
  if not self.auth():self.reply(401,{'error':'unauthorized'});return
  if self.headers.get_all('Transfer-Encoding') or self.headers.get_all('Content-Encoding') or len(self.headers.get_all('Content-Length',[]))!=1:
   self.reply(400,{'error':'explicit unencoded length required'});return
  if self.headers.get_content_type()!='application/json':self.reply(415,{'error':'JSON required'});return
  try:n=int(self.headers['Content-Length'])
  except (ValueError,TypeError):self.reply(400,{'error':'invalid length'});return
  if not 0<n<=MAX_BODY:self.reply(413,{'error':'body limit'});return
  try:
   raw=self.rfile.read(n)
   if len(raw)!=n:raise ValueError('short body')
   b=validate(json.loads(raw,object_pairs_hook=unique,parse_constant=lambda x:(_ for _ in ()).throw(ValueError('nonfinite'))))
  except (ValueError,UnicodeError,socket.timeout):self.reply(400,{'error':'invalid or prohibited request'});return
  if not active.acquire(blocking=False):self.reply(429,{'error':'two generations already active'});return
  c=http.client.HTTPConnection(BACKEND,timeout=900);begun=False
  try:
   c.request('POST','/v1/chat/completions',json.dumps(b,separators=(',',':')),{'Content-Type':'application/json','Connection':'close'})
   r=c.getresponse();self.send_response(r.status);begun=True
   self.send_header('Content-Type',r.getheader('Content-Type','application/json'));self.send_header('Cache-Control','no-store');self.send_header('X-Accel-Buffering','no');self.end_headers()
   total=0
   while True:
    buf=r.read1(65536)
    if not buf:break
    total+=len(buf)
    if total>32*1024*1024:break
    self.wfile.write(buf);self.wfile.flush()
  except (BrokenPipeError,ConnectionResetError,socket.timeout,OSError,http.client.HTTPException):
   if not begun:
    try:self.reply(502,{'error':'backend unavailable'})
    except OSError:pass
  finally:c.close();active.release()
 def do_PUT(self):self.reply(405,{'error':'method not allowed'})
 do_DELETE=do_PUT
 do_PATCH=do_PUT
 do_OPTIONS=do_PUT
 do_HEAD=do_PUT

class Server(ThreadingHTTPServer):
 daemon_threads=True
 request_queue_size=8
 def __init__(self,*a,**kw):self.slots=threading.BoundedSemaphore(16);super().__init__(*a,**kw)
 def process_request(self,req,addr):
  if not self.slots.acquire(False):self.shutdown_request(req);return
  try:super().process_request(req,addr)
  except BaseException:self.slots.release();raise
 def process_request_thread(self,req,addr):
  try:super().process_request_thread(req,addr)
  finally:self.slots.release()

if __name__=='__main__':Server(('0.0.0.0',8080),Handler).serve_forever()
