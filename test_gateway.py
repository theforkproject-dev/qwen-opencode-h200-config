import os,hashlib,json,http.client,threading,unittest,time,socket
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
KEY='synthetic-fixture-only';os.environ['QWEN_KEY_SHA256']=hashlib.sha256(KEY.encode()).hexdigest();os.environ['QWEN_BACKEND']='127.0.0.1:18000'
import gateway
SEEN=[]
class Backend(BaseHTTPRequestHandler):
 def log_message(self,*a):pass
 def do_GET(self):self.send_response(200);self.end_headers();self.wfile.write(b'{}')
 def do_POST(self):
  b=json.loads(self.rfile.read(int(self.headers['Content-Length'])));SEEN.append({'body':b,'authorization':self.headers.get('Authorization')})
  if b['messages'][0]['content']=='wait':time.sleep(0.8)
  if b.get('stream'):
   self.send_response(200);self.send_header('Content-Type','text/event-stream');self.end_headers()
   for part in [b'da',b'ta: {"choices":[]}\n',b'\ndata: [DONE]\n\n']:self.wfile.write(part);self.wfile.flush();time.sleep(.01)
  else:self.send_response(200);self.end_headers();self.wfile.write(b'{"choices":[{"message":{"content":"ok"}}]}')
back=ThreadingHTTPServer(('127.0.0.1',18000),Backend);front=gateway.Server(('127.0.0.1',18080),gateway.Handler)
for s in [back,front]:threading.Thread(target=s.serve_forever,daemon=True).start()
def call(method='POST',path='/v1/chat/completions',b=None,key=KEY,raw=None,headers=None):
 c=http.client.HTTPConnection('127.0.0.1',18080,timeout=4);h={'Content-Type':'application/json'}
 if key is not None:h['Authorization']='Bearer '+key
 if headers:h.update(headers)
 if raw is None and method=='POST':raw=json.dumps(b if b is not None else {'model':gateway.MODEL,'messages':[{'role':'user','content':'hello'}]})
 c.request(method,path,body=raw,headers=h);r=c.getresponse();out=(r.status,r.read(),dict(r.getheaders()));c.close();return out
class Tests(unittest.TestCase):
 def test_health(self):self.assertEqual(call('GET','/health',key=None)[0],200)
 def test_models(self):self.assertEqual(json.loads(call('GET','/v1/models')[1])['data'][0]['id'],gateway.MODEL)
 def test_auth(self):
  for method,path in [('GET','/v1/models'),('POST','/v1/chat/completions')]:
   for k in [None,'wrong','',KEY+'x']:self.assertEqual(call(method,path,key=k)[0],401)
 def test_closed_routes(self):
  for p in ['/invocations','/v1/completions','/metrics','/tokenize','/v1/load_lora_adapter','/v1/chat/completions?x=1','/v1//chat/completions','/v1/%63hat/completions','//v1/chat/completions','/admin','/']:
   self.assertEqual(call(path=p)[0],404,p)
 def test_methods(self):
  for m in ['PUT','DELETE','PATCH','HEAD','OPTIONS']:self.assertEqual(call(m)[0],405)
 def test_invalid(self):
  basic={'model':gateway.MODEL,'messages':[{'role':'user','content':'x'}]}
  bad=[[],{},dict(basic,model='fake'),dict(basic,stream=1),dict(basic,max_tokens=8193),dict(basic,max_tokens=True),dict(basic,max_tokens=0),dict(basic,max_tokens=2,max_completion_tokens=2),dict(basic,chat_template_kwargs={'enable_thinking':True}),dict(basic,extra_body={}),dict(basic,messages=[{'role':'user','content':[{'type':'image_url','image_url':{'url':'https://example.com'}}]}]),dict(basic,tools=[{'type':'web_search'}])]
  for b in bad:self.assertEqual(call(b=b)[0],400,str(b))
  for raw in ['{"model":"a","model":"b"}', '{"x":NaN}', '{}garbage', '\ufffd']:
   self.assertEqual(call(raw=raw.encode('utf-8'))[0],400)
 def test_size_and_type(self):
  self.assertEqual(call(raw='x',headers={'Content-Length':str(gateway.MAX_BODY+1)})[0],413)
  self.assertEqual(call(raw='{}',headers={'Content-Type':'text/plain'})[0],415)
  self.assertEqual(call(raw='{}',headers={'Content-Encoding':'gzip'})[0],400)
 def test_positive_and_defaults(self):
  self.assertEqual(call()[0],200);x=SEEN[-1];self.assertIsNone(x['authorization']);self.assertEqual(x['body']['max_tokens'],8192);self.assertEqual(x['body']['chat_template_kwargs'],{'enable_thinking':False})
 def test_stream(self):
  b={'model':gateway.MODEL,'messages':[{'role':'user','content':'x'}],'stream':True};r=call(b=b);self.assertEqual(r[0],200);self.assertEqual(r[1],b'data: {"choices":[]}\n\ndata: [DONE]\n\n')
 def test_busy(self):
  b={'model':gateway.MODEL,'messages':[{'role':'user','content':'wait'}]};t=threading.Thread(target=lambda:call(b=b));t.start();time.sleep(.1);self.assertEqual(call()[0],429);t.join();self.assertEqual(call()[0],200)
 def test_function_unicode(self):
  b={'model':gateway.MODEL,'messages':[{'role':'user','content':'\u03b1\n"quoted"'},{'role':'assistant','content':None,'tool_calls':[{'id':'x','type':'function','function':{'name':'read','arguments':'{"x":"a\\nb"}'}}]},{'role':'tool','tool_call_id':'x','content':'done'}],'tools':[{'type':'function','function':{'name':'read','parameters':{'type':'object','properties':{'x':{'type':'string'}}}}}],'max_completion_tokens':512}
  self.assertEqual(call(b=b)[0],200);self.assertEqual(SEEN[-1]['body']['max_tokens'],512)
if __name__=='__main__':
 try:unittest.main(verbosity=2)
 finally:back.shutdown();front.shutdown()
