"""Loopback-only runtime demo. No private corpus or educational content required."""
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse,parse_qs,unquote
import argparse,json,mimetypes
from agent_lab import AgentLab
from pipeline_lab import PipelineLab
ROOT=Path(__file__).resolve().parents[1]
lab=AgentLab(ROOT);pipeline=PipelineLab(ROOT,lab)
class Handler(BaseHTTPRequestHandler):
    def setup(self):super().setup();self.connection.settimeout(30)
    def local_request(self):
        hosts={f'127.0.0.1:{self.server.server_port}',f'localhost:{self.server.server_port}'}
        return self.headers.get('Host','') in hosts and (self.headers.get('Origin') is None or self.headers['Origin'] in {'http://'+h for h in hosts}) and self.headers.get('Sec-Fetch-Site')!='cross-site'
    def send(self,status,data,ctype='application/json; charset=utf-8'):
        if isinstance(data,(dict,list)):data=json.dumps(data,ensure_ascii=False).encode()
        if isinstance(data,str):data=data.encode()
        self.send_response(status)
        for key,value in {'Content-Type':ctype,'Content-Length':str(len(data)),'X-Content-Type-Options':'nosniff','Cache-Control':'no-store','Referrer-Policy':'no-referrer','Content-Security-Policy':"default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; connect-src 'self'; object-src 'none'; base-uri 'none'; frame-ancestors 'none'"}.items():self.send_header(key,value)
        self.end_headers()
        if self.command!='HEAD':self.wfile.write(data)
    def do_HEAD(self):self.do_GET()
    def do_GET(self):
        if not self.local_request():return self.send(403,{'error':'Local origin required'})
        url=urlparse(self.path);path=unquote(url.path)
        if path=='/api/health':return self.send(200,{'status':'ok','mode':'neutral_runtime_demo','modelConfiguredAutomatically':False})
        if path.startswith(('/api/lab/','/api/pipeline/')):
            try:status,data=(pipeline if path.startswith('/api/pipeline/') else lab).get(path,parse_qs(url.query));return self.send(status,data)
            except Exception:return self.send(500,{'error':'Could not read runtime state'})
        files={'/':'index.html','/index.html':'index.html','/app.js':'app.js','/style.css':'style.css','/mesh-viewer.js':'mesh-viewer.js'}
        if path in files:return self.file(ROOT/'web'/files[path])
        if path.startswith('/assets/pipeline/'):
            p=(ROOT/path.lstrip('/')).resolve()
            if (ROOT/'assets/pipeline').resolve() not in p.parents:return self.send(404,{'error':'Not found'})
            return self.file(p)
        return self.send(404,{'error':'Not found'})
    def do_POST(self):
        if not self.local_request():return self.send(403,{'error':'Local origin required'})
        if self.headers.get('Content-Type','').split(';')[0]!='application/json':return self.send(415,{'error':'JSON required'})
        if self.headers.get('Transfer-Encoding'):return self.send(400,{'error':'Transfer encoding unsupported'})
        try:
            size=int(self.headers.get('Content-Length','0'))
            if not 0<size<=2*1024*1024:return self.send(413,{'error':'Request too large or empty'})
            data=json.loads(self.rfile.read(size))
        except (ValueError,UnicodeDecodeError,TimeoutError):return self.send(400,{'error':'Invalid JSON request'})
        if not isinstance(data,dict):return self.send(400,{'error':'JSON object required'})
        path=urlparse(self.path).path
        if not path.startswith(('/api/lab/','/api/pipeline/')):return self.send(404,{'error':'Not found'})
        try:status,data=(pipeline if path.startswith('/api/pipeline/') else lab).post(path,data);return self.send(status,data)
        except Exception:return self.send(500,{'error':'Operation failed; inspect state before retrying'})
    def file(self,path):
        if not path.is_file():return self.send(404,{'error':'Not found'})
        return self.send(200,path.read_bytes(),mimetypes.guess_type(path.name)[0] or 'application/octet-stream')
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--port',type=int,default=8001);args=p.parse_args()
    server=ThreadingHTTPServer(('127.0.0.1',args.port),Handler);print(f'Runtime: http://127.0.0.1:{args.port}',flush=True)
    try:server.serve_forever()
    except KeyboardInterrupt:pass
    finally:server.server_close();pipeline.close();lab.close()
