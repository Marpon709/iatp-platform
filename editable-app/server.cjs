'use strict';
const http=require('node:http');
const fs=require('node:fs');
const path=require('node:path');
const TYPES={'.html':'text/html; charset=utf-8','.js':'text/javascript; charset=utf-8','.css':'text/css; charset=utf-8','.json':'application/json; charset=utf-8','.geojson':'application/geo+json; charset=utf-8','.svg':'image/svg+xml','.ttf':'font/ttf','.txt':'text/plain; charset=utf-8','.png':'image/png','.pbf':'application/x-protobuf'};
async function startServer({webRoot,uiRoot=__dirname,port=0,getBridge=()=>null}){
  const server=http.createServer((req,res)=>{
    const send=(status,value)=>{res.writeHead(status,{'Content-Type':'application/json; charset=utf-8','Cache-Control':'no-store'});res.end(JSON.stringify(value));};
    let pathname;try{pathname=decodeURIComponent(new URL(req.url,'http://localhost').pathname);}catch{return send(400,{success:false,message:'地址无效'});}
    if(pathname.startsWith('/api/')){
      const bridge=getBridge();if(!bridge)return send(503,{success:false,message:'计算服务正在启动，请稍候。'});
      const target=new URL(req.url,bridge);
      const headers={'Content-Type':req.headers['content-type']||'application/json'};
      if(req.headers['content-length']!==undefined)headers['Content-Length']=req.headers['content-length'];
      const upstream=http.request(target,{method:req.method,headers,timeout:250000},response=>{res.writeHead(response.statusCode,{'Content-Type':response.headers['content-type']||'application/json','Cache-Control':'no-store'});response.pipe(res);});
      upstream.on('timeout',()=>upstream.destroy());upstream.on('error',()=>{if(!res.headersSent)send(502,{success:false,message:'计算服务暂未响应，请重新启动软件。'});else res.end();});req.pipe(upstream);return;
    }
    if(req.method!=='GET'&&req.method!=='HEAD')return send(405,{success:false,message:'请求方式不支持'});
    const files={'/settings.html':'settings.html','/settings.js':'settings.js','/desktop.css':'desktop.css','/loading.html':'loading.html'};
    const special=files[pathname];const root=path.resolve(special?uiRoot:webRoot),relative=special||pathname.replace(/^\/+/, '')||'index.html',file=path.resolve(root,relative);
    if(file!==root&&!file.startsWith(root+path.sep))return send(403,{success:false,message:'地址无效'});
    fs.stat(file,(error,stat)=>{if(error||!stat.isFile())return send(404,{success:false,message:'文件不存在'});
      res.writeHead(200,{'Content-Type':TYPES[path.extname(file)]||'application/octet-stream','Content-Length':stat.size,'Cache-Control':'no-store','X-Content-Type-Options':'nosniff','Referrer-Policy':'no-referrer'});
      if(req.method==='HEAD')res.end();else fs.createReadStream(file).on('error',()=>res.end()).pipe(res);
    });
  });
  await new Promise((resolve,reject)=>{server.once('error',reject);server.listen(port,'127.0.0.1',resolve);});
  return {server,url:`http://127.0.0.1:${server.address().port}`,close:()=>new Promise(resolve=>server.close(resolve))};
}
module.exports={startServer};
