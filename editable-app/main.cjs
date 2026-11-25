'use strict';
const {app,BrowserWindow,Menu,ipcMain,dialog,safeStorage,shell}=require('electron');
const fs=require('node:fs');
const path=require('node:path');
const os=require('node:os');
const net=require('node:net');
const http=require('node:http');
const {spawn}=require('node:child_process');
const {startServer}=require('./server.cjs');
app.setName('IATP');app.setAppUserModelId('com.iatp.platform');app.commandLine.appendSwitch('lang','zh-CN');
// 独立的配置、日志、缓存和数据库，避免与原展示版共享运行数据。
app.setPath('userData',path.join(app.getPath('appData'),'IATPPlatform'));
let window,local,bridgeUrl=null,children=[],starting=false,closing=false;
const resources=app.isPackaged?process.resourcesPath:path.resolve(__dirname,'../resources');
const webRoot=path.join(__dirname,'web');
const python=path.join(resources,'runtime','python','python.exe');
const backend=path.join(resources,'backend');
const bridge=path.join(resources,'bridge','service.py');
const dataDir=()=>app.getPath('userData');
function clean(value){let s=String(value);for(const v of Object.values(configuration.keys||{}))if(v)s=s.split(v).join('[已隐藏]');return s.replace(/([?&](?:tk|key|token|api_key)=)[^\s&"']+/gi,'$1[已隐藏]');}
function log(value){try{fs.mkdirSync(dataDir(),{recursive:true});fs.appendFileSync(path.join(dataDir(),'startup.log'),new Date().toISOString()+' '+clean(value)+'\n');}catch{}}
let configuration={keys:{}};
function readConfiguration(){try{const saved=JSON.parse(fs.readFileSync(path.join(dataDir(),'configuration.json'),'utf8'));if(saved.encrypted&&safeStorage.isEncryptionAvailable())return {keys:JSON.parse(safeStorage.decryptString(Buffer.from(saved.encrypted,'base64')))};}catch{}return {keys:{}};}
function storeConfiguration(keys){if(!safeStorage.isEncryptionAvailable())throw new Error('Windows凭据存储暂不可用，请重新登录Windows后再试。');fs.mkdirSync(dataDir(),{recursive:true});fs.writeFileSync(path.join(dataDir(),'configuration.json'),JSON.stringify({version:1,encrypted:safeStorage.encryptString(JSON.stringify(keys)).toString('base64')}));configuration={keys};}
function settingSender(event){return Boolean(local&&event.senderFrame?.url===local.url+'/settings.html');}
function freePort(){return new Promise((resolve,reject)=>{const s=net.createServer();s.once('error',reject);s.listen(0,'127.0.0.1',()=>{const p=s.address().port;s.close(()=>resolve(p));});});}
function json(url){return new Promise((resolve,reject)=>{const request=http.get(url,{timeout:2500},response=>{let text='';response.on('data',x=>text+=x);response.on('end',()=>{try{resolve(JSON.parse(text));}catch(error){reject(error);}});});request.on('error',reject);request.on('timeout',()=>request.destroy(new Error('timeout')));});}
function status(message){window?.webContents.send('application:status',message);log(message);}
function child(script,args,env){const p=spawn(python,[script,...args],{cwd:backend,env,windowsHide:true,stdio:['ignore','pipe','pipe']});children.push(p);p.stdout.on('data',x=>log(x.toString()));p.stderr.on('data',x=>log(x.toString()));p.on('error',error=>log(error.message));return p;}
async function waitReady(p,url,label){for(let i=0;i<150;i++){if(p.exitCode!==null)throw new Error(label+'未能启动，请在帮助菜单中查看启动记录。');try{const value=await json(url);if(value.success)return;}catch{}await new Promise(resolve=>setTimeout(resolve,1000));}throw new Error(label+'启动超时，请重新打开软件。');}
async function stopServices(){bridgeUrl=null;const old=children;children=[];for(const p of old)if(p.exitCode===null){try{p.kill();}catch{}}}
async function startServices({selfTest=false}={}){
  if(starting)return;starting=true;
  try{
    if(!selfTest)await window.loadURL(local.url+'/loading.html');status('正在准备南京计算环境…');await stopServices();
    if(!fs.existsSync(python)||!fs.existsSync(path.join(backend,'app.py')))throw new Error('计算环境不完整，请将压缩包完整解压后运行。');
    const backendPort=await freePort(),bridgePort=await freePort();
    const env={...process.env,PYTHONUTF8:'1',PYTHONIOENCODING:'utf-8',PYTHONUNBUFFERED:'1',PYTHONDONTWRITEBYTECODE:'1',PYTHONNOUSERSITE:'1',IATP_BACKEND_PORT:String(backendPort),IATP_INSTALL_DIR:dataDir(),TIANDITU_API_KEY:configuration.keys.tianditu||'',AMAP_API_KEY:configuration.keys.amap||'',VITE_TIANDITU_API_KEY:'',VITE_AMAP_API_KEY:''};
    // Conda's native DLLs are resolved from the bundled runtime, not a host installation.
    const runtime=path.dirname(python);env.PATH=[runtime,path.join(runtime,'Library','bin'),path.join(runtime,'DLLs'),path.join(process.env.SystemRoot||'C:/Windows','System32')].join(path.delimiter);delete env.PYTHONHOME;delete env.PYTHONPATH;
    env.GDAL_DATA=path.join(runtime,'Library','share','gdal');env.PROJ_LIB=path.join(runtime,'Library','share','proj');env.PROJ_DATA=env.PROJ_LIB;
    const engine=child(path.join(backend,'app.py'),[],env);status('正在载入南京路网与活动数据…');await waitReady(engine,`http://127.0.0.1:${backendPort}/api/health`,'IATP计算服务');
    const work=path.join(dataDir(),'calculation');fs.mkdirSync(work,{recursive:true});
    const bridgeEnv={...env,IATP_SHOWCASE_BACKEND_ROOT:backend,IATP_SHOWCASE_PYTHON:python,IATP_SHOWCASE_WORK_ROOT:work,IATP_SHOWCASE_STATIC_ROOT:webRoot,IATP_SHOWCASE_UPSTREAM:`http://127.0.0.1:${backendPort}`};
    const connector=child(bridge,['--skip-backend-start','--port',String(bridgePort)],bridgeEnv);status('正在连接展示与计算服务…');await waitReady(connector,`http://127.0.0.1:${bridgePort}/api/showcase/health`,'展示服务');bridgeUrl=`http://127.0.0.1:${bridgePort}`;
    if(!selfTest)await window.loadURL(local.url+'/');
  }catch(error){log(error.stack||error);if(selfTest)throw error;await dialog.showMessageBox(window,{type:'error',title:'IATP启动',message:clean(error.message),buttons:['返回配置']});await window.loadURL(local.url+'/settings.html');}
  finally{starting=false;}
}
function menu(){Menu.setApplicationMenu(Menu.buildFromTemplate([
  {label:'文件',submenu:[{label:'地图服务配置',click:()=>window.loadURL(local.url+'/settings.html')},{label:'重新启动计算服务',click:startServices},{type:'separator'},{role:'quit',label:'退出'}]},
  {label:'视图',submenu:[{role:'reload',label:'刷新页面'},{role:'togglefullscreen',label:'全屏',accelerator:'F11'},{role:'resetZoom',label:'实际大小'},{role:'zoomIn',label:'放大'},{role:'zoomOut',label:'缩小'}]},
  {label:'帮助',submenu:[{label:'使用说明',click:()=>dialog.showMessageBox(window,{title:'IATP南京展示',message:'使用说明',detail:'城市预算、活动链和额外活动可直接操作。查询新地点与公共交通路线需联网，并在“文件 → 地图服务配置”中填写自己的天地图与高德服务Key。\n\n可按F11全屏，下方播放键播放完整功能演示。软件关闭后会自动结束自己的计算服务。'})},{label:'查看启动记录',click:()=>shell.openPath(path.join(dataDir(),'startup.log'))}]}
]));}
if(!app.requestSingleInstanceLock())app.quit();else{
  app.on('second-instance',()=>{if(window){if(window.isMinimized())window.restore();window.focus();}});
  app.whenReady().then(async()=>{
    configuration=readConfiguration();local=await startServer({webRoot,getBridge:()=>bridgeUrl});
    if(process.argv.includes('--self-test')){
      const report={started:new Date().toISOString(),checks:[],noBrowserWindow:true};configuration={keys:{}};
      try{await startServices({selfTest:true});const health=await json(local.url+'/api/health'),config=await json(local.url+'/api/showcase/activity-types');
        report.checks.push({name:'packaged backend and bridge',passed:health.success===true},{name:'activity types',passed:config.activityTypes.length===17},{name:'transport modes',passed:config.modes.length===5});
        const budget=await json(local.url+'/data/actual-budget-cells.geojson');report.checks.push({name:'all real budget cells',passed:budget.features.length===7291});
        const request=await fetch(local.url+'/api/showcase/route-chain',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({city:'南京',stops:[{id:'home',label:'家',coordinate:[118.773708,32.050131],mode:'foot'},{id:'work',label:'单位',coordinate:[118.786379,32.041372],mode:'foot'}]})});const result=await request.json();report.checks.push({name:'missing service credentials reported clearly',passed:result.success===false,code:request.status});
        const history=await json(local.url+'/api/showcase/history?kind=route-chain&pageSize=1');report.checks.push({name:'calculation history and database migration',passed:history.success===true&&history.total>=1&&history.items[0].kind==='route-chain'});
        const reviewed=await (await fetch(local.url+'/api/showcase/review-sample',{method:'POST',headers:{'Content-Type':'application/json'},body:'{}'})).json();const detail=await json(local.url+'/api/showcase/history/'+reviewed.historyId);report.checks.push({name:'saved sample review and consistency checks',passed:detail.run?.kind==='result-review'&&detail.run?.validation?.status==='passed'});
        report.passed=report.checks.every(x=>x.passed);
      }catch(error){report.passed=false;report.error=clean(error.message);}
      const target=process.argv.find(x=>x.startsWith('--self-test-report='))?.slice('--self-test-report='.length).trim().replace(/^["']|["']$/g,'')||path.join(dataDir(),'self-test.json');fs.writeFileSync(target,JSON.stringify(report,null,2));await stopServices();await local.close();app.exit(report.passed?0:1);return;
    }
    window=new BrowserWindow({width:1600,height:1000,minWidth:1024,minHeight:700,show:false,title:'IATP · 南京活动出行与碳预算',icon:path.join(__dirname,'app-icon.png'),backgroundColor:'#edf0ed',autoHideMenuBar:false,webPreferences:{preload:path.join(__dirname,'preload.cjs'),contextIsolation:true,nodeIntegration:false,sandbox:true,backgroundThrottling:false}});
    window.webContents.setWindowOpenHandler(()=>({action:'deny'}));window.webContents.on('will-navigate',(event,url)=>{if(!url.startsWith(local.url+'/'))event.preventDefault();});
    ipcMain.handle('configuration:read',event=>settingSender(event)?{tianditu:configuration.keys.tianditu||'',amap:configuration.keys.amap||''}:null);
    ipcMain.handle('configuration:save',async(event,value)=>{if(!settingSender(event))return {success:false};try{const keys={};for(const key of ['tianditu','amap']){const s=String(value?.[key]||'').trim();if(s.length>256||/[\r\n\0]/.test(s))throw new Error('服务Key格式无效。');keys[key]=s;}storeConfiguration(keys);setImmediate(startServices);return {success:true};}catch(error){return {success:false,message:error.message};}});
    ipcMain.handle('application:enter',event=>{if(settingSender(event))setImmediate(startServices);return true;});
    menu();window.once('ready-to-show',()=>{window.maximize();window.show();});
    if(configuration.keys.tianditu||configuration.keys.amap)startServices();else await window.loadURL(local.url+'/settings.html');
  }).catch(async error=>{log(error.stack||error);if(process.argv.includes('--self-test')){await stopServices();local?.server.close();app.exit(1);return;}dialog.showErrorBox('IATP启动失败',clean(error.message));app.quit();});
}
app.on('window-all-closed',()=>app.quit());app.on('before-quit',()=>{if(closing)return;closing=true;for(const p of children)try{p.kill();}catch{}local?.server.close();});
