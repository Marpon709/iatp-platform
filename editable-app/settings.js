'use strict';
const desktop=window.IATPDesktop;
if(document.getElementById('progress'))desktop.onStatus(message=>document.getElementById('progress').textContent=message);
const form=document.getElementById('configuration');
if(form){desktop.getConfiguration().then(value=>{for(const key of ['tianditu','amap'])document.getElementById(key).value=value?.[key]||'';});
form.addEventListener('submit',async event=>{event.preventDefault();const button=form.querySelector('[type=submit]');button.disabled=true;const result=await desktop.saveConfiguration({tianditu:document.getElementById('tianditu').value,amap:document.getElementById('amap').value});if(!result.success){document.getElementById('feedback').textContent=result.message||'配置未保存';button.disabled=false;}});
document.getElementById('enter').addEventListener('click',()=>desktop.enter());}
