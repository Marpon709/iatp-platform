const {contextBridge,ipcRenderer}=require('electron');
contextBridge.exposeInMainWorld('IATPDesktop',{
  getConfiguration:()=>ipcRenderer.invoke('configuration:read'),
  saveConfiguration:value=>ipcRenderer.invoke('configuration:save',value),
  enter:()=>ipcRenderer.invoke('application:enter'),
  onStatus:callback=>ipcRenderer.on('application:status',(_event,message)=>callback(message))
});
