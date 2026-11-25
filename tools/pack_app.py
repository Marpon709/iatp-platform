"""Rebuild the Electron ASAR, including SHA256 integrity records, from editable-app."""
from pathlib import Path
import hashlib,json,struct,shutil
root=Path(__file__).resolve().parents[1]
source=root/'editable-app'
header={'files':{}}
payload=[]
offset=0
for path in sorted(source.rglob('*')):
    if not path.is_file():continue
    data=path.read_bytes();node=header
    parts=path.relative_to(source).parts
    for part in parts[:-1]:node=node['files'].setdefault(part,{'files':{}})
    block=4194304
    node['files'][parts[-1]]={'size':len(data),'offset':str(offset),'integrity':{'algorithm':'SHA256','hash':hashlib.sha256(data).hexdigest(),'blockSize':block,'blocks':[hashlib.sha256(data[i:i+block]).hexdigest() for i in range(0,len(data),block)]}}
    payload.append(data);offset+=len(data)
raw=json.dumps(header,ensure_ascii=False,separators=(',',':')).encode('utf-8')
padding=(-(4+len(raw)))%4
header_pickle=struct.pack('<II',4+len(raw)+padding,len(raw))+raw+b'\0'*padding
out=root/'resources/app.asar.updated'
with out.open('wb') as f:
    f.write(struct.pack('<II',4,len(header_pickle)));f.write(header_pickle)
    for data in payload:f.write(data)
# Verify offsets and integrity before replacing the runnable archive.
with out.open('rb') as f:
    n=struct.unpack('<4I',f.read(16));parsed=json.loads(f.read(n[3]));base=8+n[1]
    def verify(node):
        for entry in node['files'].values():
            if 'files' in entry:verify(entry)
            else:
                f.seek(base+int(entry['offset']));data=f.read(entry['size'])
                assert hashlib.sha256(data).hexdigest()==entry['integrity']['hash']
    verify(parsed)
target=root/'resources/app.asar'
try:
    out.replace(target)
except PermissionError as error:
    raise RuntimeError('请先关闭 IATP，再重新打包。') from error
assert hashlib.sha256(target.read_bytes()).digest()==hashlib.sha256(b''.join([struct.pack('<II',4,len(header_pickle)),header_pickle,*payload])).digest()
print('Rebuilt app.asar; all file offsets and SHA256 hashes verified.')
