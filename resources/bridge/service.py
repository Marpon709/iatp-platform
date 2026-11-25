# -*- coding: utf-8 -*-
"""Local-only showcase bridge to the unmodified IATP calculation software.

No credentials are read by this service. IATP itself loads its own configuration.
"""
import argparse
import ast
import copy
import contextlib
import datetime as dt
import hashlib
import json
import math
import mimetypes
import os
from pathlib import Path
import re
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from activity_space import validate_request, engine_request, present_result
from history_store import HistoryStore
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

mimetypes.add_type('application/geo+json', '.geojson')
mimetypes.add_type('font/ttf', '.ttf')
mimetypes.add_type('application/x-protobuf', '.pbf')

HERE = Path(__file__).resolve().parent
BACKEND = Path(os.environ.get('IATP_SHOWCASE_BACKEND_ROOT') or HERE/'backend').resolve()
PYTHON = Path(os.environ.get('IATP_SHOWCASE_PYTHON') or sys.executable).resolve()
STATIC = Path(os.environ.get('IATP_SHOWCASE_STATIC_ROOT') or HERE/'web').resolve()
WORK = Path(os.environ.get('IATP_SHOWCASE_WORK_ROOT') or HERE/'work').resolve()
RAW = WORK/'raw'
ORIGINAL = (os.environ.get('IATP_SHOWCASE_UPSTREAM') or 'http://127.0.0.1:5000').rstrip('/')
MODEL_LOCK = threading.RLock()
JOB_LOCK = threading.Lock()
JOBS = {}
CACHE = {}
LOCAL_POIS = {}
CACHE_LOCK = threading.Lock()
LOCAL = threading.local()
PERSISTENT_CACHE = WORK/'api-cache'
RECOMMENDATION_CACHE = WORK/'recommendation-cache'
HISTORY = None
HISTORY_LOCK = threading.Lock()
MODEL_FILES = ['ActivityTravelSimulation.py','travel_recommendation.py','daily_activity_range.py','PersonalAccessibility.py','transport_modes.py','carbon_budget.py']


class JobCancelled(Exception):pass


def model_fingerprint():
    paths=list(BACKEND.glob('*.py'))
    paths+=list((BACKEND/'POI/Nanjing').glob('*.geojson'))
    paths+=list((BACKEND/'Network/Nanjing').glob('*'))
    evidence=[(str(p.relative_to(BACKEND)),p.stat().st_size,p.stat().st_mtime_ns) for p in paths if p.is_file()]
    return hashlib.sha256(json.dumps(evidence,sort_keys=True).encode()).hexdigest()[:24]


MODEL_VERSION=model_fingerprint()
UPSTREAM_UNCERTAIN = None


def get_history():
    """数据库记录与计算引擎分开：服务重启保留记录，不自动重跑任务。"""
    global HISTORY
    with HISTORY_LOCK:
        if HISTORY is None:
            store = HistoryStore(WORK, MODEL_VERSION)
            try:store.mark_interrupted()
            except Exception:
                store.close()
                raise
            HISTORY = store
    return HISTORY


def persist_job(jid):
    # 不持有任务锁做数据库 I/O，避免阻塞取消和进度更新。
    with JOB_LOCK:
        snapshot = copy.deepcopy(JOBS.get(jid))
    if snapshot is None:return False
    snapshot.pop('historyWarning',None)
    try:
        get_history().save_snapshot(redact(snapshot))
    except Exception:
        with JOB_LOCK:
            if jid in JOBS:JOBS[jid]['historyWarning']='本次记录暂未保存，可刷新重试；当前计算结果仍可使用。'
        return False
    with JOB_LOCK:
        if jid in JOBS:JOBS[jid].pop('historyWarning',None)
    return True


def request_key(path,payload):
    return hashlib.sha256(json.dumps([MODEL_VERSION,path,payload],sort_keys=True,ensure_ascii=False).encode('utf-8')).hexdigest()


def check_cancel():
    jid=getattr(LOCAL,'job_id',None)
    if not jid:return
    with JOB_LOCK:
        job=JOBS.get(jid) or {}
        if job.get('cancelRequested') or job.get('status') in ['cancelled','timed_out']:raise JobCancelled()
        if time.time()>job.get('deadlineEpoch',math.inf):
            job.update(status='timed_out',cancelRequested=True,error='本次计算超过等待时限；原软件在途计算将自然结束，未用示意结果替代。',stage='计算等待超时',result=job.get('partialResult'),finished_at=now())
            raise JobCancelled()


def update_job(**fields):
    jid=getattr(LOCAL,'job_id',None)
    if jid:
        with JOB_LOCK:
            job=JOBS.get(jid)
            if job:job.update(fields)


def load_cached(path,ttl=3600):
    try:
        value=json.loads(path.read_text(encoding='utf-8'))
        if value.get('modelVersion')==MODEL_VERSION and time.time()-value['cachedEpoch']<ttl:return value
    except (OSError,ValueError,KeyError,TypeError):pass
    return None


def job_snapshot(jid):
    if not jid:return {}
    with JOB_LOCK:
        job=JOBS.get(jid)
        if job:
            current=time.time()
            if job['status']=='running' and current>job.get('deadlineEpoch',math.inf):job.update(status='timed_out',cancelRequested=True,error='本次计算超过等待时限；原软件在途调用仍会自然结束。',stage='计算等待超时',result=job.get('partialResult'),finished_at=now())
            snapshot=copy.deepcopy(job)
        else:snapshot=None
    if snapshot is None:
        return get_history().get_run(jid) or {}
    persist_job(jid)
    snapshot.pop('historyWarning',None)
    with JOB_LOCK:
        if JOBS.get(jid,{}).get('historyWarning'):snapshot['historyWarning']=JOBS[jid]['historyWarning']
    ended=dt.datetime.fromisoformat(snapshot['finished_at']).timestamp() if snapshot.get('finished_at') else current
    snapshot['elapsedSec']=round(ended-snapshot.get('createdEpoch',current),1)
    snapshot['stageElapsedSec']=round(ended-snapshot.get('stageStartedEpoch',current),1)
    snapshot['heartbeatAt']=now()
    snapshot['backendStillRunning']=bool(snapshot.get('backendRequest'))
    return snapshot
MODES = [
    {'id':'foot','label':'步行','apiMode':'foot','rangeMode':'walk'},
    {'id':'bike','label':'骑行','apiMode':'bike','rangeMode':'bike'},
    {'id':'bus','label':'公交','apiMode':'bus','rangeMode':'bus'},
    {'id':'subway','label':'地铁','apiMode':'subway','rangeMode':'metro'},
    {'id':'drive','label':'驾车','apiMode':'drive','rangeMode':'drive'},
]
ALIASES = {'walk':'foot','metro':'subway','car':'drive'}
ALLOWED = {
    'GET':{'/api/health','/api/get_poi_subcategories','/api/get_districts','/api/get_city_info','/api/get_carbon_budget_results','/api/get_carbon_budget_progress','/api/get_carbon_budget_heatmap_image','/api/get_icb_at_point'},
    'POST':{'/api/personal_activity_carbon','/api/travel_activity_poi_candidates','/api/daily_activity_range','/api/travel_chain_routes','/api/travel_itinerary_scheme'},
}


def now(): return dt.datetime.now(dt.timezone.utc).isoformat()


def redact(value):
    if isinstance(value,str):
        return re.sub(r'(?i)([?&](?:tk|key|token|access_token|api_key)=)[^&\s\"\']+',r'\1[REDACTED]',value)
    if isinstance(value,list):return [redact(x) for x in value]
    if isinstance(value,dict):return {k:redact(v) for k,v in value.items()}
    return value


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    encoded=json.dumps(redact(value),ensure_ascii=False,allow_nan=False,separators=(',',':'))
    temp=path.with_suffix(path.suffix+'.'+uuid.uuid4().hex[:8]+'.tmp')
    temp.write_text(encoded,encoding='utf-8');temp.replace(path)


def mode_key(value):
    key=ALIASES.get(str(value or 'foot').lower(),str(value or 'foot').lower())
    if key not in {m['id'] for m in MODES}:raise ValueError('不支持的交通方式；请选择步行、骑行、公交、地铁或驾车。')
    return key


def original(path,payload=None,timeout=180,record=True):
    global UPSTREAM_UNCERTAIN
    check_cancel()
    cache_key=request_key(path,payload);disk=PERSISTENT_CACHE/(cache_key+'.json')
    with CACHE_LOCK:cached=CACHE.get(cache_key)
    if cached and time.time()-cached[0]<3600:
        update_job(cacheHits=(job_snapshot(getattr(LOCAL,'job_id','')).get('cacheHits',0)+1))
        return copy.deepcopy(cached[1])
    persisted=load_cached(disk) if record else None
    if persisted:
        with CACHE_LOCK:CACHE[cache_key]=(persisted['cachedEpoch'],persisted['response'])
        update_job(cacheHits=(job_snapshot(getattr(LOCAL,'job_id','')).get('cacheHits',0)+1))
        return copy.deepcopy(persisted['response'])
    if UPSTREAM_UNCERTAIN and path!='/api/health':raise RuntimeError('上次原软件请求超过网络等待时限，后台是否结束暂无法确认；为避免并发修改其共享计算器，已暂停新计算。请确认或重启原软件后端。')
    # The original engine mutates shared calculator state. Keep each upstream
    # request serialized; do not hold this gate across the entire recommendation.
    while not MODEL_LOCK.acquire(timeout=.25):
        check_cancel();update_job(waitingForBackend=True)
    try:
        check_cancel()
        if UPSTREAM_UNCERTAIN and path!='/api/health':raise RuntimeError('原软件上一个请求是否完成暂不确定，已暂停新计算以避免共享引擎重叠；请确认或重启原软件后端。')
        update_job(waitingForBackend=False,backendRequest=path,backendRequestStartedAt=now(),progressMode='indeterminate')
        with CACHE_LOCK:cached=CACHE.get(cache_key)
        if cached and time.time()-cached[0]<3600:return copy.deepcopy(cached[1])
        body=json.dumps(payload,ensure_ascii=False).encode('utf-8') if payload is not None else None
        request=urllib.request.Request(ORIGINAL+path,data=body,headers={'Content-Type':'application/json','User-Agent':'IATP-LocalShowcase/1.0'})
        try:
            with urllib.request.urlopen(request,timeout=timeout) as response:result=json.load(response)
        except urllib.error.HTTPError as error:
            try:result=json.loads(error.read().decode('utf-8'))
            except Exception:result={'success':False,'message':'原软件服务返回错误','http_status':error.code}
        except Exception as error:
            reason=getattr(error,'reason',error)
            if isinstance(reason,(TimeoutError,socket.timeout)) or 'timed out' in str(reason).lower():UPSTREAM_UNCERTAIN={'at':now(),'endpoint':path}
            raise RuntimeError('原软件计算服务暂不可用：'+redact(str(error)))
        result=redact(result)
        if result.get('success'):
            cached_epoch=time.time()
            with CACHE_LOCK:CACHE[cache_key]=(cached_epoch,result)
            if record:save(disk,{'modelVersion':MODEL_VERSION,'cachedEpoch':cached_epoch,'computed_at':now(),'endpoint':path,'request':payload,'response':result,'kind':'exact-original-api-cache'})
        if record:
            stamp=dt.datetime.now().strftime('%Y%m%d_%H%M%S')+'_'+cache_key[:12]
            save(RAW/(stamp+'.json'),{'endpoint':path,'request':payload,'response':result,'computed_at':now(),'kind':'original-software-api-response'})
        check_cancel()
        return copy.deepcopy(result)
    finally:
        update_job(backendRequest=None,backendRequestStartedAt=None)
        MODEL_LOCK.release()


def activity_config():
    tree=ast.parse((BACKEND/'daily_activity_range.py').read_text(encoding='utf-8-sig'))
    constants={}
    names={'POI_TYPE_DEFS','POI_TYPE_LABELS_ZH','DEFAULT_POI_ACTIVITY_TIMES'}
    for node in tree.body:
        if isinstance(node,ast.Assign):
            for target in node.targets:
                if isinstance(target,ast.Name) and target.id in names:constants[target.id]=ast.literal_eval(node.value)
    types=[]
    dining={'shopping_mall','supermarket','convenience','bookstore','fast_food','hotpot','chinese_cuisine','bbq'}
    leisure={'park','cinema','amusement_park','bar'}
    for key,(category,keywords) in constants['POI_TYPE_DEFS'].items():
        group='shopping_dining' if key in dining else 'recreation' if key in leisure else 'life_services'
        types.append({'id':key,'label':constants['POI_TYPE_LABELS_ZH'][key],'category':category,'keywords':keywords,'defaultStayMin':constants['DEFAULT_POI_ACTIVITY_TIMES'][key],'group':group})
    return {'modes':MODES,'activityTypes':types,'types':types,'groups':[{'id':'shopping_dining','label':'购物餐饮'},{'id':'life_services','label':'生活服务'},{'id':'recreation','label':'休闲娱乐'}],'keywordSearchSupported':True,'keywordActivityType':'keyword','defaultModes':['drive','bus'],'defaultChainUrl':'data/default-chain.json','precomputedChainsUrl':'data/precomputed-chains.json','endpoints':{'routeChain':'/api/showcase/route-chain','searchPois':'/api/showcase/search-pois','recommendExtra':'/api/showcase/recommend-extra','jobs':'/api/showcase/jobs/'},'source':'Original IATP daily_activity_range.py constants and ActivityTravelSimulation transport modes','backend':ORIGINAL,'showcase':None,'notice':'路线与碳排由原软件实时计算；结果可能采用其回退方案，按实际返回来源标注。'}


def coordinates(stop):
    c=stop.get('coordinate')
    if c is None:c=[stop.get('lng',stop.get('lon')),stop.get('lat')]
    if not isinstance(c,list) or len(c)!=2:raise ValueError('地点需要经纬度坐标。')
    lon,lat=map(float,c)
    if not all(math.isfinite(x) for x in [lon,lat]) or not -180<=lon<=180 or not -90<=lat<=90:raise ValueError('地点坐标无效。')
    return [lon,lat]


def actual_mode(label,walk_fallback):
    if walk_fallback:return 'foot'
    for text,key in [('地铁','subway'),('公交','bus'),('骑行','bike'),('骑车','bike'),('驾车','drive'),('开车','drive'),('私家车','drive'),('步行','foot')]:
        if text in str(label or ''):return key
    return None


def normalized_chain(payload):
    city=payload.get('city') or '南京';stops=payload.get('stops') or []
    if len(stops)<2 or len(stops)>12:raise ValueError('请提供2至12个完整活动站点。')
    parsed=[];native=[]
    for index,s in enumerate(stops):
        c=coordinates(s);mode=mode_key(s.get('mode',s.get('transport_mode','foot')))
        p={**s,'id':s.get('id') or 'stop-'+str(index),'label':s.get('label') or s.get('name') or '活动'+str(index+1),'coordinate':c,'mode':mode}
        parsed.append(p);native.append({'label':p['label'],'lng':c[0],'lat':c[1],'mode':mode,'time':s.get('time',s.get('start'))})
    # Original route selection only uses these coordinates and destination mode;
    # labels and activity times do not enter _resolve_leg_plan. Canonical per-OD
    # requests therefore reuse the exact same computation across day scenarios.
    raw_legs=[];raw_warnings=[]
    for index in range(len(parsed)-1):
        origin,target=parsed[index],parsed[index+1]
        raw=original('/api/personal_activity_carbon',{'city':city,'stops':[{'label':'起点','lng':origin['coordinate'][0],'lat':origin['coordinate'][1],'mode':'foot'},{'label':'终点','lng':target['coordinate'][0],'lat':target['coordinate'][1],'mode':target['mode']}]})
        if not raw.get('success'):return raw
        for leg in raw.get('legs') or []:
            leg['from_label']=origin['label'];leg['to_label']=target['label'];raw_legs.append(leg)
        raw_warnings.extend(raw.get('warnings') or [])
    legs=[]
    for index,leg in enumerate(raw_legs):
        requested=parsed[index+1]['mode'];effective=actual_mode(leg.get('mode_label'),leg.get('walk_fallback'))
        warning=''
        if leg.get('route_source')=='straight_line':warning='原软件路线不可用，按直线估算。'
        elif leg.get('walk_fallback'):warning='原软件回退为步行，排放保留其回退估算。'
        elif effective and effective!=requested:warning='原软件为本段采用'+str(leg.get('mode_label') or effective)+'。'
        legs.append({'from':parsed[index]['id'],'to':parsed[index+1]['id'],'fromLabel':leg.get('from_label'),'toLabel':leg.get('to_label'),'mode':requested,'requestedMode':requested,'effectiveMode':effective,'warning':warning,'modeLabel':leg.get('mode_label'),'timeMin':leg.get('time_min'),'distanceKm':leg.get('distance_km'),'distanceM':leg.get('distance_m'),'carbonG':leg.get('carbon_g'),'coordinates':[[float(lon),float(lat)] for lat,lon in leg.get('route_coords') or []],'routeParts':leg.get('route_parts'),'routeSource':leg.get('route_source'),'walkFallback':bool(leg.get('walk_fallback')),'alphaGPerKm':leg.get('alpha_g_per_km')})
    return {'success':True,'city':city,'stops':parsed,'legs':legs,'totalCarbonG':round(sum(float(l['carbonG'] or 0) for l in legs),1),'totalTravelMin':round(sum(float(l['timeMin'] or 0) for l in legs),1),'totalDistanceKm':round(sum(float(l['distanceM'] or 0) for l in legs)/1000,2),'warnings':raw_warnings,'source':{'kind':'actual-software-api','endpoint':'/api/personal_activity_carbon','backend':str(BACKEND/'app.py'),'carbonValuesUnmodified':True,'perLegExactCache':True,'aggregation':'Carbon and time totals sum the original API displayed per-leg values; distance sums original distance_m before conversion. Whole-chain rounding may differ by 0.1 from a direct multi-stop response.'},'computed_at':now(),'notice':'交通方式显示原软件实际返回标签；耗时为模型预估，回退时保留原软件排放估算。'}


def minutes(value):
    if isinstance(value,(int,float)):return float(value)
    hour,minute=map(int,str(value).split(':'));return hour*60+minute


def clock(value):return '%02d:%02d'%(int(value)//60,int(value)%60)


def hav(a,b):
    x1,y1,x2,y2=map(math.radians,[*a,*b]);q=math.sin((y2-y1)/2)**2+math.cos(y1)*math.cos(y2)*math.sin((x2-x1)/2)**2
    return 12742000*math.asin(min(1,math.sqrt(q)))


def local_category_pois(type_definition,origin,destination,max_radius_km):
    category=type_definition['category']
    if category not in LOCAL_POIS:
        path=next((BACKEND/'POI/Nanjing').glob('*_'+category+'.geojson'),None)
        if path is None:raise ValueError('原软件本地POI类别文件不存在：'+category)
        data=json.loads(path.read_text(encoding='utf-8-sig'))
        LOCAL_POIS[category]=(path,data['features'])
    path,features=LOCAL_POIS[category];result=[]
    for feature in features:
        geometry=feature.get('geometry') or {};props=feature.get('properties') or {}
        if geometry.get('type')!='Point':continue
        c=geometry.get('coordinates') or []
        if len(c)<2:continue
        blob=' '.join(str(props.get(k) or '') for k in ['名称','name','类别','category']).lower()
        if not any(keyword.lower() in blob for keyword in type_definition['keywords']):continue
        if min(hav(origin,c),hav(destination,c))>max_radius_km*1000:continue
        result.append({'name':props.get('名称') or props.get('name') or '场所','coordinate':c[:2],'address':props.get('地址') or props.get('address'),'category':category,'sourcePoiId':props.get('PoiID')})
    return result,{'file':str(path),'kind':'original-software-local-poi-data','matchedCategory':category,'keywords':type_definition['keywords'],'radiusKmAroundEitherAnchor':max_radius_km}


def recommend(payload,progress=lambda stage,percent:None):
    with contextlib.nullcontext():
        check_cancel()
        semantic={k:v for k,v in payload.items() if k not in ['async','timeoutSec']};semantic['travelMode']=mode_key(semantic.get('travelMode','foot'))
        exact_key=request_key('recommend-extra-v2',semantic)
        saved_recommendation=load_cached(RECOMMENDATION_CACHE/(exact_key+'.json'))
        if saved_recommendation:
            result=copy.deepcopy(saved_recommendation['response']);result['source']['exactRequestCache']=True;result['source']['cachedComputedAt']=result.get('computed_at');update_job(checkedCandidates=result.get('validatedCandidateCount'),totalCandidates=result.get('validatedCandidateCount'));progress('已读取完全相同条件的真实计算缓存',100);return result
        city=payload.get('city') or '南京'
        if city!='南京':raise ValueError('当前展示连接原软件南京数据，请选择南京。')
        origin=payload.get('from') or {};dest=payload.get('to') or {};a=coordinates(origin);b=coordinates(dest)
        start=origin.get('end') or (payload.get('gap') or {}).get('start') or '17:00';end=dest.get('start') or (payload.get('gap') or {}).get('end') or '18:30'
        t0,t1=minutes(start),minutes(end);gap=t1-t0
        if not 0<gap<=1440:raise ValueError('请提供同一天内结束晚于开始的活动空档。')
        stay=float(payload.get('stayMin',30));mode=mode_key(payload.get('travelMode','foot'));range_mode=next(m['rangeMode'] for m in MODES if m['id']==mode)
        if not math.isfinite(stay) or stay<=0:raise ValueError('活动停留时间必须大于0。')
        type_id=str(payload.get('activityType') or 'park');query=str(payload.get('query') or '').strip()[:100]
        types={t['id']:t for t in activity_config()['activityTypes']}
        if type_id not in types and type_id!='keyword':raise ValueError('请选择原软件活动类别，或使用关键词搜索。')
        if type_id=='keyword' and not query:raise ValueError('请输入想去的活动或地点关键词。')
        nlabel=origin.get('label') or '上一活动';dlabel=dest.get('label') or '下一活动'
        anchors=[{'label':nlabel,'coordinate':a,'start':origin.get('start') or clock(max(0,t0-60)),'end':start},{'label':dlabel,'coordinate':b,'start':end,'end':dest.get('end') or clock(min(1439,t1+60))}]
        progress('正在调用原软件计算固定行程',5)
        baseline=normalized_chain({'city':city,'stops':[{'id':'from',**anchors[0],'mode':mode},{'id':'to',**anchors[1],'mode':mode}]})
        if not baseline.get('success'):return baseline
        gap_details=[];polygons=[];overall=None;local_source=None
        if query:
            progress('正在通过原软件搜索真实场所',15)
            search=original('/api/travel_activity_poi_candidates',{'city':city,'lat':a[1],'lng':a[0],'query':query,'limit':10,'max_distance_km':float(payload.get('maxDistanceKm') or 10)})
            if not search.get('success'):return search
            candidates=[{'name':p.get('name'),'coordinate':[p['lng'],p['lat']],'address':p.get('address'),'category':p.get('poi_category')} for p in search.get('places') or []]
            candidate_source='original_keyword_poi_search'
        elif payload.get('fast',True):
            progress('正在按原软件活动类别检索本地真实POI',15)
            candidates,local_source=local_category_pois(types[type_id],a,b,float(payload.get('maxDistanceKm') or 10))
            candidate_source='original_local_category_pois_then_real_routes'
        else:
            progress('正在调用原软件生成空档活动空间并检索场所',15)
            request={'city':city,'activities':[{'name':x['label'],'lng':x['coordinate'][0],'lat':x['coordinate'][1],'start_time':x['start'],'end_time':x['end']} for x in anchors],'commute_modes':[range_mode],'travel_mode':range_mode,'selected_poi_types':[type_id],'poi_activity_times':{type_id:stay}}
            daily=original('/api/daily_activity_range',request)
            if not daily.get('success'):return daily
            candidates=[{'name':p.get('name'),'coordinate':[p['lng'],p['lat']],'category':p.get('category')} for p in daily.get('poi_markers') or []]
            overall=daily.get('overall_score');gap_details=daily.get('gap_details') or []
            polygons=[[[float(lon),float(lat)] for lat,lon in p['coords']] for p in daily.get('reachable_polygons') or []]
            candidate_source='original_daily_activity_range'
        unique={}
        for c in candidates:unique.setdefault(tuple(round(float(x),6) for x in c['coordinate']),c)
        total_count=len(unique);max_candidates=max(1,min(int(payload.get('maxCandidates') or 6),20));limit=max(1,min(int(payload.get('limit') or 6),20))
        chosen=sorted(unique.values(),key=lambda p:(hav(a,p['coordinate'])+hav(p['coordinate'],b),p.get('name') or ''))[:max_candidates]
        update_job(totalCandidates=len(chosen),checkedCandidates=0,sourceCandidateCount=total_count)
        checked=[];excluded=[]
        for index,c in enumerate(chosen):
            check_cancel()
            progress('正在用原软件逐段校验场所 %d/%d'%(index+1,len(chosen)),35+int(60*(index+1)/max(1,len(chosen))))
            route=normalized_chain({'city':city,'stops':[{'id':'from','label':nlabel,'coordinate':a,'mode':mode},{'id':'extra','label':c.get('name') or '额外活动','coordinate':c['coordinate'],'mode':mode},{'id':'to','label':dlabel,'coordinate':b,'mode':mode}]})
            if not route.get('success') or len(route.get('legs') or [])!=2:
                excluded.append({'name':c.get('name'),'reason':route.get('message') or '路线不可用'});update_job(checkedCandidates=index+1,errors=copy.deepcopy(excluded));continue
            legs=route['legs']
            if any(l['routeSource']=='straight_line' or len(l['coordinates'])<3 for l in legs):
                excluded.append({'name':c.get('name'),'reason':'原软件仅返回直线估算，未作为真实路线推荐'});update_job(checkedCandidates=index+1,errors=copy.deepcopy(excluded));continue
            before=float(legs[0]['timeMin']);after=float(legs[1]['timeMin']);total=round(before+after,1);remaining=round(gap-stay-total,1)
            checked.append({'id':type_id+'-'+hashlib.sha256(json.dumps(c['coordinate']).encode('utf-8')).hexdigest()[:10],'name':c.get('name') or '场所','coordinate':c['coordinate'],'address':c.get('address'),'travelBeforeMin':before,'travelAfterMin':after,'totalTravelMin':total,'stayMin':stay,'remainingMin':remaining,'feasible':remaining>=-1e-8,'score':None,'beforeLeg':legs[0],'afterLeg':legs[1],'routeBeforeCoordinates':legs[0]['coordinates'],'routeAfterCoordinates':legs[1]['coordinates'],'routeSources':[l['routeSource'] for l in legs],'actualModeLabels':[l['modeLabel'] for l in legs],'walkFallback':any(l['walkFallback'] for l in legs),'routeCarbonG':route['totalCarbonG'],'carbonBeforeG':legs[0]['carbonG'],'carbonAfterG':legs[1]['carbonG'],'extraTravelMinComparedWithDirect':round(total-float(baseline['totalTravelMin']),1),'arrivalAtActivityMinutes':t0+before,'departureFromActivityMinutes':t0+before+stay,'arrivalHomeMinutes':t0+total+stay,'warnings':route.get('warnings') or []})
            partial_candidates=sorted([p for p in checked if p['feasible']],key=lambda p:(p['totalTravelMin'],p['name']))[:limit]
            for rank,p in enumerate(partial_candidates,1):p['rank']=rank
            partial={'success':True,'partial':True,'activityType':type_id,'activityLabel':query or types.get(type_id,{}).get('label',type_id),'query':query,'stayMin':stay,'fixedAnchors':anchors,'gap':{'start':start,'end':end,'totalMin':gap},'travelMode':mode,'travelModeLabel':next(m['label'] for m in MODES if m['id']==mode),'candidates':copy.deepcopy(partial_candidates),'selectedCandidateId':partial_candidates[0]['id'] if partial_candidates else None,'baselineCommute':baseline['legs'][0],'overallSpaceScore':overall,'activitySpacePolygons':polygons,'originalPpaCandidateCount':total_count,'validatedCandidateCount':len(checked),'feasibleCandidateCount':len(partial_candidates),'source':{'kind':'actual-software-live-partial-results','candidateSource':candidate_source,'candidateLocalData':local_source},'parameters':payload,'computed_at':now(),'metadata':{'partial':True,'ranking':'只对本次已完成原软件路线校验的可行候选按总出行时间排序；候选尚未全部算完，不是全局最优。','syntheticValues':False},'errors':copy.deepcopy(excluded)}
            update_job(checkedCandidates=index+1,partialResult=partial,progressMode='candidate-count',progress=100*(index+1)/max(1,len(chosen)))
        checked.sort(key=lambda p:(not p['feasible'],p['totalTravelMin'],p['name']))
        for index,c in enumerate(checked,1):c['rank']=index
        feasible=[c for c in checked if c['feasible']];displayed=feasible[:limit]
        baseline_leg=baseline['legs'][0]
        result={'success':True,'activityType':type_id,'activityLabel':types.get(type_id,{}).get('label',query),'query':query,'stayMin':stay,'fixedAnchors':anchors,'gap':{'start':start,'end':end,'totalMin':gap},'travelMode':mode,'travelModeLabel':next(m['label'] for m in MODES if m['id']==mode),'candidates':displayed,'selectedCandidateId':displayed[0]['id'] if displayed else None,'baselineCommute':baseline_leg,'overallSpaceScore':overall,'originalGapDetails':gap_details,'originalPpaCandidateCount':total_count,'validatedCandidateCount':len(checked),'feasibleCandidateCount':len(feasible),'activitySpacePolygons':polygons,'source':{'kind':'actual-software-live-results','candidateEndpoint':'/api/travel_activity_poi_candidates' if query else '/api/daily_activity_range','routeEndpoint':'/api/personal_activity_carbon','candidateSource':candidate_source},'parameters':payload,'computed_at':now(),'message':'已找到时间预算内的真实候选场所' if displayed else '已计算候选中没有满足本次时间预算的场所，请调整停留或出行方式。','metadata':{'ranking':'原软件返回候选；展示层按两固定锚点的直线距离总和预取最多%d处，再逐个调用原软件真实路线校验。先筛选出行+停留不超过空档的候选，再按总出行时间较短排序；这是已校验候选范围内的排序，不是全局最优。'%max_candidates,'score':'原接口不返回地点独立评分，score为null。overallSpaceScore只在原活动空间接口返回时保留，不能冒充个体评分。','searchRating':'原搜索接口rating=4.0为占位值，本展示不使用星级评分。','estimatedTimes':True,'modeFallbacksPreserved':True,'syntheticValues':False,'excludedCandidates':excluded,'infeasibleCheckedCandidates':[{k:v for k,v in c.items() if 'Coordinates' not in k} for c in checked if not c['feasible']]}}
        result['activityLabel']=query or result['activityLabel']
        result['source']['candidateLocalData']=local_source
        if local_source:
            result['source']['candidateEndpoint']=None
            result['metadata']['candidateRetrieval']='原软件本地类别POI检索＋原软件真实路由校验；本次未计算完整PPA活动空间。'
        result['metadata']['queryPriority']='有关键词时使用原关键词检索，未同时限定活动类别。' if query else '按所选原软件活动类别及关键词过滤。'
        result['errors']=excluded
        check_cancel()
        save(WORK/'last-recommendation.json',result);save(RECOMMENDATION_CACHE/(exact_key+'.json'),{'modelVersion':MODEL_VERSION,'cachedEpoch':time.time(),'response':result});progress('原软件计算完成',100)
        return result


def calculate_activity_space(payload, progress):
    types = activity_config()['activityTypes']
    parameters = validate_request(payload, types)
    progress('正在计算路线与时间约束下的可活动范围', 10)
    raw = original('/api/daily_activity_range', engine_request(parameters, types), timeout=540)
    progress('正在整理范围内设施与可达性得分', 90)
    return present_result(raw, parameters, types)


def new_job(payload, kind='recommendation'):
    jid=uuid.uuid4().hex;created=time.time();key=request_key('job:'+kind,{k:v for k,v in payload.items() if k!='async'})
    with JOB_LOCK:
        for prior in JOBS.values():
            if prior.get('requestKey')==key and prior['status']=='running':return {'success':True,'jobId':prior['jobId'],'status':'running','reusedRunningJob':True,'pollUrl':'/api/showcase/jobs/'+prior['jobId']}
        JOBS[jid]={'jobId':jid,'status':'running','stage':'等待原软件计算','progress':None,'progressMode':'indeterminate','created_at':now(),'createdEpoch':created,'stageStartedEpoch':created,'deadlineEpoch':created+max(10,min(float(payload.get('timeoutSec') or 240),600)),'cancelRequested':False,'checkedCandidates':0,'totalCandidates':None,'cacheHits':0,'requestKey':key,'backendRequest':None,'errors':[]}
        # 先保存输入，再启动线程；初次写入失败时不启动无法追踪的计算。
        try:get_history().create_run(jid,kind,key,redact(payload),redact(copy.deepcopy(JOBS[jid])))
        except Exception:
            del JOBS[jid]
            raise
    def run():
        LOCAL.job_id=jid
        def progress(stage,percent):
            check_cancel()
            with JOB_LOCK:
                job=JOBS[jid]
                if job.get('stage')!=stage:job['stageStartedEpoch']=time.time()
                job.update(stage=stage)
                if percent==100:job.update(progress=100,progressMode='completed')
        try:
            result=calculate_activity_space(payload,progress) if kind=='activity-space' else recommend(payload,progress)
            check_cancel()
            if not result.get('success') and not result.get('errors'):result['errors']=[{'message':result.get('message') or '原软件未返回成功结果'}]
            with JOB_LOCK:
                # 取消可能恰好发生在 check_cancel 之后，终态不能再变回完成。
                if JOBS[jid]['status']=='running':JOBS[jid].update(status='completed' if result.get('success') else 'failed',result=result,errors=result.get('errors') or [],stage='计算完成' if result.get('success') else result.get('message','计算未成功'),progress=100,finished_at=now())
        except JobCancelled:
            with JOB_LOCK:
                job=JOBS[jid]
                if job['status']!='timed_out':job.update(status='cancelled',stage='已取消；已完成结果可继续使用')
                job.update(finished_at=now(),result=job.get('partialResult'),backendRequest=None)
        except Exception as error:
            with JOB_LOCK:
                if JOBS[jid]['status']=='running':JOBS[jid].update(status='failed',error=redact(str(error)),errors=[{'message':redact(str(error))}],stage='计算未成功',finished_at=now())
        finally:
            LOCAL.job_id=None
            persist_job(jid)
    threading.Thread(target=run,daemon=True).start()
    return {'success':True,'jobId':jid,'status':'running','pollUrl':'/api/showcase/jobs/'+jid}


def cancel_job(jid):
    with JOB_LOCK:
        job=JOBS.get(jid)
        if not job:return {'success':False,'message':'计算任务不存在。'}
        if job['status']=='running':job.update(cancelRequested=True,status='cancelled',stage='已取消后续计算；原软件在途调用将自然结束',result=job.get('partialResult'),finished_at=now())
    return {'success':True,**job_snapshot(jid)}


def review_sample():
    """复核已保存的路线样例，执行真实入库与查询，不声称重新请求了地图路线。"""
    data=json.loads((HERE/'samples'/'route-chain.json').read_text(encoding='utf-8'))
    key=request_key('result-review',data)
    jid='review-'+key[:24]
    history=get_history()
    if history.get_run(jid):return {'success':True,'historyId':jid,'reused':True}
    snapshot={'jobId':jid,'status':'completed','stage':'已保存样例的结果复核完成',
              'created_at':now(),'createdEpoch':time.time(),'finished_at':now(),
              'result':data,'recordSource':'saved-sample-review'}
    parameters={'来源':'随程序保存的历史路线样例','原计算时间':data.get('computed_at'),
                '处理方式':'读取已保存结果，复核数值与坐标并保存记录；未重新查询地图服务。'}
    try:history.create_run(jid,'result-review',key,parameters,snapshot)
    except ValueError:
        if not history.get_run(jid):raise
    return {'success':True,'historyId':jid,'reused':False}


def recorded_route_chain(payload):
    """同步路线也保存历史，页面切换交通方式后就能查看真实记录。"""
    jid=uuid.uuid4().hex
    snapshot={'jobId':jid,'status':'running','stage':'正在计算行程','created_at':now(),'createdEpoch':time.time()}
    history=get_history()
    history.create_run(jid,'route-chain',request_key('route-chain',payload),redact(payload),snapshot)
    try:
        result=normalized_chain(payload)
    except Exception as error:
        snapshot.update(status='failed',stage='计算未成功',error=redact(str(error)),finished_at=now())
        try:history.save_snapshot(snapshot)
        except Exception:pass
        raise
    snapshot.update(status='completed' if result.get('success') else 'failed',stage='计算完成' if result.get('success') else '计算未成功',result=redact(result),finished_at=now())
    try:history.save_snapshot(snapshot)
    except Exception:
        return {**result,'historyId':jid,'historyWarning':'计算已结束，但本次记录保存失败。'}
    return {**result,'historyId':jid}


class Handler(SimpleHTTPRequestHandler):
    def __init__(self,*args,**kwargs):super().__init__(*args,directory=str(STATIC),**kwargs)
    def log_message(self,*args):pass
    def end_headers(self):
        self.send_header('Access-Control-Allow-Origin','*');self.send_header('Access-Control-Allow-Headers','Content-Type');self.send_header('Access-Control-Allow-Methods','GET,POST,OPTIONS');self.send_header('X-Content-Type-Options','nosniff');super().end_headers()
    def respond(self,value,status=200):
        data=json.dumps(redact(value),ensure_ascii=False,allow_nan=False).encode('utf-8');self.send_response(status);self.send_header('Content-Type','application/json; charset=utf-8');self.send_header('Content-Length',str(len(data)));self.send_header('Cache-Control','no-store');self.end_headers();self.wfile.write(data)
    def do_OPTIONS(self):self.respond({})
    def do_GET(self):
        path=urllib.parse.urlsplit(self.path).path
        try:
            if path=='/api/showcase/health':return self.respond({'success':True,'service':'IATP local showcase bridge','staticRoot':str(STATIC),'originalBackend':ORIGINAL,'modelVersion':MODEL_VERSION,'upstreamUncertain':UPSTREAM_UNCERTAIN})
            if path=='/api/showcase/activity-types':return self.respond(activity_config())
            if path=='/api/showcase/history':
                query=urllib.parse.parse_qs(urllib.parse.urlsplit(self.path).query)
                page=int(query.get('page',['1'])[0]);size=int(query.get('pageSize',['10'])[0])
                kind=query.get('kind',[None])[0] or None
                if kind is not None and kind not in {'route-chain','recommendation','activity-space','result-review'}:raise ValueError('不支持的计算类型。')
                records=get_history().list_runs(page=page,page_size=size,status=query.get('status',[None])[0] or None,kind=kind)
                return self.respond({'success':True,**records})
            if path.startswith('/api/showcase/history/'):
                record=get_history().get_run(path.rsplit('/',1)[-1])
                return self.respond({'success':True,'run':record} if record else {'success':False,'message':'计算记录不存在。'},200 if record else 404)
            if path.startswith('/api/showcase/jobs/'):
                jid=path.rsplit('/',1)[-1]
                job=job_snapshot(jid)
                return self.respond(job or {'success':False,'message':'计算任务不存在。'},200 if job else 404)
            if path in ALLOWED['GET']:return self.respond(original(self.path,timeout=30,record=False))
            if path.startswith('/api/'):return self.respond({'success':False,'message':'该接口未开放。'},404)
            return super().do_GET()
        except (ValueError,TypeError) as error:return self.respond({'success':False,'message':redact(str(error))},400)
        except Exception as error:return self.respond({'success':False,'message':redact(str(error))},502)
    def do_POST(self):
        try:
            length=int(self.headers.get('Content-Length') or 0)
            if not 0<length<=8*1024*1024:raise ValueError('请求内容为空或超过大小限制。')
            payload=json.loads(self.rfile.read(length).decode('utf-8'));path=urllib.parse.urlsplit(self.path).path
            if not isinstance(payload,dict):raise ValueError('请求内容须为 JSON 对象。')
            if path=='/api/showcase/review-sample':return self.respond(review_sample())
            if path.startswith('/api/showcase/jobs/') and path.endswith('/cancel'):
                return self.respond(cancel_job(path.split('/')[-2]))
            if path=='/api/showcase/route-chain':return self.respond(recorded_route_chain(payload))
            if path=='/api/showcase/activity-space':
                validate_request(payload, activity_config()['activityTypes'])
                return self.respond(new_job(payload, kind='activity-space'), 202)
            if path=='/api/showcase/search-pois':
                result=original('/api/travel_activity_poi_candidates',payload)
                # Original rating is a hardcoded placeholder, not consumer data.
                for place in result.get('places') or []:place.pop('rating',None)
                return self.respond(result)
            if path=='/api/showcase/recommend-extra':
                if payload.get('async'):return self.respond(new_job(payload),202)
                return self.respond(recommend(payload))
            if path in ALLOWED['POST']:
                if path=='/api/personal_activity_carbon':
                    for stop in payload.get('stops') or []:stop['mode']=mode_key(stop.get('mode',stop.get('transport_mode','foot')))
                return self.respond(original(path,payload))
            return self.respond({'success':False,'message':'该接口未开放。'},404)
        except (ValueError,KeyError,TypeError) as error:return self.respond({'success':False,'message':redact(str(error))},400)
        except Exception as error:return self.respond({'success':False,'message':redact(str(error))},502)


def ensure_backend():
    try:
        req=urllib.request.Request(ORIGINAL+'/api/health')
        with urllib.request.urlopen(req,timeout=4) as response:health=json.load(response)
        if health.get('success') and health.get('status')=='running':return
        raise RuntimeError('计算服务地址已有其他服务，请先确认原软件后端。')
    except urllib.error.URLError:
        pass
    env=dict(os.environ);env.update(PYTHONUTF8='1',PYTHONUNBUFFERED='1',PYTHONDONTWRITEBYTECODE='1')
    p=subprocess.Popen([str(PYTHON),str(BACKEND/'app.py')],cwd=str(BACKEND),env=env,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    (WORK/'backend.pid').write_text(str(p.pid),encoding='utf-8')
    for _ in range(90):
        if p.poll() is not None:raise RuntimeError('原软件启动未成功；未保留可能含密钥的第三方日志。')
        try:
            with urllib.request.urlopen(ORIGINAL+'/api/health',timeout=3) as response:
                if json.load(response).get('success'):return
        except Exception:pass
        time.sleep(1)
    raise RuntimeError('原软件启动超时。')


def main():
    global STATIC
    parser=argparse.ArgumentParser();parser.add_argument('--port',type=int,default=8890);parser.add_argument('--static-root',type=Path,default=STATIC);parser.add_argument('--skip-backend-start',action='store_true');args=parser.parse_args();STATIC=args.static_root.resolve()
    if not 0<=args.port<=65535:parser.error('--port must be between 0 and 65535')
    RAW.mkdir(parents=True,exist_ok=True)
    get_history()
    if not args.skip_backend_start:ensure_backend()
    # Packaged web resources may be read-only. Activity constants remain available
    # through /api/showcase/activity-types without mutating bundled live-config.
    server=ThreadingHTTPServer(('127.0.0.1',args.port),Handler)
    (WORK/'service.pid').write_text(str(os.getpid()),encoding='utf-8')
    actual_port=server.server_address[1]
    print(json.dumps({'event':'ready','service':'IATP local showcase bridge','host':'127.0.0.1','port':actual_port,'url':'http://127.0.0.1:%d/'%actual_port,'upstream':ORIGINAL,'backendStartSkipped':args.skip_backend_start}),flush=True)
    server.serve_forever()


if __name__=='__main__':main()
