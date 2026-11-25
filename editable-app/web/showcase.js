'use strict';
(() => {
  const $=id=>document.getElementById(id);
  const esc=v=>String(v??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const num=(v,d=1)=>Number(v||0).toLocaleString('zh-CN',{maximumFractionDigits:d});
  const fmt=n=>num(n,0),fmtBudget=n=>Number(n).toLocaleString('zh-CN',{minimumFractionDigits:2,maximumFractionDigits:2});
  const reduced=matchMedia('(prefers-reduced-motion: reduce)').matches;
  const MODE={walk:{label:'步行',api:'foot',color:'#14a879'},bike:{label:'骑行',api:'bike',color:'#087eff'},bus:{label:'公交',api:'bus',color:'#ec8b10'},metro:{label:'地铁',api:'subway',color:'#8460ed'},drive:{label:'驾车',api:'drive',color:'#f15a42'}};
  const paths={walk:'<circle cx="13" cy="4" r="2"/><path d="m9 21 3-7 2 3 3 4M9 13l2-6 4 3 3 1M12 14l-1-5-4 3"/>',bike:'<circle cx="6" cy="17" r="4"/><circle cx="19" cy="17" r="4"/><path d="m6 17 5-9 4 9H6l-2-7h3m4-2h6l-1-3h3m-4 12 4-9"/>',bus:'<rect x="4" y="3" width="16" height="17" rx="3"/><path d="M4 12h16M7 20v2m10-2v2M8 16h1m6 0h1M8 6h8"/>',metro:'<rect x="5" y="2" width="14" height="18" rx="5"/><path d="M5 11h14M8 16h1m6 0h1M8 23l2-3m6 3-2-3M10 5h4"/>',drive:'<path d="m4 10 2-6h12l2 6M3 10h18v9H3zM6 19v3m12-3v3M6 14h2m8 0h2M3 10H1m20 0h2"/>',search:'<circle cx="10" cy="10" r="6"/><path d="m15 15 6 6"/>',plus:'<path d="M12 4v16M4 12h16"/>',arrow:'<path d="M4 12h16m-6-6 6 6-6 6"/>',layers:'<path d="m12 3 10 6-10 6L2 9zM3 14l9 6 9-6"/>',locate:'<circle cx="12" cy="12" r="6"/><path d="M12 1v5m0 12v5M1 12h5m12 0h5"/>',view:'<path d="m3 8 9-5 9 5v10l-9 5-9-5zM3 8l9 6 9-6M12 14v9"/>',grid:'<path d="M3 3h18v18H3zM3 9h18M3 15h18M9 3v18m6-18v18"/>',check:'<path d="m5 12 4 4L20 5"/>',close:'<path d="m5 5 14 14M19 5 5 19"/>'};
  function icon(name,cls=''){return `<svg class="icon ${cls}" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.65" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">${paths[name]||paths.locate}</svg>`;}
  const placeName=value=>String(value??'').split(' - ')[0].trim();
  const normalize=m=>({foot:'walk',subway:'metro',car:'drive'}[m]||m||'walk');
  const STORY={weightStart:10,distributionStart:12.5,distributionComplete:16.5,allocationEnd:18,changeMode:22,chainEnd:27,resultsAt:30,insertAt:33,recordsAt:38,total:50};
  const state={scene:'convenience',time:0,playing:false,flat:true,budgetLayer:false,budgetView:'district',comparePoint:'a',selection:null,activeLeg:0,modes:['drive','bus'],motion:!reduced,recType:'park',recQuery:'',recSearch:false,stayMin:30,gap:1,recMode:'walk',recEditing:true,recommended:false,added:false,candidateId:null,recBusy:false,routeBusy:false,allocationStage:0};
  let contributionStory,allocationMapAnchors,allocationProcess,actualBudget,actualBudgetCells,config,day,initialDay,cachedChains,seed,recResult,map,ready=false,cityContext=false,cityLoading=false,actualCellLookup;
  let lastPanelKey='',lastMapKey='',routeVersion=0,selectionVersion=0,comparison=null,notice='',requestId=0,searchId=0,searchTimer,searchResults=[];
  let jobSnapshot=null,jobStartedAt=0,jobClockTimer,allocationStarted=performance.now(),allocationAuto=false,allocationMarker,routeController,jobTimer,jobId=null,recVersion=0,frameId=0,lastFrame=0,motionFrame=0,lastFlow=0,animationStart=0,storyEvents=new Set();
  let budgetPopup, convenience;
  let nodeMarkers=[],placeMarkers=[],routeMarkers=[],carbonMarkers=[],candidateMarkers=[],extraMarker;
  const tweens=new Map(),cityBudgetColors=new Map();
  const cityLayerIds=['city-green','city-water','city-roads','city-buildings','city-place-names'];
  const lineCache=new WeakMap(),vehiclePathCache=new WeakMap();
  function fixedChain(){return day;}
  function displayDay(result){return {...result,legs:result.legs.map(l=>({...l,from:l.fromLabel||l.from,to:l.toLabel||l.to}))};}
  function activity(){return recResult||seed;}
  function candidate(){return activity()?.candidates?.find(c=>c.id===state.candidateId)||activity()?.candidates?.[0]||null;}
  function clock(v){const n=Math.round(v);return `${String(Math.floor(n/60)).padStart(2,'0')}:${String(n%60).padStart(2,'0')}`;}
  function modeInfo(leg){let key=normalize(leg?.effectiveMode||leg?.mode||'walk');if(leg?.walkFallback)key='walk';return {...(MODE[key]||MODE.walk),key,label:leg?.walkFallback?'步行接驳':leg?.modeLabel||(MODE[key]||MODE.walk).label};}
  function routeLegs(){
    if(state.scene==='recommendation'&&state.added&&candidate()){
      const c=candidate(),a=activity(),from=gapAnchors().from.label,to=gapAnchors().to.label;
      const fallback=(before)=>({from:before?from:c.name,to:before?c.name:to,mode:a.travelMode||state.recMode,modeLabel:MODE[normalize(a.travelMode||state.recMode)]?.label,timeMin:before?c.travelBeforeMin:c.travelAfterMin,carbonG:before?c.beforeCarbonG:c.afterCarbonG,coordinates:before?c.routeBeforeCoordinates:c.routeAfterCoordinates});
      return [c.beforeLeg||fallback(true),c.afterLeg||fallback(false)].map((l,i)=>({...l,from:i===0?from:c.name,to:i===0?c.name:to}));
    }
    return day?.legs||[];
  }
  function configureStoryRecommendation(){
    state.recType=seed?.activityType||'keyword';state.recQuery=seed?.query||seed?.activityLabel||'咖啡';state.recSearch=state.recType==='keyword';state.recMode=normalize(seed?.travelMode||'bus');state.stayMin=seed?.stayMin||20;state.gap=1;
  }
  const preferenceActivityName=id=>({medical:'就医',work:'工作',education:'学习',transportation_station:'换乘',shopping:'购物',catering:'用餐',recreation:'休闲'}[id]||'日常');
  function allocationHTML(){
    const a=allocationProcess.aggregate;
    return `<div class="chapter-kicker">城市总量 · 个人分配</div><h2 class="allocation-process-title">碳预算如何分配</h2><div class="allocation-explanation">将城市每日出行碳预算，按地区人口与居住地服务可达性分配到个人。</div><div class="budget-basis">分配依据采用全市居住网格的服务可达性。沿途可达性得分用于一次行程的活动空间评估。</div><ol class="allocation-reasoning"><li>居住地获取日常服务越困难，给予的分配权重越高。</li><li>结合各地区人口，分配城市预算总量，折算为人均日预算。</li></ol><div class="personal-budget-result"><span>示例居住地人均日碳预算</span><strong>${numberHTML('allocation-budget',a.icbG,2)}<small>g / 人·天</small></strong></div><div class="allocation-pool"><span>城市每日出行碳预算池</span><strong>${num(a.dailyPoolG/1000000,2)} <small>吨 / 天</small></strong></div><details class="allocation-method"><summary>查看分配公式与参数</summary><p>分配权重 q = 1 ÷ 居住地综合服务可达性 A</p><p>人均日碳预算 = 城市每日预算池 × 本地区权重 ÷ 全市各地区“人口 × 权重”的总和。</p><p>示例网格 A = ${num(a.A,2)}，q = ${num(a.inverseWeight,6)}。</p><p>当前预算情景采用年输入量3000万吨、10%分配系数和365天折算。</p></details><div class="allocation-process-foot"><button type="button" id="allocation-replay">↺ 重播预算分配</button></div>`;
  }
  function setAllocationStage(stage,renderNow=true){
    const next=Math.max(0,Math.min(2,stage));if(state.allocationStage===next){updateAllocationRibbon();return;}
    state.allocationStage=next;resetAllocationMapAnimation(next);const a=allocationProcess.aggregate;
    if(next===0)tween('allocation-a',a.A,0,1500,true);
    if(next===1)tween('allocation-q',a.inverseWeight,0,1500,true);
    if(next===2)tween('allocation-budget',a.icbG,0,1800,true);
    if(renderNow){renderPanel(true);updateCaption();}updateAllocationRibbon();syncAllocationMarker();
  }
  function updateAllocationRibbon(){const el=$('allocation-ribbon');if(el)el.hidden=true;syncAllocationMap();}
  function syncAllocationMarker(){if(allocationMarker)allocationMarker.getElement().hidden=true;syncAllocationMap();}
  function updateAllocationAnimation(now){
    if(state.scene==='allocation'&&allocationAuto&&!state.playing&&!document.hidden){const elapsed=(now-allocationStarted)/1000;setAllocationStage(elapsed<STORY.weightStart?0:elapsed<STORY.distributionStart?1:2);if(elapsed>=STORY.allocationEnd)completeAllocationMap();}
    animateAllocationMap(now);
  }
  function recommendationCarbonHTML(c,result){
    if(!Number.isFinite(c.routeCarbonG))return '';const baseline=result.baselineCommute?.carbonG??result.baselineCommute?.carbon_g,delta=Number.isFinite(baseline)?c.routeCarbonG-baseline:null;
    return `<div class="recommendation-carbon"><span>本段出行排放</span><strong>${num(c.routeCarbonG)} <small>g CO₂</small></strong></div>${delta!==null?`<div class="recommendation-carbon-change"><span>直接前往 ${num(baseline)} g → 加入活动 ${num(c.routeCarbonG)} g</span><b class="${delta<=0?'down':'up'}">${delta<=0?'↓':'↑'} ${num(Math.abs(delta))} g</b></div>`:''}`;
  }
  function jobProgressHTML(){
    return `<section class="job-progress live-job-progress" role="status"><div class="job-title-row"><span class="loading-orbit"></span><strong id="job-stage">正在准备行程条件…</strong><b id="job-elapsed">0 秒</b></div><div class="job-progress-track indeterminate" id="job-progress-track"><span id="job-bar"></span></div><div class="job-detail-row"><span id="job-detail">正在查找场所与通行路线。</span><b id="job-percent"></b></div><p id="job-wait-note"></p><div id="job-partial-summary"></div><div class="job-actions"><button type="button" id="cancel-recommendation">取消计算</button><button type="button" id="use-partial-result" hidden>先看已完成结果</button></div></section>`;
  }
  function paintJobProgress(){
    if(!state.recBusy||!$('job-stage'))return;const j=jobSnapshot||{},elapsed=Number.isFinite(j.elapsedSec)?Math.max(j.elapsedSec,(performance.now()-jobStartedAt)/1000):(performance.now()-jobStartedAt)/1000;
    $('job-stage').textContent=j.stage||'正在连接计算服务…';$('job-elapsed').textContent=Math.floor(elapsed)+' 秒';
    const checked=Number(j.checkedCandidates||0),total=Number(j.totalCandidates||0),partial=j.partialResult,count=partial?.candidates?.length||0;
    const track=$('job-progress-track'),known=Number.isFinite(j.progress)&&j.progressMode!=='indeterminate';track.classList.toggle('indeterminate',!known);
    $('job-bar').style.width=known?Math.max(0,Math.min(100,j.progress))+'%':'';$('job-percent').textContent=total?checked+' / '+total+' 场所':'';
    $('job-detail').textContent=j.waitingForBackend?'等待上一条路线请求结束':total?'逐段校验通行、停留与到达时限':'正在查找符合条件的场所';
    $('job-wait-note').textContent=elapsed>=20?(count?'已有可行地点，可先查看已完成结果。':'首次公共交通查询可能较慢；计算仍在进行，可随时取消。'):'';
    const summary=$('job-partial-summary');if(count){summary.innerHTML=`<span>已验证 ${count} 个可行地点</span><strong>${esc(placeName(partial.candidates[0].name))}</strong><small>${num(partial.candidates[0].totalTravelMin)} min 通行 · ${num(partial.candidates[0].routeCarbonG)} g CO₂</small>`;}else summary.innerHTML='';
    $('use-partial-result').hidden=!count;
  }
  function cancelRemoteJob(id=jobId){if(!id)return;jsonRequest('/api/showcase/jobs/'+encodeURIComponent(id)+'/cancel',{}).catch(()=>{});}
  async function cancelRecommendation(){
    const id=jobId,version=++recVersion;jobId=null;clearTimeout(jobTimer);clearInterval(jobClockTimer);state.recBusy=false;jobSnapshot=null;notice='已取消本次计算';renderPanel(true);
    if(id)try{const r=await jsonRequest('/api/showcase/jobs/'+encodeURIComponent(id)+'/cancel',{});if(version===recVersion&&!state.recBusy&&r.backendStillRunning){notice='已取消等待，当前地图请求正在结束。';renderPanel(true);}}catch{}
  }
  function usePartialResult(){
    const result=jobSnapshot?.partialResult;if(!result?.candidates?.length)return;const partial=JSON.parse(JSON.stringify(result));partial.metadata={...(partial.metadata||{}),selectedBeforeAllCandidatesComplete:true};cancelRemoteJob();recVersion++;clearTimeout(jobTimer);clearInterval(jobClockTimer);finishRecommendation(partial,recVersion);
  }

  const allocationCategoryArt={
    medical:{color:'#e58b83',path:'<path d="M9 3h6v6h6v6h-6v6H9v-6H3V9h6z"/>'},
    work:{color:'#628ab3',path:'<rect x="3" y="7" width="18" height="14" rx="2"/><path d="M8 7V3h8v4M3 12c6 4 12 4 18 0M12 12v4"/>'},
    education:{color:'#a28bc2',path:'<path d="m2 8 10-5 10 5-10 5zM6 10v7c4 3 8 3 12 0v-7M22 8v9"/>'},
    transportation_station:{color:'#ddaf5b',path:paths.bus},
    shopping:{color:'#77a7a0',path:'<path d="M4 8h16l1 13H3zM8 8V6a4 4 0 0 1 8 0v2"/>'},
    catering:{color:'#d5a07b',path:'<path d="M4 3v7c0 3 5 3 5 0V3M6.5 3v19M18 3c-4 4-4 8 0 9v10M18 3v9"/>'},
    recreation:{color:'#8eae79',path:'<path d="M12 2 4 13h5l-5 5h16l-5-5h5zM12 18v5"/>'}
  };
  let allocationMapReady=false,allocationMapScene=false,allocationVisualStage=-1,allocationVisualClock=0,allocationVisualStart=0,allocationVisualRunning=false,allocationCameraToken='',allocationLocalCamera=false,lastAllocationPaint=0,lastAllocationParticleFrame=0,lastAllocationLabelLayout=0,lastAllocationFrameSignature='';
  let allocationHomeNode,allocationPoiNodes=[],allocationAssociationLines=[],allocationRevealComplete=false,allocationRowsPrefix=[];
  const allocationStageTimes=[0,STORY.weightStart,STORY.distributionStart],allocationMapLayerIds=['allocation-association-halo','allocation-association-lines','allocation-weight-glow','allocation-weight-ring','allocation-selected-cell-outline','allocation-scan-line','allocation-particles-glow','allocation-particles'];
  function categoryIcon(category){return `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">${allocationCategoryArt[category]?.path||paths.locate}</svg>`;}
  function calculationArc(from,to,bend=.13){
    const dx=to[0]-from[0],dy=to[1]-from[1],control=[(from[0]+to[0])/2-dy*bend,(from[1]+to[1])/2+dx*bend],points=[];
    for(let i=0;i<=40;i++){const t=i/40,s=1-t;points.push([s*s*from[0]+2*s*t*control[0]+t*t*to[0],s*s*from[1]+2*s*t*control[1]+t*t*to[1]]);}return points;
  }
  function allocationMapClock(){return Math.min(STORY.allocationEnd,state.time);}
  function pauseAllocationMap(){
    if(!allocationMapReady)return;allocationVisualClock=allocationMapClock();allocationVisualRunning=false;setAllocationMotionClass(false);
  }
  function seekAllocationMap(time){
    allocationVisualClock=Math.max(0,Math.min(STORY.allocationEnd,time));allocationVisualStart=performance.now();allocationVisualRunning=false;allocationAuto=false;
    if(allocationMapReady&&ready&&state.scene==='allocation'){focusAllocationCamera(state.allocationStage,0);lastAllocationFrameSignature='';paintAllocationMap(reduced?1:allocationStageProgress(allocationVisualClock),performance.now());setAllocationMotionClass(false);requestAnimationFrame(positionAllocationLabels);}
  }
  function resumeAllocationMap(){if(state.scene==='allocation')setAllocationMotionClass(!reduced&&state.motion);}
  function allocationStageProgress(clock){const stage=state.allocationStage;if(stage===0)return Math.max(0,Math.min(1,clock/(STORY.weightStart-1)));if(stage===1)return Math.max(0,Math.min(1,(clock-STORY.weightStart)/(STORY.distributionStart-STORY.weightStart)));return Math.max(0,Math.min(1,(clock-STORY.distributionStart-.6)/(STORY.distributionComplete-STORY.distributionStart-.6)));}
  function completeAllocationMap(){
    if(allocationRevealComplete&&!allocationAuto&&!allocationVisualRunning&&allocationVisualClock===STORY.allocationEnd)return;
    allocationVisualClock=STORY.allocationEnd;allocationVisualStart=performance.now();allocationVisualRunning=false;allocationAuto=false;allocationRevealComplete=true;
    if(allocationMapReady&&ready){lastAllocationFrameSignature='';paintAllocationMap(1,performance.now(),true);map.getSource('allocation-particle-data').setData(featureCollection([]));setAllocationMotionClass(false);}
  }
  function resetAllocationMapAnimation(stage=state.allocationStage){
    allocationVisualStage=-1;allocationVisualClock=allocationStageTimes[stage];allocationVisualStart=performance.now();allocationVisualRunning=false;allocationLocalCamera=false;allocationCameraToken='';lastAllocationPaint=0;lastAllocationParticleFrame=0;lastAllocationFrameSignature='';allocationRevealComplete=false;resetResidentStory();
    if(allocationMapReady){map.getSource('allocation-particle-data').setData(featureCollection([]));syncAllocationMap(true);}
  }
  function setAllocationMotionClass(running){
    [...allocationPoiNodes.map(x=>x.marker),allocationHomeNode,allocationResidentMarker].filter(Boolean).forEach(m=>m.getElement().classList.toggle('calculation-paused',!running));
  }
  function setupAllocationMap(){
    const anchors=allocationMapAnchors,home=anchors.home.coordinate;
    allocationAssociationLines=anchors.poiAnchors.map((p,i)=>({coordinates:calculationArc(p.coordinate,home,(i%2?1:-1)*.18),color:allocationCategoryArt[p.category].color,contribution:p.weightedContribution,id:p.id}));
    allocationAssociationLines.forEach(line=>line.outwardCoordinates=line.coordinates.slice().reverse());
    map.addSource('allocation-association-data',{type:'geojson',data:featureCollection(allocationAssociationLines.map((p,i)=>({type:'Feature',geometry:{type:'LineString',coordinates:p.coordinates},properties:{color:p.color,activity_index:i}})))});
    map.addSource('allocation-weight-data',{type:'geojson',data:featureCollection([{type:'Feature',geometry:{type:'Point',coordinates:home},properties:{q:anchors.home.q,A:anchors.home.A}}])});
    map.addSource('allocation-particle-data',{type:'geojson',data:featureCollection([])});
    map.addSource('allocation-selected-cell-data',{type:'geojson',data:featureCollection(actualBudgetCells.features.filter(f=>f.properties.row===anchors.home.row&&f.properties.col===anchors.home.col))});
    const layout={'line-cap':'round','line-join':'round'},hidden={visibility:'none'},weightRadius=34;
    map.addLayer({id:'allocation-association-halo',type:'line',source:'allocation-association-data',layout:{...layout,...hidden},paint:{'line-color':['get','color'],'line-width':9,'line-blur':5,'line-opacity':.17}});
    map.addLayer({id:'allocation-association-lines',type:'line',source:'allocation-association-data',layout:{...layout,...hidden},paint:{'line-color':['get','color'],'line-width':1.6,'line-opacity':.72,'line-dasharray':[2,2]}});
    map.addLayer({id:'allocation-weight-glow',type:'circle',source:'allocation-weight-data',layout:hidden,paint:{'circle-color':'#8db9a1','circle-radius':weightRadius,'circle-opacity':.11,'circle-blur':.35}});
    map.addLayer({id:'allocation-weight-ring',type:'circle',source:'allocation-weight-data',layout:hidden,paint:{'circle-color':'#ffffff','circle-opacity':.30,'circle-radius':weightRadius,'circle-stroke-color':'#8ab69d','circle-stroke-width':1.4,'circle-stroke-opacity':.8}});
    map.addLayer({id:'allocation-selected-cell-outline',type:'line',source:'allocation-selected-cell-data',layout:hidden,paint:{'line-color':'#ffffff','line-width':2.2,'line-opacity':.9}});
    map.addLayer({id:'allocation-scan-line',type:'line',source:'budget',layout:hidden,filter:['==',['get','row'],-1],paint:{'line-color':'#ffffff','line-width':1.2,'line-opacity':.75}});
    map.addLayer({id:'allocation-particles-glow',type:'circle',source:'allocation-particle-data',layout:hidden,paint:{'circle-color':['get','color'],'circle-radius':7,'circle-opacity':.23,'circle-blur':.7}});
    map.addLayer({id:'allocation-particles',type:'circle',source:'allocation-particle-data',layout:hidden,paint:{'circle-color':['get','color'],'circle-radius':['get','radius'],'circle-opacity':.94,'circle-stroke-color':'#ffffff','circle-stroke-width':.8}});
    for(const p of anchors.poiAnchors){
      const el=document.createElement('div');el.className='map-activity-node';el.style.setProperty('--activity-color',allocationCategoryArt[p.category].color);el.title=p.label+' · '+p.name;
      el.innerHTML=`<span class="activity-node-dot">${categoryIcon(p.category)}</span><span class="activity-node-copy"><b>${esc(p.label)}</b><small title="您对${esc(preferenceActivityName(p.category))}活动的重视程度">活动重视程度 ${num(p.activityWeight*100)}%</small><strong>${num(p.weightedContribution,2)}</strong></span>`;
      const marker=new maplibregl.Marker({element:el,anchor:'top-left',offset:[12,-24]}).setLngLat(p.coordinate).addTo(map);allocationPoiNodes.push({marker,data:p});
    }
    const homeEl=document.createElement('div');homeEl.className='map-calculation-home';
    allocationHomeNode=new maplibregl.Marker({element:homeEl,anchor:'top-left',offset:[-78,24]}).setLngLat(home).addTo(map);
    const counts=Array(actualBudget.height).fill(0);actualBudgetCells.features.forEach(f=>counts[f.properties.row]++);let sum=0;allocationRowsPrefix=counts.map(count=>sum+=count);
    setupResidentStory();allocationMapReady=true;syncAllocationMap(true);
  }
  function setAllocationLayerVisibility(id,show){if(map.getLayer(id))map.setLayoutProperty(id,'visibility',show?'visible':'none');}
  function syncAllocationMap(force=false){
    if(!allocationMapReady||!ready)return;
    const visible=state.scene==='allocation',stage=state.allocationStage,newStage=allocationVisualStage!==stage||allocationMapScene!==visible;
    allocationMapScene=visible;
    if(newStage){allocationVisualStage=stage;allocationVisualClock=allocationStageTimes[stage];allocationVisualStart=performance.now();allocationVisualRunning=false;allocationCameraToken='';allocationLocalCamera=false;lastAllocationPaint=0;map.getSource('allocation-particle-data').setData(featureCollection([]));}
    if(!force&&!newStage)return;
    for(const id of allocationMapLayerIds)setAllocationLayerVisibility(id,false);
    for(const node of allocationPoiNodes)node.marker.getElement().hidden=true;
    allocationHomeNode.getElement().hidden=true;
    const key=$('allocation-map-key');if(key){key.hidden=!visible;key.innerHTML=stage===0?'<i class="key-computation-line"></i><span>七类生活需求</span>':stage===1?'<span>生活圈 → 预算权重</span>':'<span>南京 · 个人日碳预算</span>';}
    const homeEl=allocationHomeNode.getElement();homeEl.dataset.stage=String(stage);
    if(stage===0)homeEl.innerHTML='<i class="calculation-home-rings"></i><div><span>我的生活圈</span><strong id="map-home-a">0.00</strong><small>居住地综合可达性指数 A</small></div>';
    if(stage===1)homeEl.innerHTML=`<i class="calculation-home-rings"></i><div><span>我的预算权重</span><div class="home-weight-equation"><small>1 / ${num(allocationMapAnchors.home.A,2)}</small><i>↓</i><strong id="map-home-q">${num(allocationMapAnchors.home.q,6)}</strong></div></div>`;
    $('legend').hidden=visible&&stage<2;
    if(!visible){hideResidentStory();setAllocationMotionClass(false);return;}
    const active=stage===0?['allocation-association-halo','allocation-association-lines']:stage===1?['allocation-weight-glow','allocation-weight-ring','allocation-selected-cell-outline']:['allocation-scan-line'];
    [...active,'allocation-particles-glow','allocation-particles'].forEach(id=>setAllocationLayerVisibility(id,true));
    setAllocationMotionClass(allocationVisualRunning||state.playing||allocationAuto);
    if(stage!==0)focusAllocationCamera(stage,force&&state.budgetView==='district'?0:1900);
    if(reduced){focusAllocationCamera(stage,0);paintAllocationMap(1,performance.now());}
    lastAllocationFrameSignature='';paintAllocationMap(reduced?1:allocationStageProgress(allocationMapClock()),performance.now());requestAnimationFrame(positionAllocationLabels);
  }
  function focusAllocationCamera(stage,duration=2000){
    if(!ready||state.scene!=='allocation')return;const key=stage+'|'+innerWidth+'|'+innerHeight+'|'+state.flat;if(allocationCameraToken===key&&duration)return;allocationCameraToken=key;
    if(stage===0){allocationLocalCamera=true;frameCoordinates([allocationMapAnchors.home.coordinate,...allocationMapAnchors.poiAnchors.map(p=>p.coordinate)],duration,15.1);}
    else if(stage===1){frameCoordinates([allocationMapAnchors.home.coordinate,...allocationMapAnchors.poiAnchors.map(p=>p.coordinate)],duration,14.2);}
    else{const b=actualBudget.bounds,padding=cameraPadding(),available=Math.max(180,innerHeight-padding.top-padding.bottom);map.easeTo({center:[(b.west+b.east)/2,(b.south+b.north)/2],zoom:(state.flat?7.20:7.35)+Math.log2(available/385),padding,pitch:state.flat?0:48,bearing:-16,duration:reduced?0:duration,easing:t=>1-Math.pow(1-t,3)});}
  }
  function paintAllocationMap(progress,now,forceComplete=false){
    const stage=forceComplete?2:state.allocationStage,showBudget=state.budgetLayer&&stage===2,p=stage===2&&allocationRevealComplete?1:Math.max(0,Math.min(1,progress)),rowLimit=Math.floor(p*(actualBudget.height-1));
    const signature=[stage,Math.round(p*1000),state.motion,state.budgetLayer,cityContext].join('|');if(lastAllocationFrameSignature===signature)return;lastAllocationFrameSignature=signature;
    if(map.getLayer('budget-fill'))map.setPaintProperty('budget-fill','fill-color',stage<2?'#d1dbd4':['get','budget_color']);
    if(map.getLayer('budget-fill'))map.setPaintProperty('budget-fill','fill-opacity',!state.budgetLayer?0:stage===0?.10:stage===1?.16:['case',['<=',['get','row'],rowLimit],1,.12]);
    for(const id of ['city-budget-3d','buildings-3d'])if(map.getLayer(id))map.setPaintProperty(id,'fill-extrusion-color',showBudget?['case',['<=',['coalesce',['get','row'],-1],rowLimit],budgetColorExpression(),'#d0d5d7']:'#d0d5d7');
    if(map.getLayer('city-buildings'))map.setPaintProperty('city-buildings','fill-extrusion-color',showBudget?['case',['<=',['coalesce',['feature-state','budget_row'],-1],rowLimit],cityBudgetColorExpression(),'#d0d5d7']:'#d0d5d7');
    if(map.getLayer('allocation-scan-line'))map.setFilter('allocation-scan-line',['==',['get','row'],p<1?rowLimit:-1]);
    if(stage===0){
      const value=residentStoryFrame(allocationMapClock(now)).cumulative,el=$('map-home-a');if(el)el.textContent=num(value,2);tween('allocation-a',value,value,0,true);
    }else if(stage===2){
      const homeBudget=allocationMapAnchors.home.ICB*(1-Math.pow(1-p,3));tween('allocation-budget',homeBudget,homeBudget,0,true);
      const covered=allocationRowsPrefix[rowLimit]||0;
      map.getCanvas().dataset.allocationCoveredCells=String(covered);if(p===1)allocationRevealComplete=true;
    }
    const particles=[];
    if(state.motion&&!reduced&&p<1){
      const lines=stage===0?allocationAssociationLines:[];
      lines.forEach((line,i)=>{for(let j=0;j<2;j++){const fraction=(p*2.2+i*.137+j*.5)%1,position=pointAt(line.coordinates,fraction).coordinate;particles.push({type:'Feature',geometry:{type:'Point',coordinates:position},properties:{color:stage===0?line.color:'#b1d6b9',radius:stage===0?2.7:3.4}});}});
    }
    map.getSource('allocation-particle-data').setData(featureCollection(particles));
    if(stage===1){const pulse=1+Math.sin(p*Math.PI*4)*.10;map.setPaintProperty('allocation-weight-glow','circle-opacity',.10+.07*pulse);allocationHomeNode.getElement().style.setProperty('--weight-reveal',String(Math.max(0,Math.min(1,(p-.15)/.45))));}
    map.getCanvas().dataset.allocationProgress=String(Math.round(p*100));
  }
  function animateAllocationMap(now){
    if(!allocationMapReady||!ready)return;syncAllocationMap();
    if(state.scene!=='allocation')return;
    const stage=state.allocationStage,clock=allocationMapClock(now),length=stage===0?STORY.weightStart:stage===1?STORY.distributionStart-STORY.weightStart:STORY.allocationEnd-STORY.distributionStart,local=Math.max(0,clock-allocationStageTimes[stage]);
    if(stage===0&&!allocationLocalCamera&&(allocationVisualRunning||state.playing||allocationAuto)&&(local>.65||!allocationAuto&&!state.playing))focusAllocationCamera(0,reduced?0:2200);
    const showNearby=stage===0&&map.getZoom()>11.7;allocationPoiNodes.forEach(({marker})=>marker.getElement().hidden=!showNearby);allocationHomeNode.getElement().hidden=stage===2||stage===0&&!showNearby;
    paintResidentStory(clock,now);
    const progress=allocationStageProgress(clock);
    if(now-lastAllocationPaint>85){lastAllocationPaint=now;paintAllocationMap(reduced?1:progress,now);}
    if(now-lastAllocationLabelLayout>200){lastAllocationLabelLayout=now;positionAllocationLabels();}
    if(stage===2&&clock>=STORY.distributionComplete&&!allocationRevealComplete)completeAllocationMap();
    if(!state.playing&&!allocationAuto&&local>=length){if(stage===2)completeAllocationMap();else{allocationVisualClock=allocationStageTimes[stage]+length;allocationVisualRunning=false;setAllocationMotionClass(false);}}
  }
  function positionAllocationLabels(){
    if(!ready||!allocationMapReady||state.scene!=='allocation')return;
    const left=innerWidth>680?document.querySelector('.data-panel').getBoundingClientRect().right+14:12,right=innerWidth-80,top=134,bottom=innerHeight>600?innerHeight-145:innerHeight-80,obstacles=[];
    const addObstacle=m=>{if(m&&!m.getElement().hidden){const r=m.getElement().getBoundingClientRect();obstacles.push({x:r.x,y:r.y,width:r.width,height:r.height});}};
    addObstacle(allocationHomeNode);
    addObstacle(allocationResidentMarker);const card=$('resident-calculation-card');if(card&&!card.hidden){const r=card.getBoundingClientRect();obstacles.push({x:r.x,y:r.y,width:r.width,height:r.height});}
    const nodes=state.allocationStage===0?allocationPoiNodes:[];
    for(const {marker} of nodes){const el=marker.getElement();if(el.hidden)continue;const width=el.offsetWidth||110,height=el.offsetHeight||56,p=map.project(marker.getLngLat()),options=[[14,-height-12],[-width-14,-height-12],[14,14],[-width-14,14],[20,-height/2],[-width-20,-height/2],[-width/2,-height-32],[-width/2,32]];let best=null;
      for(const [index,offset] of options.entries()){const r={x:p.x+offset[0],y:p.y+offset[1],width,height};let cost=(Math.max(0,left-r.x)+Math.max(0,r.x+width-right))*height*5+(Math.max(0,top-r.y)+Math.max(0,r.y+height-bottom))*width*5+index*8;
        for(const o of obstacles){const x=Math.max(0,Math.min(r.x+width,o.x+o.width)-Math.max(r.x,o.x)),y=Math.max(0,Math.min(r.y+height,o.y+o.height)-Math.max(r.y,o.y));cost+=x*y*12;}if(!best||cost<best.cost)best={cost,offset,rect:r};}
      marker.setOffset(best.offset);obstacles.push(best.rect);
    }
  }

  let allocationResidentMarker,residentCardBeat='',residentLastClock=-1,residentLineBeat='',residentPosition='',residentSvgParts;
  let residentWaypoints=null;
  const residentBeatDurations=[2.5,.85,.85,.85,.85,.85,.85];
  const residentRouteCache=new WeakMap();
  const residentClamp=(x,a=0,b=1)=>Math.max(a,Math.min(b,x));
  const residentEase=x=>{x=residentClamp(x);return x*x*(3-2*x);};
  function residentStoryFrame(clock){
    const activities=contributionStory.activities;let cursor=.6,cumulative=0;
    for(let i=0;i<activities.length;i++){
      const duration=residentBeatDurations[i],end=cursor+duration;
      if(clock<end){const t=residentClamp((clock-cursor)/duration),arrival=residentEase((t-.77)/.23);return {index:i,t,start:cursor,end,item:activities[i],cumulative:cumulative+activities[i].weightedContribution*arrival,done:false,returning:false};}
      cumulative+=activities[i].weightedContribution;cursor=end;
    }
    return {index:activities.length-1,t:1,start:cursor,end:STORY.weightStart,item:activities[activities.length-1],cumulative:contributionStory.totalA,returning:clock<cursor+1.45,done:clock>=cursor+1.45};
  }
  // Each segment joins exact source coordinates. Tangent handles give continuous turns.
  function residentRouteGeometry(waypoints){
    if(residentRouteCache.has(waypoints))return residentRouteCache.get(waypoints);
    const cos=Math.cos(waypoints.reduce((s,p)=>s+p[1],0)/waypoints.length*Math.PI/180),metric=p=>[p[0]*cos,p[1]],points=waypoints.map(metric),segments=[];
    const unit=(a,b)=>{const x=b[0]-a[0],y=b[1]-a[1],n=Math.hypot(x,y)||1;return [x/n,y/n];};
    let start=.6,cyclesBefore=0,distanceBefore=0;
    for(let i=0;i<points.length-1;i++){
      const a=points[i],b=points[i+1],d=Math.hypot(b[0]-a[0],b[1]-a[1]),ta=unit(points[Math.max(0,i-1)],b),tb=unit(a,points[Math.min(points.length-1,i+2)]),handle=d*.28;
      const c=[a[0]+ta[0]*handle,a[1]+ta[1]*handle],e=[b[0]-tb[0]*handle,b[1]-tb[1]*handle],samples=[],lengths=[0];let length=0;
      for(let k=0;k<=64;k++){const t=k/64,u=1-t,p=[u*u*u*a[0]+3*u*u*t*c[0]+3*u*t*t*e[0]+t*t*t*b[0],u*u*u*a[1]+3*u*u*t*c[1]+3*u*t*t*e[1]+t*t*t*b[1]];if(k)length+=Math.hypot(p[0]-samples[k-1][0],p[1]-samples[k-1][1]);samples.push(p);if(k)lengths.push(length);}
      const duration=i===0?1.95:i===7?1.45:1.6,end=i===7?start+duration:start+residentBeatDurations[i];
      segments.push({samples,lengths,length,cos,start,end,duration,arrival:start+duration,cyclesBefore,distanceBefore,index:i});
      cyclesBefore+=duration*1.62;distanceBefore+=length;start=end;
    }
    const result={segments,cos,totalCycles:cyclesBefore,totalDistance:distanceBefore,end:start};residentRouteCache.set(waypoints,result);return result;
  }
  function residentPointOnSegment(segment,fraction){
    const target=residentClamp(fraction)*segment.length;let i=1;while(i<segment.lengths.length-1&&segment.lengths[i]<target)i++;
    const a=segment.samples[i-1],b=segment.samples[i],f=(target-segment.lengths[i-1])/(segment.lengths[i]-segment.lengths[i-1]||1);
    return [(a[0]+(b[0]-a[0])*f)/segment.cos,a[1]+(b[1]-a[1])*f];
  }
  // Pure, clock-driven motion: repeat/seek/pause at the same clock yields the same pose.
  function residentMotionSample(clock,waypoints,project){
    const route=residentRouteGeometry(waypoints);let segment=route.segments.find(s=>clock<s.end)||route.segments[route.segments.length-1];
    const started=clock>=.6,finished=clock>=route.end,u=residentClamp((clock-segment.start)/segment.duration),fraction=residentEase(u);
    const coordinate=!started?waypoints[0].slice():finished?waypoints[waypoints.length-1].slice():residentPointOnSegment(segment,fraction);
    const a=residentPointOnSegment(segment,Math.max(0,fraction-.012)),b=residentPointOnSegment(segment,Math.min(1,fraction+.012));
    const screen=p=>{if(project){const q=project(p);return Array.isArray(q)?q:[q.x,q.y];}return [p[0]*route.cos,-p[1]];},sa=screen(a),sb=screen(b),dx=sb[0]-sa[0],dy=sb[1]-sa[1],norm=Math.hypot(dx,dy)||1,direction=[dx/norm,dy/norm];
    const travelling=started&&!finished&&clock<segment.arrival,cycles=finished?route.totalCycles:!started?0:segment.cyclesBefore+segment.duration*fraction*1.62;
    return {coordinate,segmentIndex:segment.index,travelPhase:u,travelFraction:fraction,travelling,finished,cycles,walkingDistance:finished?route.totalDistance:!started?0:segment.distanceBefore+segment.length*fraction,direction,heading:Math.atan2(direction[0],-direction[1])*180/Math.PI,motion:travelling?Math.pow(Math.sin(Math.PI*u),.45):0};
  }
  function residentLegJoints(phase,motion,side,direction,bob){
    const p=((phase%1)+1)%1,stance=p<.62,t=stance?p/.62:(p-.62)/.38,fore=motion*(stance?8-16*t:-8+16*residentEase(t)),lift=stance?0:6*motion*Math.pow(Math.sin(Math.PI*t),1.35);
    const hip=[0,44+bob],ankle=[fore,77.5-lift],dx=ankle[0]-hip[0],dy=ankle[1]-hip[1],distance=Math.min(34.999,Math.hypot(dx,dy)),half=distance/2,bend=Math.sqrt(Math.max(0,17.5*17.5-half*half)),knee=[hip[0]+dx*.5+dy/distance*bend,hip[1]+dy*.5-dx/distance*bend];
    const spread=side*(1.1+Math.abs(direction[1])),screen=p=>[32+p[0]*direction[0]+spread,p[1]];
    return {hip:screen(hip),knee:screen(knee),ankle:screen(ankle),heel:screen([ankle[0]-2.3,ankle[1]]),toe:screen([ankle[0]+3.4,ankle[1]]),stance,footLift:lift};
  }
  function residentArmJoints(phase,motion,side,direction,bob,lean){
    const swing=Math.cos(phase*Math.PI*2)*.44*motion,shoulder=[32+side*4.1+lean,27+bob],upper=swing,lower=swing+.20+.15*motion;
    const elbow=[shoulder[0]+Math.sin(upper)*10*direction[0],shoulder[1]+Math.cos(upper)*10],hand=[elbow[0]+Math.sin(lower)*10*direction[0],elbow[1]+Math.cos(lower)*10];
    return {shoulder,elbow,hand};
  }
  function residentPoseAt(sample){
    const motion=residentClamp(sample.motion||0),cycles=sample.cycles||0,direction=sample.direction||[1,0],bob=-.8*motion*(1-Math.cos(4*Math.PI*cycles))*.5,lean=direction[0]*motion*1.05;
    return {motion,cycles,direction,bob,head:[32+lean,13+bob],torso:{left:26+lean,right:38+lean,top:24+bob,bottom:45+bob},backLeg:residentLegJoints(cycles+.5,motion,-1,direction,bob),frontLeg:residentLegJoints(cycles,motion,1,direction,bob),backArm:residentArmJoints(cycles,motion,-1,direction,bob,lean),frontArm:residentArmJoints(cycles+.5,motion,1,direction,bob,lean),nose:[32+lean+direction[0]*5.3,14+bob],faceOpacity:.25+.75*residentClamp((direction[1]+1)/2)};
  }
  const residentPath=points=>points.map((p,i)=>(i?'L':'M')+p.map(x=>x.toFixed(2)).join(' ')).join('');
  function setResidentSvgAttribute(part,name,value){const el=residentSvgParts?.[part];if(el&&el.getAttribute(name)!==String(value))el.setAttribute(name,value);}
  function paintResidentPose(sample){
    if(!residentSvgParts)return;const p=residentPoseAt(sample);
    for(const [part,key] of [['back-leg','backLeg'],['front-leg','frontLeg']]){const leg=p[key];setResidentSvgAttribute(part,'d',residentPath([leg.hip,leg.knee,leg.ankle]));setResidentSvgAttribute(part==='back-leg'?'back-foot':'front-foot','d',residentPath([leg.heel,leg.toe]));}
    for(const [part,key] of [['back-arm','backArm'],['front-arm','frontArm']]){const arm=p[key];setResidentSvgAttribute(part,'d',residentPath([arm.shoulder,arm.elbow,arm.hand]));}
    setResidentSvgAttribute('neck','d',residentPath([[p.head[0],p.head[1]+4],[p.head[0],p.head[1]+13]]));
    const t=p.torso;setResidentSvgAttribute('torso','d',`M${t.left} ${t.top}Q${p.head[0]} ${t.top-3} ${t.right} ${t.top}L${t.right-1.5} ${t.bottom}H${t.left+1.5}Z`);
    for(const [key,coordinate] of [['head',p.head],['nose',p.nose]]){setResidentSvgAttribute(key,'cx',coordinate[0].toFixed(2));setResidentSvgAttribute(key,'cy',coordinate[1].toFixed(2));}
    setResidentSvgAttribute('nose','opacity',p.faceOpacity.toFixed(2));setResidentSvgAttribute('hair','d',`M${p.head[0]-5.5} ${p.head[1]}Q${p.head[0]-5} ${p.head[1]-7} ${p.head[0]+1} ${p.head[1]-5.8}Q${p.head[0]+5.5} ${p.head[1]-5.5} ${p.head[0]+5.3} ${p.head[1]-1.3}`);
  }
  function setupResidentStory(){
    const el=document.createElement('div');el.className='allocation-resident';el.hidden=true;
    el.innerHTML=`<span class="resident-floor-shadow"></span><span class="resident-person-figure"><svg viewBox="0 0 64 84" aria-hidden="true"><path data-resident-part="back-arm" class="resident-skin resident-back-limb"/><path data-resident-part="back-leg" class="resident-trousers resident-back-limb"/><path data-resident-part="back-foot" class="resident-shoe resident-back-limb"/><path data-resident-part="neck" class="resident-neck"/><path data-resident-part="torso" class="resident-shirt"/><path data-resident-part="front-leg" class="resident-trousers"/><path data-resident-part="front-foot" class="resident-shoe"/><path data-resident-part="front-arm" class="resident-skin"/><circle data-resident-part="head" r="5.6" class="resident-head"/><circle data-resident-part="nose" r="1.15" class="resident-head"/><path data-resident-part="hair" class="resident-hair"/></svg></span><span class="resident-speech" id="resident-speech">家</span>`;
    residentSvgParts=Object.fromEntries(Array.from(el.querySelectorAll('[data-resident-part]')).map(node=>[node.dataset.residentPart,node]));
    residentWaypoints=[allocationMapAnchors.home.coordinate,...allocationMapAnchors.poiAnchors.map(p=>p.coordinate),allocationMapAnchors.home.coordinate];
    allocationResidentMarker=new maplibregl.Marker({element:el,anchor:'bottom',offset:[0,-3]}).setLngLat(allocationMapAnchors.home.coordinate).addTo(map);paintResidentPose(residentMotionSample(0,residentWaypoints));
  }
  function renderResidentCard(frame){
    const el=$('resident-calculation-card');if(!el)return;const item=frame.item,isExpanded=frame.index===0,complete=frame.done||frame.returning,key=complete?'complete':String(frame.index);
    el.hidden=state.scene!=='allocation'||state.allocationStage!==0;if(el.hidden||residentCardBeat===key)return;residentCardBeat=key;el.dataset.category=item.id;el.style.setProperty('--resident-activity-color',allocationCategoryArt[item.id]?.color||'#68967e');
    if(complete){el.innerHTML=`<h3>七类需求，汇成综合指标</h3><div class="resident-sum-equation">${contributionStory.activities.map(x=>`<span>${esc(x.label)}<b>${num(x.weightedContribution,2)}</b></span>`).join('')}</div><div class="resident-contribution-result"><span>居住地综合可达性指数 A</span><strong>${num(contributionStory.totalA,2)}</strong></div>`;return;}
    const serviceName=item.label==='交通站点'?'交通站点可达性':`${item.label}服务可达性`;
    el.innerHTML=`<h3>${esc(item.residentNeed)}</h3>${isExpanded?`<div class="resident-index-source"><span>医疗设施分布</span><b>＋</b><span>周边人口需求</span><b>＋</b><span>出行距离与时间</span></div><div class="resident-source-arrow">↓ 分别计算四种方式的可达性</div><div class="resident-four-modes">${item.modes.map((m,i)=>`<div data-resident-mode="${i}">${icon(normalize(m.id))}<span>${esc(m.label)}</span><strong>${num(m.scaledValue,3)}</strong><small>× ${num(m.weight*100)}%</small></div>`).join('')}</div><div class="resident-combination-caption">↓ 四种方式各占25%，加权汇总</div>`:''}<div class="resident-mode-total"><span>${esc(serviceName)}</span><strong>${num(item.activityAccessibility,2)}</strong></div><div class="resident-weight-equation"><span>${num(item.activityAccessibility,2)}</span><i>×</i><span class="resident-preference-label" title="可自主设置；当前展示采用默认值">您对${esc(preferenceActivityName(item.id))}活动的重视程度</span><b>${num(item.activityWeight*100)}%</b></div><div class="resident-contribution-result"><span>${esc(item.label)}贡献值</span><strong>+ ${num(item.weightedContribution,2)}</strong></div>${isExpanded?'<p class="resident-index-note">数值为模型指数，非场所数量或百分制评分；四种方式的原始指标统一放大1000倍展示。</p>':''}`;
  }
  function paintResidentStory(clock,now){
    if(!allocationResidentMarker)return;const markerEl=allocationResidentMarker.getElement(),card=$('resident-calculation-card'),visible=state.scene==='allocation'&&state.allocationStage<2;markerEl.hidden=!visible;
    if(!visible){if(card)card.hidden=true;return;}const stage=state.allocationStage,speech=$('resident-speech');
    if(stage===0){
      const frame=residentStoryFrame(clock);renderResidentCard(frame);let sample=residentMotionSample(clock,residentWaypoints,p=>map.project(p));
      const position=sample.coordinate.join(',');if(position!==residentPosition){allocationResidentMarker.setLngLat(sample.coordinate);residentPosition=position;}if(!state.motion||reduced)sample={...sample,motion:0};paintResidentPose(sample);markerEl.classList.toggle('resident-is-walking',sample.travelling&&sample.motion>0);
      speech.hidden=sample.travelling;speech.textContent=sample.finished||clock<.6?'家':frame.item.residentNeed;
      allocationPoiNodes.forEach(({marker},i)=>{marker.getElement().classList.toggle('story-active',!frame.done&&!frame.returning&&i===frame.index);marker.getElement().classList.toggle('story-done',frame.done||frame.returning||i<frame.index);});
      document.querySelectorAll('[data-contribution]').forEach(el=>el.classList.toggle('story-active',!frame.done&&!frame.returning&&Number(el.dataset.contribution)===frame.index));
      const lineBeat=frame.done||frame.returning?'complete':String(frame.index);if(residentLineBeat!==lineBeat&&map.getLayer('allocation-association-lines')){residentLineBeat=lineBeat;map.setPaintProperty('allocation-association-lines','line-opacity',(frame.done||frame.returning)? .44:['case',['==',['get','activity_index'],frame.index],.95,['<',['get','activity_index'],frame.index],.45,.13]);}
      if(card){card.style.setProperty('--resident-beat-progress',String(frame.t));card.dataset.step=frame.t<.30?'approach':frame.t<.61?'modes':frame.t<.69?'combined':frame.t<.77?'weight':'contribution';card.querySelectorAll('[data-resident-mode]').forEach((el,i)=>el.classList.toggle('mode-in-focus',frame.index===0&&frame.t>=.30&&frame.t<.61&&i===Math.min(3,Math.floor((frame.t-.30)/.0775))));}
    }else{
      if(card)card.hidden=true;allocationResidentMarker.setLngLat(allocationMapAnchors.home.coordinate);residentPosition=allocationMapAnchors.home.coordinate.join(',');markerEl.classList.remove('resident-is-walking');speech.hidden=true;paintResidentPose({...residentMotionSample(22,residentWaypoints,p=>map.project(p)),motion:0});
    }
    residentLastClock=clock;
  }
  function hideResidentStory(){if(allocationResidentMarker)allocationResidentMarker.getElement().hidden=true;const card=$('resident-calculation-card');if(card)card.hidden=true;}
  function resetResidentStory(){residentCardBeat='';residentLastClock=-1;residentLineBeat='';residentPosition='';hideResidentStory();}

  function homePoint() {return actualBudget.points.find(p=>p.id==='a')||actualBudget.points[0];}
  function currentPoint() {return state.selection||actualBudget.points.find(p=>p.id===state.comparePoint)||homePoint();}
  function cityBounds() {const b=actualBudget.bounds;return [[b.west,b.south],[b.east,b.north]];}
  function budgetColorExpression() {return ['coalesce',['get','budget_color'],'#c8d1d2'];}
  function cityBudgetColorExpression() {return ['coalesce',['feature-state','budget_color'],'#c8d1d2'];}
  function ringAreaCentroid(ring) {
    if(!Array.isArray(ring)||ring.length<3)return null;
    const origin=ring[0];
    if(!Array.isArray(origin)||!origin.every(Number.isFinite))return null;
    let crossSum=0,cx=0,cy=0;
    for(let i=0;i<ring.length;i++) {
      const a=ring[i],b=ring[(i+1)%ring.length];
      if(!a||!b||!Number.isFinite(a[0])||!Number.isFinite(a[1])||!Number.isFinite(b[0])||!Number.isFinite(b[1]))return null;
      const ax=a[0]-origin[0],ay=a[1]-origin[1],bx=b[0]-origin[0],by=b[1]-origin[1],cross=ax*by-bx*ay;
      crossSum+=cross;cx+=(ax+bx)*cross;cy+=(ay+by)*cross;
    }
    if(!Number.isFinite(crossSum)||Math.abs(crossSum)<1e-20)return null;
    return {area:Math.abs(crossSum)/2,center:[origin[0]+cx/(3*crossSum),origin[1]+cy/(3*crossSum)]};
  }
  function polygonAreaCentroid(rings) {
    if(!Array.isArray(rings)||!rings.length)return null;
    const exterior=ringAreaCentroid(rings[0]);
    if(!exterior)return null;
    let area=exterior.area,cx=area*exterior.center[0],cy=area*exterior.center[1];
    for(const hole of rings.slice(1)) {
      const part=ringAreaCentroid(hole);if(!part)continue;
      area-=part.area;cx-=part.area*part.center[0];cy-=part.area*part.center[1];
    }
    if(!Number.isFinite(area)||area<=1e-20)return null;
    return {area,center:[cx/area,cy/area]};
  }
  function geometryAreaCentroid(geometry) {
    if(!geometry||!['Polygon','MultiPolygon'].includes(geometry.type))return null;
    const polygons=geometry.type==='Polygon'?[geometry.coordinates]:geometry.coordinates;
    if(!Array.isArray(polygons))return null;
    let area=0,cx=0,cy=0;
    for(const polygon of polygons) {
      const part=polygonAreaCentroid(polygon);if(!part)continue;
      area+=part.area;cx+=part.area*part.center[0];cy+=part.area*part.center[1];
    }
    if(!Number.isFinite(area)||area<=1e-20)return null;
    const center=[cx/area,cy/area];
    return center.every(Number.isFinite)?{area,center}:null;
  }
  function actualCellAt(coordinate) {
    if(!actualCellLookup)actualCellLookup=new Map(actualBudgetCells.features.map(f=>[`${f.properties.row},${f.properties.col}`,f.properties]));
    const b=actualBudget.bounds,[lon,lat]=coordinate;
    if(lon<b.west||lon>=b.east||lat<b.south||lat>=b.north)return null;
    const dx=actualBudget.mapping?.dx||(b.east-b.west)/actualBudget.width,dy=actualBudget.mapping?.dy||(b.north-b.south)/actualBudget.height;
    const col=Math.floor((lon-b.west)/dx),row=Math.floor((lat-b.south)/dy);
    return actualCellLookup.get(`${row},${col}`)||null;
  }
  function syncCityBudgetStates() {
    if(!map||!ready||!cityContext||!state.budgetLayer||!map.getSource('city-context'))return;
    const byId=new Map();
    for(const feature of map.querySourceFeatures('city-context',{sourceLayer:'building'})) {
      const id=feature.id;
      if(id==null||!['number','string'].includes(typeof id)||(typeof id==='number'&&!Number.isSafeInteger(id)))continue;
      const centroid=geometryAreaCentroid(feature.geometry),previous=byId.get(id);
      if(!byId.has(id)||(centroid&&(!previous||centroid.area>previous.area)))byId.set(id,centroid);
    }
    for(const [id,centroid] of byId) {
      const cell=centroid?actualCellAt(centroid.center):null;
      const color=typeof cell?.budget_g==='number'&&typeof cell?.budget_color==='string'?cell.budget_color:'#c8d1d2';
      if(cityBudgetColors.get(id)===color)continue;
      cityBudgetColors.set(id,color);
      map.setFeatureState({source:'city-context',sourceLayer:'building',id},{budget_color:color,budget_row:Number.isFinite(cell?.row)?cell.row:-1});
    }
  }


  function tween(key,to,from,duration=1350,force=false){
    if(!Number.isFinite(to))return;
    const current=force&&Number.isFinite(from)?from:tweens.has(key)?tweenValue(key):Number.isFinite(from)?from:to;
    tweens.set(key,{from:current,to,start:state.time*1000,duration:reduced||!state.playing?0:duration});
  }
  function tweenValue(key){const t=tweens.get(key);if(!t)return 0;const x=t.duration?Math.min(1,Math.max(0,state.time*1000-t.start)/t.duration):1;return t.from+(t.to-t.from)*(1-Math.pow(1-x,3));}
  function numberHTML(key,value,d=1){if(!tweens.has(key))tween(key,value,0);return `<span data-number="${key}" data-decimals="${d}">${num(tweenValue(key),d)}</span>`;}
  function animateNumbers(){document.querySelectorAll('[data-number]').forEach(el=>{const key=el.dataset.number;if(tweens.has(key)){const text=num(tweenValue(key),Number(el.dataset.decimals||0));if(el.textContent!==text)el.textContent=text;}});}
  async function jsonRequest(path,body,signal){
    const controller=new AbortController(),abort=()=>controller.abort();let timedOut=false;
    if(signal?.aborted)controller.abort();else signal?.addEventListener('abort',abort,{once:true});
    const timeout=setTimeout(()=>{timedOut=true;controller.abort();},path.includes('/route-chain')?205000:20000);
    try{
      const response=await fetch(path,{method:body?'POST':'GET',headers:body?{'Content-Type':'application/json'}:{},body:body?JSON.stringify(body):undefined,signal:controller.signal});
      let result;try{result=await response.json();}catch{throw new Error('计算服务暂未响应，请稍后重试。');}
      if(!response.ok||result.success===false)throw new Error(result.error||result.message||'本次计算未完成');
      return result;
    }catch(error){if(timedOut)throw new Error('连接等待超时，可取消后重新计算。');if(error instanceof TypeError)throw new Error('计算服务暂未连接，请重新打开展示入口。');throw error;}
    finally{clearTimeout(timeout);signal?.removeEventListener('abort',abort);}
  }
  function setConnection(ok){$('connection-status').classList.toggle('connected',ok);$('connection-status').textContent=ok?'计算服务已连接':'样例回放';$('connection-status').title=ok?'实时计算服务可用':'实时计算服务未连接；可浏览预计算样例并复核已保存结果。';}
  function modeButtons(index){return `<div class="mode-selector" role="group" aria-label="第${index+1}段出行方式">${Object.entries(MODE).map(([id,m])=>`<button type="button" data-mode="${id}" data-mode-leg="${index}" class="mode-option ${state.modes[index]===id?'active':''}" aria-pressed="${state.modes[index]===id}" style="--mode-color:${m.color}">${icon(id)}<span>${m.label}</span></button>`).join('')}</div>`;}
  function carbonText(v){return Number.isFinite(v)?num(v)+'<small>g CO₂</small>':'—<small>g CO₂</small>';}
  function renderPanel(force=false){
    if(!day||!actualBudget)return;
    const key=[state.scene,state.activeLeg,state.budgetView,selectionVersion,routeVersion,state.routeBusy,state.recType,state.recQuery,state.recSearch,state.stayMin,state.gap,state.recMode,state.recEditing,state.recommended,state.added,state.candidateId,state.recBusy,state.allocationStage===0?0:1,notice].join('|');
    if(!force&&lastPanelKey===key)return;lastPanelKey=key;
    const panel=$('panel-content'),a=activity(),c=candidate(),p=actualBudget.parameters;
    document.querySelector('.data-panel').dataset.scene=state.scene;document.querySelector('.data-panel').dataset.changed=String(!!comparison);document.querySelector('.data-panel').dataset.busy=String(state.routeBusy);document.querySelector('.data-panel').dataset.editing=String(state.recEditing);
    document.querySelectorAll('[data-scene]').forEach(b=>{b.classList.toggle('active',(b.dataset.scene==='chain'?['chain','recommendation'].includes(state.scene):b.dataset.scene===state.scene));b.setAttribute('aria-pressed',String((b.dataset.scene==='chain'?['chain','recommendation'].includes(state.scene):b.dataset.scene===state.scene)));});
    if(state.scene==='convenience'){
      panel.innerHTML=convenience?.html()||'<p class="panel-intro">正在准备路线与活动空间…</p>';
      convenience?.bind();
    }else if(state.scene==='allocation'){
      panel.innerHTML=allocationHTML();
    }else if(state.scene==='chain'){
      const delta=comparison?comparison.after-comparison.before:0;
      panel.innerHTML=`<div class="chapter-kicker">南京市 / 个人尺度</div><div class="panel-title-row"><h2>我的一天</h2><span class="day-date" role="status">${state.routeBusy?'计算中…':'08:00—18:30'}</span></div><div class="hero-label">当日出行排放 ${state.routeBusy?'<span class="calculating">计算中</span>':''}</div><div class="hero-number">${numberHTML('day-carbon',day.totalCarbonG,1)}<span>g CO₂</span></div><div class="day-metrics"><span><strong>${numberHTML('day-time',day.totalTravelMin,1)}</strong> min</span><span><strong>${num(day.totalDistanceKm,2)}</strong> km</span><span class="budget-balance ${day.totalCarbonG>homePoint().budget_g?'over':'within'}">${day.totalCarbonG>homePoint().budget_g?'超预算':'预算余量'} <b>${num(Math.abs(homePoint().budget_g-day.totalCarbonG))}</b> g</span></div>${comparison?`<div class="change-result ${delta<=0?'decrease':'increase'}"><span>${delta<=0?'↓ 减少排放':'↑ 排放增加'} <strong>${num(Math.abs(delta))}</strong> g</span><small>${comparison.afterTime<comparison.beforeTime?'节约':'增加'} ${num(Math.abs(comparison.afterTime-comparison.beforeTime))} min</small></div>`:'<div class="chain-hint">选择每一段的出行方式，观察时间与碳排的变化。</div>'}<div class="trip-list">${day.legs.map((l,i)=>{const m=modeInfo(l);return `<section class="trip-leg ${state.activeLeg===i?'selected':''}" style="--leg-color:${m.color}"><button type="button" class="trip-heading" data-leg="${i}"><span class="trip-order">0${i+1}</span><span class="trip-names">${esc(l.from)} <i>→</i> ${esc(l.to)}</span><span class="trip-depart">${i===0?'08:00':'17:00'}</span></button>${modeButtons(i)}<div class="trip-summary"><span>${esc(m.label)} · ${num(l.timeMin)} min</span><strong>${numberHTML('leg-'+i,l.carbonG,1)}<small>g</small></strong></div>${l.walkFallback||l.routeSource==='straight_line'?`<div class="route-availability">${l.walkFallback?'步行接驳':'路线估算'}</div>`:''}</section><button type="button" class="gap-add-action" data-add-gap="${i}">${icon('plus')} ${i===0?'上班前 · 08:00—09:00':'下班后 · 17:00—18:30'}，在此空档添加活动</button>`;}).join('')}</div>`;
    }else{
      const category=config.activityTypes.find(x=>(x.id||x.value)===state.recType),name=state.recSearch?(state.recQuery||'搜索活动场所'):(category?.label||category?.name||'公园');
      const anchors=gapAnchors();
      panel.innerHTML=`<button type="button" id="back-to-day" class="back-to-day">← 我的一天 / 添加活动</button><div class="chapter-kicker">日程安排 / 时间预算</div><h2>今天，还想做什么？</h2><p class="panel-intro">日程已经确定，把一件想做的事放进空档。</p><div class="gap-summary"><span><b>${anchors.from.end}</b>${esc(anchors.from.label)}出发</span><div><i></i><strong>${anchors.minutes}<small> min</small></strong><i></i></div><span><b>${anchors.to.start}</b>前到${esc(anchors.to.label)}</span></div>${state.recEditing?`<form id="activity-form"><label class="field-label" for="gap-select">安排在哪段行程？</label><select id="gap-select" class="field-control"><option value="1" ${state.gap===1?'selected':''}>下班后 · 单位 → 家</option><option value="0" ${state.gap===0?'selected':''}>上班前 · 家 → 单位</option></select><div class="search-tabs"><button type="button" data-search-tab="types" class="${!state.recSearch?'active':''}">选择活动</button><button type="button" data-search-tab="keyword" class="${state.recSearch?'active':''}">搜索场所</button></div>${state.recSearch?`<div class="search-field">${icon('search')}<input id="activity-search" type="search" autocomplete="off" placeholder="搜索咖啡、图书馆、健身房或地点名称" value="${esc(state.recQuery)}" aria-label="活动或地点关键词"></div><div id="search-suggestions" class="search-suggestions"></div>`:`<select id="activity-type" class="field-control" aria-label="活动类型">${config.activityTypes.map(t=>`<option value="${esc(t.id||t.value)}" ${(t.id||t.value)===state.recType?'selected':''}>${esc(t.label||t.name)}</option>`).join('')}</select>`}<div class="form-pair"><label><span class="field-label">停留时长</span><span class="duration-control"><input id="activity-duration" type="number" min="1" max="240" value="${state.stayMin}" aria-label="活动停留分钟"><small>分钟</small></span></label><label><span class="field-label">前往方式</span><select id="rec-mode" class="field-control" aria-label="额外活动出行方式">${Object.entries(MODE).map(([k,v])=>`<option value="${k}" ${state.recMode===k?'selected':''}>${v.label}</option>`).join('')}</select></label></div><button type="submit" class="primary-action" ${state.recBusy?'disabled':''}>${state.recBusy?'<span class="loading-orbit"></span> 正在寻找可行场所':`推荐场所 ${icon('arrow')}`}</button></form>`:`<div class="condition-summary"><span>${esc(name)} · ${state.stayMin}分钟 · ${MODE[state.recMode].label}</span><button type="button" id="edit-recommendation">修改条件</button></div>`}${state.recBusy?jobProgressHTML():''}${state.recommended&&a?`<div class="results-heading"><span>${a.candidates.length} 个可行场所</span><small>通行时间优先</small></div>${a.candidates.length?`<div class="candidate-list">${a.candidates.map((x,i)=>`<button type="button" class="candidate-card ${state.added&&c?.id===x.id?'selected':''}" data-candidate="${esc(x.id)}"><span class="candidate-rank">${String(i+1).padStart(2,'0')}</span><span><strong>${esc(placeName(x.name))}</strong><small>${num(x.totalTravelMin)} min 通行 · ${num(x.routeCarbonG)} g CO₂</small></span><i>${state.added&&c?.id===x.id?icon('check'):icon('plus')}</i></button>`).join('')}</div>`:'<div class="empty-result">当前条件下没有可行地点。试试缩短停留时间或更换出行方式。</div>'}${state.added&&c?`<div class="inserted-summary"><div><span>${icon('check')} 已加入日程</span><strong>${clock(c.arrivalHomeMinutes)} 到${esc(anchors.to.label)}</strong></div><p>${esc(placeName(c.name))}</p><div class="time-strip"><span style="flex:${c.travelBeforeMin}" class="travel"></span><span style="flex:${state.stayMin}" class="stay"></span><span style="flex:${c.travelAfterMin}" class="travel"></span><span style="flex:${Math.max(.1,c.remainingMin)}" class="free"></span></div>${recommendationCarbonHTML(c,a)}<div class="time-strip-labels"><span>通行 ${num(c.totalTravelMin)} + 停留 ${state.stayMin} min</span><b>余 ${num(c.remainingMin)} min</b></div></div>`:''}`:''}${notice?`<p class="inline-notice" role="status">${esc(notice)}</p>`:''}`;
    }
    panel.querySelectorAll('[data-allocation-stage]').forEach(b=>b.addEventListener('click',()=>{pause();budgetPopup?.remove();seek(Number(b.dataset.allocationStage)===0?0:STORY.weightStart);frameScene(0);}));
    $('allocation-replay')?.addEventListener('click',()=>{pause();budgetPopup?.remove();playbackStart=STORY.weightStart;playbackEnd=STORY.allocationEnd;seek(playbackStart);play();});
    $('back-to-day')?.addEventListener('click',()=>setScene('chain'));
    panel.querySelectorAll('[data-add-gap]').forEach(b=>b.addEventListener('click',()=>{state.gap=Number(b.dataset.addGap);state.recMode=state.modes[state.gap];setScene('recommendation');}));
    $('home-budget')?.addEventListener('click',()=>{state.budgetView='district';state.flat=false;updateCaption();updateMap();frameCoordinates([homePoint().coordinate,[118.786379,32.041372]],1200);});
    panel.querySelectorAll('[data-mode]').forEach(b=>b.addEventListener('click',()=>changeMode(Number(b.dataset.modeLeg),b.dataset.mode)));
    panel.querySelectorAll('[data-leg]').forEach(b=>b.addEventListener('click',()=>{pause();state.activeLeg=Number(b.dataset.leg);lastPanelKey='';render();frameCoordinates(day.legs[state.activeLeg].coordinates,850);}));
    $('add-activity')?.addEventListener('click',()=>{state.recMode=state.modes[state.gap];setScene('recommendation');});
    panel.querySelectorAll('[data-search-tab]').forEach(b=>b.addEventListener('click',()=>{state.recSearch=b.dataset.searchTab==='keyword';invalidateRecommendation();renderPanel(true);}));
    $('gap-select')?.addEventListener('change',e=>{state.gap=Number(e.target.value);state.recMode=state.modes[state.gap];invalidateRecommendation();renderPanel(true);});
    $('activity-type')?.addEventListener('change',e=>{state.recType=e.target.value;const t=config.activityTypes.find(x=>(x.id||x.value)===state.recType);if(t?.defaultStayMin)state.stayMin=t.defaultStayMin;invalidateRecommendation();renderPanel(true);});
    $('activity-duration')?.addEventListener('input',e=>{state.stayMin=Number(e.target.value);invalidateRecommendation(false);});
    $('rec-mode')?.addEventListener('change',e=>{state.recMode=e.target.value;invalidateRecommendation(false);});
    $('activity-search')?.addEventListener('input',e=>{state.recQuery=e.target.value;invalidateRecommendation(false);clearTimeout(searchTimer);searchTimer=setTimeout(()=>searchPlaces(state.recQuery),350);});
    $('activity-form')?.addEventListener('submit',e=>{e.preventDefault();pause();recommend();});
    $('edit-recommendation')?.addEventListener('click',()=>{pause();state.recEditing=true;invalidateRecommendation();renderPanel(true);});
    $('cancel-recommendation')?.addEventListener('click',()=>cancelRecommendation());
    $('use-partial-result')?.addEventListener('click',()=>usePartialResult());
    if(state.recBusy)paintJobProgress();
    panel.querySelectorAll('[data-candidate]').forEach(b=>b.addEventListener('click',()=>insertCandidate(b.dataset.candidate)));
  }
  function gapAnchors(){
    return state.gap===0?{from:{label:'家',coordinate:[118.773708,32.050131],end:'08:00',start:'00:00'},to:{label:'单位',coordinate:[118.786379,32.041372],start:'09:00',end:'17:00'},minutes:60}:{from:{label:'单位',coordinate:[118.786379,32.041372],start:'09:00',end:'17:00'},to:{label:'家',coordinate:[118.773708,32.050131],start:'18:30',end:'23:00'},minutes:90};
  }
  function invalidateRecommendation(clear=true){notice='';cancelRemoteJob();recVersion++;jobId=null;clearTimeout(jobTimer);clearInterval(jobClockTimer);jobSnapshot=null;state.recBusy=false;state.recommended=false;state.added=false;state.candidateId=null;recResult=null;lastMapKey='';updateMap();updateCaption();if(!clear){document.querySelectorAll('.results-heading,.candidate-list,.inserted-summary,.job-progress,.empty-result').forEach(e=>e.remove());const submit=$('activity-form')?.querySelector('[type="submit"]');if(submit){submit.disabled=false;submit.innerHTML='推荐场所 '+icon('arrow');}}}
  async function searchPlaces(query){
    const seq=++searchId,target=$('search-suggestions');if(!target)return;if(query.trim().length<1){target.innerHTML='';return;}
    target.innerHTML='<span class="search-status">搜索中…</span>';
    try{const r=await jsonRequest('/api/travel_activity_poi_candidates',{city:'南京',lat:32.045,lng:118.78,query:query.trim(),limit:6});if(seq!==searchId||!$('search-suggestions'))return;searchResults=r.places||r.candidates||[];
      $('search-suggestions').innerHTML=searchResults.length?searchResults.map((x,i)=>`<button type="button" data-search-result="${i}">${icon('locate')}<span>${esc(placeName(x.name))}</span><small>${x.distance_m?num(x.distance_m/1000)+' km':''}</small></button>`).join(''):'<span class="search-status">没有找到匹配场所，可换一个关键词。</span>';
      document.querySelectorAll('[data-search-result]').forEach(b=>b.addEventListener('click',()=>{state.recQuery=placeName(searchResults[Number(b.dataset.searchResult)].name);invalidateRecommendation(false);$('activity-search').value=state.recQuery;$('search-suggestions').innerHTML='';}));
      setConnection(true);
    }catch(e){if(seq===searchId&&$('search-suggestions'))$('search-suggestions').innerHTML='<span class="search-status">'+esc(e.message)+'</span>';}
  }
  async function changeMode(index,mode,{story=false}={}){
    if(state.modes[index]===mode||state.routeBusy)return;if(!story){pause();demoPrepared=false;}
    const previous=day,modes=[...state.modes];state.modes[index]=mode;state.routeBusy=true;notice='';renderPanel(true);
    const id=++requestId;routeController?.abort();routeController=new AbortController();
    const stops=previous.stops.map((s,i)=>({...s,mode:i?MODE[state.modes[i-1]].api:'foot'}));
    try{const next=await jsonRequest('/api/showcase/route-chain',{city:'南京',stops},routeController.signal);if(id!==requestId)return;
      if(!next.legs?.length)throw new Error('没有返回可用路线');
      day=displayDay(next);comparison={before:previous.totalCarbonG,after:day.totalCarbonG,beforeTime:previous.totalTravelMin,afterTime:day.totalTravelMin,index};
      tween('day-carbon',day.totalCarbonG,previous.totalCarbonG);tween('day-time',day.totalTravelMin,previous.totalTravelMin);
      day.legs.forEach((l,i)=>tween('leg-'+i,l.carbonG,previous.legs[i]?.carbonG));routeVersion++;state.activeLeg=index;state.routeBusy=false;lastMapKey='';setConnection(true);render();frameScene(700);
    }catch(e){if(id!==requestId)return;state.modes=modes;state.routeBusy=false;if(e.name!=='AbortError'){notice=e.message;setConnection(false);}renderPanel(true);}
  }
  async function recommend(){
    demoPrepared=false;
    if(state.recBusy)return;if(!Number.isFinite(state.stayMin)||state.stayMin<1||state.stayMin>240){notice='请输入1—240分钟的停留时间';renderPanel(true);return;}
    if(state.stayMin>=gapAnchors().minutes){notice='停留时间已用完 '+gapAnchors().minutes+' 分钟空档，请缩短停留并留出通行时间。';renderPanel(true);return;}
    if(state.recSearch&&!state.recQuery.trim()){notice='请输入活动或地点关键词';renderPanel(true);return;}
    const version=++recVersion;jobStartedAt=performance.now();jobSnapshot={status:'running',stage:'正在准备本次行程条件',progress:0};clearInterval(jobClockTimer);jobClockTimer=setInterval(paintJobProgress,500);state.recBusy=true;state.recommended=false;state.added=false;state.candidateId=null;notice='';recResult=null;render();
    const anchor=gapAnchors();
    try{const r=await jsonRequest('/api/showcase/recommend-extra',{city:'南京',from:anchor.from,to:anchor.to,activityType:state.recSearch?'keyword':state.recType,query:state.recSearch?state.recQuery.trim():'',stayMin:state.stayMin,travelMode:state.recMode,limit:6,maxCandidates:6,fast:true,async:true});
      if(version!==recVersion){if(r.jobId)cancelRemoteJob(r.jobId);return;}
      if(r.jobId){jobId=r.jobId;pollJob(version);}else finishRecommendation(r.result||r,version);
      setConnection(true);
    }catch(e){if(version!==recVersion)return;state.recBusy=false;notice=e.message;clearInterval(jobClockTimer);setConnection(false);renderPanel(true);}
  }
  async function pollJob(version){
    const current=jobId;if(!current)return;
    try{const r=await jsonRequest('/api/showcase/jobs/'+encodeURIComponent(current));if(jobId!==current||version!==recVersion)return;
      jobSnapshot=r;paintJobProgress();
      if(r.status==='completed'){finishRecommendation(r.result,version);return;}
      if(r.status==='cancelled'){state.recBusy=false;clearInterval(jobClockTimer);jobId=null;notice=r.backendStillRunning?'已取消等待，当前地图请求正在结束。':'已取消本次计算';renderPanel(true);return;}
      if(r.status==='failed'||r.status==='timeout'||r.status==='timed_out'){
        if(r.partialResult?.candidates?.length){notice='部分路线未完成，已保留 '+r.partialResult.candidates.length+' 个验证通过的地点。';finishRecommendation(r.partialResult,version);return;}
        throw new Error(r.error||r.message||'本次计算未能完成，可调整条件重试');
      }
      jobTimer=setTimeout(()=>pollJob(version),800);
    }catch(e){if(version!==recVersion)return;state.recBusy=false;clearInterval(jobClockTimer);notice=e.message;jobId=null;renderPanel(true);}
  }
  function finishRecommendation(result,version=recVersion){
    if(version!==recVersion)return;
    jobId=null;clearInterval(jobClockTimer);jobSnapshot=null;state.recBusy=false;recResult=result;state.recommended=true;state.recEditing=false;state.added=false;state.candidateId=null;routeVersion++;lastMapKey='';render();frameScene(1100);
  }
  function insertCandidate(id){pause();demoPrepared=false;state.candidateId=id;state.added=true;routeVersion++;routeLegs().forEach((l,i)=>{if(Number.isFinite(l.carbonG))tween('rec-leg-'+i,l.carbonG,0);});render();frameScene(1000);}
  function lineBounds(coords){return coords.reduce((b,c)=>b.extend(c),new maplibregl.LngLatBounds(coords[0],coords[0]));}
  function cameraPadding(){const w=innerWidth,h=innerHeight;return w<=680?{left:20,right:20,top:100,bottom:Math.round(h*.53)}:{left:w<=1100?408:430,right:120,top:130,bottom:170};}
  function frameCoordinates(coords,duration=1100,maxZoom=15.3){if(!ready||!coords?.length)return;const box=lineBounds(coords),sw=box.getSouthWest(),ne=box.getNorthEast(),center=[(sw.lng+ne.lng)/2,(sw.lat+ne.lat)/2],padding=cameraPadding();const width=Math.max(150,innerWidth-padding.left-padding.right),height=Math.max(150,innerHeight-padding.top-padding.bottom),x=Math.max(.002,ne.lng-sw.lng)/360*512,y=Math.max(.002,ne.lat-sw.lat)/Math.cos(center[1]*Math.PI/180)/360*512,zoom=Math.min(maxZoom,Math.log2(Math.min(width/x,height/y))-.4);map.easeTo({center,zoom,padding,pitch:state.flat?0:55,bearing:-25,duration:reduced?0:duration,easing:t=>t<.5?4*t*t*t:1-Math.pow(-2*t+2,3)/2});}
  function frameScene(duration=2200){
    if(!ready||state.scene==='records')return;if(state.scene==='convenience'){convenience?.frame();return;}if(state.scene==='allocation'){focusAllocationCamera(state.allocationStage,duration);return;}
    const coords=day.stops.map(s=>s.coordinate);if(state.scene==='recommendation'&&state.added)routeLegs().forEach(l=>coords.push(...l.coordinates));else if(state.scene==='recommendation'&&state.recommended)activity()?.candidates?.slice(0,6).forEach(c=>coords.push(c.coordinate));frameCoordinates(coords,duration);
  }
  function setScene(scene,{manual=true,camera=true}={}){
    if(manual){pause();budgetPopup?.remove();}const previousScene=state.scene;if(previousScene==='allocation'&&scene!=='allocation'){completeAllocationMap();budgetPopup?.remove();}state.scene=scene;state.budgetView=scene==='allocation'?'city':'district';state.budgetLayer=scene==='allocation';state.flat=scene==='convenience';if(scene==='allocation'&&manual){allocationStarted=performance.now();allocationAuto=false;state.allocationStage=1;resetAllocationMapAnimation(1);}
    if(scene==='chain'&&previousScene!=='chain'){tween('day-carbon',day.totalCarbonG,0,1800,true);tween('day-time',day.totalTravelMin,0,1800,true);day.legs.forEach((l,i)=>tween('leg-'+i,l.carbonG,0,1800,true));}
    if(manual){demoPrepared=false;playbackStart=0;playbackEnd=STORY.total;}if(manual)state.time=scene==='convenience'?0:scene==='allocation'?STORY.weightStart:scene==='chain'?STORY.allocationEnd+1:scene==='records'?STORY.recordsAt:STORY.chainEnd+1;lastPanelKey='';lastMapKey='';if(scene==='convenience'&&manual)convenience?.present(null);render();if(camera)frameScene();
  }
  function updateCaption(){
    document.body.dataset.scene=state.scene;
    if(state.scene==='records')return;
    const captions={convenience:['沿途，还能去哪里','出发地与目的地 · 时间约束 · 活动偏好'],allocation:[['生活所需，连接一座城','从生活便利水平，到预算分配','从生活便利水平，到预算分配'][state.allocationStage],['工作、学习、就医，以及日常的片刻','生活条件 → 分配权重 → 人均日碳预算','生活条件 → 分配权重 → 人均日碳预算'][state.allocationStage]],chain:['每一种出行，都有自己的碳足迹','改变交通方式，让时间与排放一起变化'],recommendation:['让日常，多一种可能','从既定日程出发，找到时间预算内的活动场所']};
    $('scene-title').textContent=captions[state.scene][0];$('scene-subtitle').textContent=captions[state.scene][1];$('map-region').textContent=state.scene==='allocation'&&state.budgetView==='city'?(state.allocationStage===0?'居住地生活圈':state.allocationStage===1?'居住地生活圈':'南京市域'):'鼓楼 — 秦淮';$('selected-scheme-label').textContent=state.scene==='convenience'?'ACTIVITY SPACE':state.scene==='allocation'?'CARBON ALLOCATION':state.scene==='chain'?'ACTIVITY CHAIN':'ACTIVITY PLANNING';
    $('legend').classList.toggle('budget-legend',state.budgetLayer);
    $('legend').innerHTML=state.budgetLayer?`<div class="legend-heading">人均日碳预算 <span>g / 人·天</span></div><div class="budget-legend-list">${actualBudget.legend.map(x=>`<div><i style="background:${x.color}"></i><span>${esc(x.label)}</span></div>`).join('')}</div>`:Object.entries(MODE).filter(([key])=>routeLegs().some(l=>modeInfo(l).key===key)).map(([key,v])=>`<div class="mode-legend"><i style="background:${v.color}"></i>${v.label}</div>`).join('');
    $('budget-layer-toggle').setAttribute('aria-pressed',String(state.budgetLayer));$('view-toggle').setAttribute('aria-label',state.flat?'切换三维':'切换俯视');$('budget-view-toggle').setAttribute('aria-pressed',String(state.budgetView==='city'));$('motion-toggle').setAttribute('aria-pressed',String(state.motion));$('motion-toggle').title=state.motion?'暂停路线动效':'播放路线动效';$('map-source-note').textContent=state.scene==='allocation'&&state.allocationStage>0?'南京市域 · 建成区三维概览':'南京中心城区 · © OpenStreetMap';
  }
  function render(){
    const records=state.scene==='records';document.body.dataset.scene=state.scene;
    $('history-stage').hidden=!records;
    if(records){
      convenience?.setVisible(false);updatePlayback();
      const frame=$('history-frame');
      if(!frame.getAttribute('src'))frame.src='./history.html?showcase=1';
      frame.contentWindow?.postMessage({type:'iatp-showcase',time:state.time},location.origin);
      return;
    }
    renderPanel();updateCaption();updatePlayback();updateMap();updateAllocationRibbon();convenience?.setVisible(state.scene==='convenience');
  }
  function updatePlayback(){document.body.classList.toggle('playback-paused',!state.playing);$('timeline').value=String(state.time);$('timeline').style.setProperty('--progress',state.time/STORY.total*100+'%');$('elapsed').textContent=state.time.toFixed(1).padStart(4,'0');$('play-glyph').hidden=state.playing;$('pause-glyph').hidden=!state.playing;$('play-toggle').setAttribute('aria-label',state.playing?'暂停展示':'播放展示');}
  let playbackStart=0,playbackEnd=STORY.total,demoPrepared=false;
  function pause(){pauseAllocationMap();allocationAuto=false;state.playing=false;lastFrame=0;cancelAnimationFrame(frameId);map?.stop();updatePlayback();}
  function applyPreparedBike(animate=false){
    const previous=day;
    const first=displayDay(cachedChains.modes.bike).legs[0];
    const second=initialDay.legs[1];
    day={...initialDay,legs:[first,second]};
    day.totalCarbonG=first.carbonG+second.carbonG;
    day.totalTravelMin=first.timeMin+second.timeMin;
    day.totalDistanceKm=first.distanceKm+second.distanceKm;
    state.modes=['bike',normalize(second.requestedMode||second.mode)];
    comparison={before:initialDay.totalCarbonG,after:day.totalCarbonG,beforeTime:initialDay.totalTravelMin,afterTime:day.totalTravelMin,index:0};
    storyEvents.add('bike');
    const duration=animate?1350:0;
    tween('day-carbon',day.totalCarbonG,animate?previous.totalCarbonG:day.totalCarbonG,duration,true);
    tween('day-time',day.totalTravelMin,animate?previous.totalTravelMin:day.totalTravelMin,duration,true);
    day.legs.forEach((leg,index)=>tween('leg-'+index,leg.carbonG,animate?previous.legs[index].carbonG:leg.carbonG,duration,true));
    routeVersion++;lastMapKey='';
  }
  function resetStory(){
    budgetPopup?.remove();
    requestId++;routeController?.abort();state.routeBusy=false;cancelRemoteJob();recVersion++;jobId=null;clearTimeout(jobTimer);clearInterval(jobClockTimer);jobSnapshot=null;storyEvents.clear();comparison=null;notice='';state.allocationStage=1;allocationAuto=false;resetAllocationMapAnimation(1);
    convenience?.prepareDemo();demoPrepared=true;
    day=displayDay(JSON.parse(JSON.stringify(initialDay)));state.modes=day.legs.map(l=>normalize(l.requestedMode||l.mode));state.recBusy=false;state.recommended=false;state.added=false;state.recEditing=true;configureStoryRecommendation();state.candidateId=null;recResult=null;routeVersion++;lastPanelKey='';lastMapKey='';
    tween('day-carbon',day.totalCarbonG,day.totalCarbonG,0);tween('day-time',day.totalTravelMin,day.totalTravelMin,0);day.legs.forEach((l,i)=>tween('leg-'+i,l.carbonG,l.carbonG,0));
  }
  function seek(time){
    resetStory();if(time>=STORY.changeMode)applyPreparedBike();
    applyTime(time);seekAllocationMap(time);
  }
  function applyTime(time){
    state.time=Math.max(0,Math.min(STORY.total,time));const next=(state.time<STORY.weightStart||playbackEnd===STORY.weightStart&&state.time===STORY.weightStart)?'convenience':(state.time<STORY.allocationEnd||playbackEnd===STORY.allocationEnd&&state.time===STORY.allocationEnd)?'allocation':state.time<STORY.chainEnd?'chain':state.time<STORY.recordsAt?'recommendation':'records';if(next!==state.scene)setScene(next,{manual:false});
    if(state.playing&&state.time>=STORY.changeMode&&!storyEvents.has('bike'))applyPreparedBike(true);
    if(state.time>=STORY.chainEnd&&!storyEvents.has('rec')){storyEvents.add('rec');invalidateRecommendation();configureStoryRecommendation();state.recEditing=true;state.recommended=false;state.added=false;recResult=null;lastPanelKey='';}
    if(state.scene==='recommendation'){
      const recommended=state.time>=STORY.resultsAt,added=state.time>=STORY.insertAt,changed=state.recommended!==recommended||state.added!==added;
      state.recommended=recommended;state.added=added;state.recEditing=!recommended;recResult=recommended?seed:null;state.candidateId=added?seed.candidates[0]?.id:null;
      if(changed){routeVersion++;lastPanelKey='';lastMapKey='';if(added){routeLegs().forEach((l,i)=>Number.isFinite(l.carbonG)&&tween('rec-leg-'+i,l.carbonG,0));frameScene();}}
    }
    if(state.scene==='allocation')setAllocationStage(state.time<STORY.distributionStart?1:2,false);
    if(state.scene==='convenience')convenience?.present(state.time);
    render();
  }
  function play(){
    if(!ready||state.playing)return;
    cancelAnimationFrame(frameId);
    if(!demoPrepared||state.time>=playbackEnd)seek(playbackStart);
    frameScene(900);state.playing=true;resumeAllocationMap();lastFrame=0;updatePlayback();
    frameId=requestAnimationFrame(tick);
  }
  function tick(now){if(!state.playing)return;if(lastFrame)state.time=Math.min(playbackEnd,state.time+Math.max(0,(now-lastFrame)/1000));lastFrame=now;applyTime(state.time);if(state.time>=playbackEnd){pause();return;}frameId=requestAnimationFrame(tick);}
  function pointAt(coords,fraction){
    let cached=lineCache.get(coords);if(!cached){const lengths=[0];for(let i=1;i<coords.length;i++){const a=coords[i-1],b=coords[i],dx=(b[0]-a[0])*Math.cos((a[1]+b[1])*.5*Math.PI/180),dy=b[1]-a[1];lengths.push(lengths[i-1]+Math.hypot(dx,dy));}cached={lengths,total:lengths[lengths.length-1]};lineCache.set(coords,cached);}
    const distance=cached.total*Math.max(0,Math.min(1,fraction));let low=1,high=coords.length-1;while(low<high){const mid=(low+high)>>1;if(cached.lengths[mid]<distance)low=mid+1;else high=mid;}const i=low,a=coords[i-1]||coords[0],b=coords[i]||a,t=(distance-cached.lengths[i-1])/(cached.lengths[i]-cached.lengths[i-1]||1);return {coordinate:[a[0]+(b[0]-a[0])*t,a[1]+(b[1]-a[1])*t],angle:Math.atan2(b[0]-a[0],b[1]-a[1])*180/Math.PI};
  }
  function vehiclePath(leg){
    const parts=leg.routeParts;if(!Array.isArray(parts)||!parts.length)return leg.coordinates;
    if(vehiclePathCache.has(parts))return vehiclePathCache.get(parts);
    const mode=modeInfo(leg).key,paths=parts.filter(p=>normalize(p.type)===mode&&p.coords?.length>=2).map(p=>p.coords.map(([lat,lon])=>[lon,lat]));
    let best=null,bestLength=-1;for(const path of paths){let length=0;for(let i=1;i<path.length;i++)length+=Math.hypot((path[i][0]-path[i-1][0])*Math.cos(path[i][1]*Math.PI/180),path[i][1]-path[i-1][1]);if(length>bestLength){best=path;bestLength=length;}}
    vehiclePathCache.set(parts,best||leg.coordinates);return best||leg.coordinates;
  }
  function featureCollection(features){return {type:'FeatureCollection',features};}
  function updateMap(){
    if(!ready)return;const legs=routeLegs(),key=[state.scene,state.budgetLayer,state.budgetView,state.activeLeg,routeVersion,state.recommended,state.added,state.candidateId,state.allocationStage].join('|');if(key===lastMapKey)return;lastMapKey=key;
    const features=legs.filter(l=>l.coordinates?.length>=2).map((l,index)=>({type:'Feature',geometry:{type:'LineString',coordinates:l.coordinates},properties:{index,color:modeInfo(l).color}}));map.getSource('routes').setData(featureCollection(features));
    map.setLayoutProperty('city-budget-3d','visibility',state.scene==='convenience'?'none':'visible');
    ['route-halo','route-casing','route-line','route-flow'].forEach(id=>map.setLayoutProperty(id,'visibility',['allocation','convenience'].includes(state.scene)?'none':'visible'));
    map.setLayoutProperty('budget-fill','visibility',state.budgetLayer?'visible':'none');map.setPaintProperty('budget-fill','fill-color',['get','budget_color']);map.setPaintProperty('budget-fill','fill-opacity',state.budgetView==='city'?1:.35);
    ['buildings-3d','city-budget-3d'].forEach(id=>map.setPaintProperty(id,'fill-extrusion-color',state.budgetLayer?budgetColorExpression():'#d0d5d7'));
    if(map.getLayer('city-buildings'))map.setPaintProperty('city-buildings','fill-extrusion-color',state.budgetLayer?cityBudgetColorExpression():'#d0d5d7');
    const showNodes=['chain','recommendation'].includes(state.scene);nodeMarkers.forEach(m=>m.getElement().style.display=showNodes?'':'none');placeMarkers.forEach(m=>m.getElement().style.display=state.budgetView==='city'?'none':'');
    [...routeMarkers,...carbonMarkers,...candidateMarkers].forEach(m=>m.remove());routeMarkers=[];carbonMarkers=[];candidateMarkers=[];extraMarker?.remove();extraMarker=null;
    if(['chain','recommendation'].includes(state.scene))legs.forEach((l,i)=>{
      if(!l.coordinates?.length)return;const m=modeInfo(l),el=document.createElement('div');el.className='moving-vehicle';el.style.setProperty('--mode-color',m.color);el.innerHTML=`<span class="vehicle-aura"></span><span class="vehicle-body">${icon(m.key)}</span><i></i>`;el.setAttribute('aria-label',m.label+'移动图标');routeMarkers.push(new maplibregl.Marker({element:el,anchor:'center'}).setLngLat(l.coordinates[0]).addTo(map));
      const label=document.createElement('button');label.type='button';label.className='leg-carbon-label';label.style.setProperty('--mode-color',m.color);label.setAttribute('aria-label',`${l.from}到${l.to}的碳排放`);
      const numberKey=(state.scene==='recommendation'&&state.added?'rec-leg-':'leg-')+i,change=state.scene==='chain'&&comparison?.index===i?comparison.after-comparison.before:null;
      label.innerHTML=`<span class="carbon-leg-name">0${i+1} <b>${esc(placeName(l.from))} → ${esc(placeName(l.to))}</b></span><span class="carbon-reading">${Number.isFinite(l.carbonG)?numberHTML(numberKey,l.carbonG,1):'<span>—</span>'}<small>g CO₂</small></span><span class="carbon-leg-meta">${icon(m.key)} ${esc(m.label)} · ${num(l.timeMin)} min${l.walkFallback?'<small class="fallback-caption">接驳估算</small>':''}${change!==null?`<em class="${change<=0?'down':'up'}">${change<=0?'↓':'↑'} ${num(Math.abs(change))} g</em>`:''}</span>`;
      label.addEventListener('click',()=>{pause();state.activeLeg=i;renderPanel(true);frameCoordinates(l.coordinates,800);});
      carbonMarkers.push(new maplibregl.Marker({element:label,anchor:'top-left',offset:[18,-120]}).setLngLat(pointAt(l.coordinates,i===0?.43:.59).coordinate).addTo(map));
    });
    if(state.scene==='recommendation'&&state.recommended)activity()?.candidates?.slice(0,6).forEach((c,i)=>{if(state.added&&c.id===candidate()?.id)return;const el=document.createElement('button');el.className='poi-candidate';el.textContent=String(i+1);el.title=c.name;el.setAttribute('aria-label','加入'+c.name);el.addEventListener('click',()=>insertCandidate(c.id));candidateMarkers.push(new maplibregl.Marker({element:el}).setLngLat(c.coordinate).addTo(map));});
    if(state.scene==='recommendation'&&state.added&&candidate()){const c=candidate(),el=document.createElement('div');el.className='map-stop extra-stop';el.innerHTML=`<span class="map-stop-dot">${icon('plus')}</span><span><b>${esc(placeName(c.name))}</b><small>${clock(c.arrivalAtActivityMinutes)}—${clock(c.departureFromActivityMinutes)}</small></span>`;extraMarker=new maplibregl.Marker({element:el,anchor:'bottom-left',offset:[-12,-10]}).setLngLat(c.coordinate).addTo(map);}
    syncAllocationMap(true);syncCityBudgetStates();requestAnimationFrame(positionCarbonLabels);
  }
  function positionCarbonLabels(){
    if(!ready||!carbonMarkers.length)return;
    const left=innerWidth>680?document.querySelector('.data-panel').getBoundingClientRect().right+15:15,right=innerWidth-80,top=125,bottom=innerHeight-160;
    const options=carbonMarkers.map(marker=>{const el=marker.getElement(),width=el.offsetWidth,height=el.offsetHeight,p=map.project(marker.getLngLat());return [[18,-height-24],[-width-18,-height-24],[18,24],[-width-18,24]].map((offset,preference)=>{const rect={x:p.x+offset[0],y:p.y+offset[1],width,height},outside=Math.max(0,left-rect.x)*height+Math.max(0,rect.x+width-right)*height+Math.max(0,top-rect.y)*width+Math.max(0,rect.y+height-bottom)*width;return {offset,rect,score:outside*4+preference*20};});});
    if(options.length===1){options[0].sort((a,b)=>a.score-b.score);carbonMarkers[0].setOffset(options[0][0].offset);return;}
    let best=null;for(const a of options[0])for(const b of options[1]){const x=Math.max(0,Math.min(a.rect.x+a.rect.width,b.rect.x+b.rect.width)-Math.max(a.rect.x,b.rect.x)),y=Math.max(0,Math.min(a.rect.y+a.rect.height,b.rect.y+b.rect.height)-Math.max(a.rect.y,b.rect.y)),score=a.score+b.score+x*y*12;if(!best||score<best.score)best={a,b,score};}
    carbonMarkers[0].setOffset(best.a.offset);carbonMarkers[1].setOffset(best.b.offset);
  }
  function motionLoop(now){
    animateNumbers();updateAllocationAnimation(now);if(ready&&state.scene!=='allocation'&&!document.hidden){const legs=routeLegs();if(state.motion){if(!animationStart)animationStart=now;const seconds=state.time;routeMarkers.forEach((marker,i)=>{const l=legs[i];if(!l?.coordinates?.length)return;const speed=modeInfo(l).key==='walk'?18:modeInfo(l).key==='bike'?14:11,part=((seconds/speed)+(i*.44))%1,p=pointAt(vehiclePath(l),part);marker.setLngLat(p.coordinate);const target=p.angle-map.getBearing(),previous=Number.isFinite(marker.headingValue)?marker.headingValue:target,difference=((target-previous+180)%360+360)%360-180;marker.headingValue=target;marker.getElement().style.setProperty('--heading',marker.headingValue+'deg');marker.getElement().style.setProperty('--travel-opacity',String(Math.max(0,Math.min(1,part/.035,(1-part)/.035))));});
        if(now-lastFlow>80&&map.getLayer('route-flow')){lastFlow=now;const phase=(seconds*.16)%1;map.setPaintProperty('route-flow','line-gradient',['interpolate',['linear'],['%',['+',['line-progress'],phase],1],0,'rgba(255,255,255,0)',.30,'rgba(255,255,255,0)',.39,'rgba(255,255,255,.95)',.43,'rgba(255,255,255,.95)',.53,'rgba(255,255,255,0)',1,'rgba(255,255,255,0)']);}}
    }motionFrame=requestAnimationFrame(motionLoop);
  }
  function setupOverlays(){
    map.addSource('routes',{type:'geojson',lineMetrics:true,data:featureCollection([])});
    const layout={'line-cap':'round','line-join':'round'},offset=['case',['==',['get','index'],0],-2,2];
    map.addLayer({id:'route-halo',type:'line',source:'routes',layout,paint:{'line-color':['get','color'],'line-width':16,'line-opacity':.12,'line-blur':4,'line-offset':offset}});
    map.addLayer({id:'route-casing',type:'line',source:'routes',layout,paint:{'line-color':'#ffffff','line-width':9,'line-opacity':.95,'line-offset':offset}});
    map.addLayer({id:'route-line',type:'line',source:'routes',layout,paint:{'line-color':['get','color'],'line-width':5.5,'line-opacity':1,'line-offset':offset}});
    map.addLayer({id:'route-flow',type:'line',source:'routes',layout,paint:{'line-width':2.2,'line-opacity':.8,'line-offset':offset,'line-gradient':['interpolate',['linear'],['line-progress'],0,'rgba(255,255,255,0)',.5,'rgba(255,255,255,.95)',1,'rgba(255,255,255,0)']}});
    day.stops.slice(0,2).forEach((s,i)=>{const el=document.createElement('button');el.type='button';el.className='map-stop';el.innerHTML=`<span class="map-stop-dot">${i===0?'<svg class="icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6"><path d="m3 11 9-8 9 8M6 9v12h12V9M10 21v-7h4v7"/></svg>':icon('locate')}</span><span><b>${esc(s.label)}</b><small>${i===0?'日预算 '+num(homePoint().budget_g)+' g':'09:00—17:00'}</small></span>`;el.addEventListener('click',()=>{pause();frameCoordinates([s.coordinate],850,16);});nodeMarkers.push(new maplibregl.Marker({element:el,anchor:'bottom-left',offset:[-12,-9]}).setLngLat(s.coordinate).addTo(map));});
    [['新街口',[118.784,32.04]],['鼓楼',[118.7805,32.0587]],['五台山',[118.769,32.0537]],['上海路',[118.7785,32.048]],['朝天宫',[118.7737,32.0305]],['大行宫',[118.7972,32.04]]].forEach(([label,coordinate])=>{const el=document.createElement('span');el.className='place-name';el.textContent=label;placeMarkers.push(new maplibregl.Marker({element:el}).setLngLat(coordinate).addTo(map));});
    setupAllocationMap();
    map.on('click',e=>{if(state.scene!=='allocation'||state.allocationStage!==2||!state.budgetLayer)return;pause();const features=map.queryRenderedFeatures(e.point,{layers:['buildings-3d','city-budget-3d','budget-fill']}),f=features.find(x=>x.layer.id==='buildings-3d')||features[0];if(!f)return;state.selection={...f.properties};selectionVersion++;renderPanel(true);const el=document.createElement('div');el.className='budget-popup';el.innerHTML=`<small>所在地人均日碳预算</small><strong>${Number.isFinite(f.properties.budget_g)?fmtBudget(f.properties.budget_g)+' g':'暂无数据'}</strong>`;budgetPopup?.remove();budgetPopup=new maplibregl.Popup({maxWidth:'250px'}).setLngLat(e.lngLat).setDOMContent(el).addTo(map);});
  }
  function localStyle() {
    const roadWidth=['interpolate',['linear'],['zoom'],13,['match',['get','class'],['primary','secondary'],2.3,['tertiary'],1.5,.7],16,['match',['get','class'],['primary','secondary'],12,['tertiary'],8,3]];
    return {version:8,name:'Nanjing local city model',glyphs:'https://tiles.openfreemap.org/fonts/{fontstack}/{range}.pbf',light:{anchor:'viewport',color:'#ffffff',intensity:.48,position:[1.5,205,35]},sources:{
      buildings:{type:'geojson',data:'./data/buildings.geojson'},roads:{type:'geojson',data:'./data/roads.geojson'},landcover:{type:'geojson',data:'./data/landcover.geojson'},water:{type:'geojson',data:'./data/water.geojson'},citybudget:{type:'geojson',data:'./data/city-budget-lod.geojson'},budget:{type:'geojson',data:actualBudgetCells}
    },layers:[
      {id:'base',type:'background',paint:{'background-color':'#e8ece9'}},
      {id:'green-areas',type:'fill',source:'landcover',paint:{'fill-color':'#c9d9c9','fill-opacity':.8}},
      {id:'water-areas',type:'fill',source:'water',filter:['==',['geometry-type'],'Polygon'],paint:{'fill-color':'#b8d6dc'}},
      {id:'water-lines',type:'line',source:'water',filter:['==',['geometry-type'],'LineString'],paint:{'line-color':'#b8d6dc','line-width':4}},
      {id:'budget-fill',type:'fill',source:'budget',layout:{visibility:'none'},paint:{'fill-color':['get','budget_color'],'fill-opacity':.62}},
      {id:'road-casings',type:'line',source:'roads',layout:{'line-cap':'round','line-join':'round'},paint:{'line-color':'#d5dcd9','line-width':['interpolate',['linear'],['zoom'],13,['match',['get','class'],['primary','secondary'],3.9,['tertiary'],3.1,2.3],16,['match',['get','class'],['primary','secondary'],13.6,['tertiary'],9.6,4.6]]}},
      {id:'road-surfaces',type:'line',source:'roads',layout:{'line-cap':'round','line-join':'round'},paint:{'line-color':'#fcfdfb','line-width':roadWidth}},
      {id:'city-budget-3d',type:'fill-extrusion',source:'citybudget',maxzoom:13.5,paint:{'fill-extrusion-color':budgetColorExpression(),'fill-extrusion-height':['get','overview_height'],'fill-extrusion-opacity':1,'fill-extrusion-vertical-gradient':false}},
      {id:'buildings-3d',type:'fill-extrusion',source:'buildings',minzoom:12.5,paint:{'fill-extrusion-color':budgetColorExpression(),'fill-extrusion-height':['get','height'],'fill-extrusion-base':['get','min_height'],'fill-extrusion-opacity':1,'fill-extrusion-vertical-gradient':false}}
    ]};
  }

  async function toggleCityContext(automatic=false) {
    if(!ready||cityLoading) return;
    pause();
    const button=$('city-map-toggle');
    try {
      if(!map.getSource('city-context')) {
        cityLoading=true;button.disabled=true;
        if(!automatic) {$('map-status').hidden=false;$('map-status').textContent='正在连接南京全市底图…';}
        const controller=new AbortController();
        const timeout=setTimeout(()=>controller.abort(),9000);
        let response,meta;
        try {response=await fetch('https://tiles.openfreemap.org/planet',{signal:controller.signal});if(!response.ok) throw new Error('底图服务暂不可用');meta=await response.json();}
        finally {clearTimeout(timeout);}
        map.addSource('city-context',{type:'vector',tiles:meta.tiles,minzoom:meta.minzoom||0,maxzoom:meta.maxzoom||14,attribution:'<a href="https://openfreemap.org/" target="_blank" rel="noopener">OpenFreeMap</a> · <a href="https://www.openmaptiles.org/" target="_blank" rel="noopener">© OpenMapTiles</a>'});
        const core={type:'Polygon',coordinates:[[[118.765,32.025],[118.805,32.025],[118.805,32.060],[118.765,32.060],[118.765,32.025]]]};
        const layers=[
          {id:'city-green',type:'fill',source:'city-context','source-layer':'landcover',paint:{'fill-color':'#d1dece','fill-opacity':.65}},
          {id:'city-water',type:'fill',source:'city-context','source-layer':'water',paint:{'fill-color':'#b8d6dc'}},
          {id:'city-roads',type:'line',source:'city-context','source-layer':'transportation',layout:{'line-cap':'round','line-join':'round'},paint:{'line-color':'#fdfefb','line-width':['interpolate',['linear'],['zoom'],10,.7,14,3,17,11]}},
          {id:'city-buildings',type:'fill-extrusion',source:'city-context','source-layer':'building',minzoom:14,filter:['all',['!',['has','hide_3d']],['!',['within',core]]],paint:{'fill-extrusion-height':['coalesce',['get','render_height'],12],'fill-extrusion-base':['coalesce',['get','render_min_height'],0],'fill-extrusion-color':state.budgetLayer?cityBudgetColorExpression():'#c8d1d2','fill-extrusion-opacity':1,'fill-extrusion-vertical-gradient':false}},
          {id:'city-place-names',type:'symbol',source:'city-context','source-layer':'place',filter:['!',['within',core]],minzoom:9,layout:{'text-field':['coalesce',['get','name:zh'],['get','name']],'text-font':['Noto Sans Regular'],'text-size':['interpolate',['linear'],['zoom'],9,11,15,14],'text-max-width':8},paint:{'text-color':'#647d7e','text-halo-color':'#f1f4ef','text-halo-width':1.5}}
        ];
        layers.forEach(layer=>map.addLayer(layer,'green-areas'));
      }
      cityContext=!cityContext;
      cityLayerIds.forEach(id=>map.setLayoutProperty(id,'visibility',cityContext?'visible':'none'));
      map.setMinZoom(7);
      
      button.setAttribute('aria-pressed',String(cityContext));
      button.title=cityContext?'关闭在线背景':'在线地图背景';
      $('map-source-note').textContent=cityContext?'南京 · 在线地图背景':'南京中心城区 · © OpenStreetMap';
      syncCityBudgetStates();
      $('map-status').hidden=true;
      if(!cityContext) frameScene();
    } catch(error) {
      if(!automatic) {$('map-status').hidden=false;$('map-status').textContent='在线背景暂未连接。';setTimeout(()=>{$('map-status').hidden=true;},4500);}
      console.warn('全市底图连接：',error.message);
    } finally {cityLoading=false;button.disabled=false;}
  }

  function bindEvents(){
    document.querySelectorAll('[data-scene]').forEach(b=>b.addEventListener('click',()=>setScene(b.dataset.scene)));
    $('play-toggle').addEventListener('click',()=>state.playing?pause():play());$('replay').addEventListener('click',()=>{pause();playbackStart=0;playbackEnd=STORY.total;seek(0);play();});
    $('timeline').addEventListener('input',e=>{const target=Number(e.target.value);pause();playbackStart=0;playbackEnd=STORY.total;seek(target);});
    $('fullscreen').addEventListener('click',async()=>{try{if(document.fullscreenElement)await document.exitFullscreen();else await document.documentElement.requestFullscreen();}catch{}});
    $('view-toggle').addEventListener('click',()=>{pause();state.flat=!state.flat;if(state.scene==='allocation'&&state.allocationStage===2)focusAllocationCamera(2,850);else map.easeTo({pitch:state.flat?0:55,bearing:state.flat?0:-25,duration:850});updateCaption();});
    $('reset-view').addEventListener('click',()=>{pause();frameScene();});$('budget-layer-toggle').addEventListener('click',()=>{state.budgetLayer=!state.budgetLayer;render();});
    $('budget-view-toggle').addEventListener('click',()=>{pause();state.scene='allocation';state.budgetLayer=true;state.budgetView=state.budgetView==='city'?'district':'city';state.allocationStage=state.budgetView==='city'?2:1;state.time=state.allocationStage===2?STORY.distributionStart:STORY.weightStart;state.flat=false;allocationAuto=false;resetAllocationMapAnimation(state.allocationStage);render();frameScene();});$('city-map-toggle').addEventListener('click',()=>toggleCityContext());
    $('motion-toggle').addEventListener('click',()=>{state.motion=!state.motion;updateCaption();});document.addEventListener('visibilitychange',()=>{if(document.hidden)pause();});
    let resize;addEventListener('resize',()=>{clearTimeout(resize);resize=setTimeout(()=>{map.resize();frameScene(0);},140);});
    addEventListener('keydown',e=>{if(e.code==='Space'&&!['INPUT','SELECT','TEXTAREA','BUTTON'].includes(document.activeElement.tagName)){e.preventDefault();state.playing?pause():play();}});
  }
  async function init(){
    try{
      const names=['actual-budget.json','actual-budget-cells.geojson','live-config.json','default-chain.json','story-extra-activity.json','precomputed-chains.json','allocation-process.json','allocation-map-anchors.json','allocation-contribution-story.json'];
      const responses=await Promise.all(names.map(n=>fetch('./data/'+n)));if(responses.some(r=>!r.ok))throw new Error('展示数据尚未准备完成');
      [actualBudget,actualBudgetCells,config,day,seed,cachedChains,allocationProcess,allocationMapAnchors,contributionStory]=await Promise.all(responses.map(r=>r.json()));day=displayDay(day);initialDay=JSON.parse(JSON.stringify(day));state.modes=day.legs.map(l=>normalize(l.requestedMode||l.mode));config.activityTypes=config.activityTypes||config.activity_types||[];
      if(!config.activityTypes.length)throw new Error('活动类型未能读取');
      if(!cachedChains?.modes?.bike?.legs?.length)throw new Error('演示行程数据未能读取');
      seed.candidates.forEach(c=>{if(!c.beforeLeg||!c.afterLeg)throw new Error('自动演示缺少完整分段计算结果');});
      tween('day-carbon',day.totalCarbonG,0);tween('day-time',day.totalTravelMin,0);day.legs.forEach((l,i)=>tween('leg-'+i,l.carbonG,0));
      configureStoryRecommendation();bindEvents();renderPanel();updateCaption();updatePlayback();
      document.querySelectorAll('[data-icon]').forEach(el=>el.innerHTML=icon(el.dataset.icon));
      if(!window.maplibregl)throw new Error('地图组件未能加载');
      map=new maplibregl.Map({container:'map',style:localStyle(),center:[118.8,31.95],zoom:8,pitch:48,bearing:-16,antialias:true,maxPitch:70,minZoom:7,maxZoom:18.5,localIdeographFontFamily:'Microsoft YaHei',attributionControl:false,canvasContextAttributes:{antialias:true,preserveDrawingBuffer:true}});
      map.addControl(new maplibregl.AttributionControl({compact:true,customAttribution:'<a href="https://www.openstreetmap.org/copyright" target="_blank" rel="noopener">© OpenStreetMap contributors</a>'}),'bottom-right');
      map.once('load',async()=>{
        try {
          setupOverlays();
          convenience=window.createActivitySpaceView({map,panel:$('panel-content'),config,pause,frameCoordinates,
            onChange:()=>{lastPanelKey='';renderPanel(true);},
            replay:()=>{pause();playbackStart=0;playbackEnd=STORY.weightStart;seek(0);play();}});
          await convenience.initialize();
          ready=true;lastPanelKey='';$('map-status').hidden=true;render();frameScene(0);
          motionFrame=requestAnimationFrame(motionLoop);
        }catch(error){$('map-status').hidden=false;$('map-status').textContent=error.message;console.error(error);}
      });
      map.on('dragstart',pause);map.on('zoomstart',e=>{if(e.originalEvent)pause();});map.on('move',()=>{positionCarbonLabels();positionAllocationLabels();});map.on('error',e=>console.warn('地图资源',e.error?.message||e.message));
      map.on('moveend',()=>{const c=map.getCenter();$('map').dataset.renderStatus=JSON.stringify({center:[c.lng,c.lat],zoom:map.getZoom(),pitch:map.getPitch(),scene:state.scene,routeVersion});});
      map.on('idle',()=>{syncCityBudgetStates();const c=map.getCenter();$('map').dataset.renderStatus=JSON.stringify({center:[c.lng,c.lat],zoom:map.getZoom(),pitch:map.getPitch(),features:map.queryRenderedFeatures().length,scene:state.scene,routeVersion});});
      jsonRequest('/api/health').then(()=>setConnection(true)).catch(()=>setConnection(false));
      window.IATPShowcase={play,pause,seek,story:{...STORY},get state(){return {...state,ready}},get day(){return day},get config(){return config},get recommendation(){return recResult},get actualBudget(){return actualBudget},get activitySpace(){return convenience?.result}};
    }catch(e){$('map-status').hidden=false;$('map-status').textContent=e.message;console.error(e);}
  }
  init();
})();
