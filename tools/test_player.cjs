const fs=require('node:fs'),vm=require('node:vm'),assert=require('node:assert/strict');
const path=require('node:path'),root=path.resolve(__dirname,'../editable-app/web');
const data=name=>JSON.parse(fs.readFileSync(path.join(root,'data',name),'utf8'));
const elements=new Map();
function element(id){if(!elements.has(id))elements.set(id,{value:'0',dataset:{},style:{setProperty(){}},classList:{toggle(){}},listeners:{},setAttribute(){},addEventListener(type,fn){this.listeners[type]=fn;}});return elements.get(id);}
let nextFrame=0;const frames=new Map();
const context={console,performance:{now:()=>1000},matchMedia:()=>({matches:false}),document:{body:{dataset:{},classList:{toggle(){}}},getElementById:element,querySelectorAll:()=>[],addEventListener(){},hidden:false},requestAnimationFrame:fn=>{frames.set(++nextFrame,fn);return nextFrame;},cancelAnimationFrame:id=>frames.delete(id),setTimeout,clearTimeout,setInterval,clearInterval,AbortController,addEventListener(){},innerWidth:1440,innerHeight:900,fixtures:{day:data('default-chain.json'),seed:data('story-extra-activity.json'),cached:data('precomputed-chains.json'),process:data('allocation-process.json'),story:data('allocation-contribution-story.json')}};
let source=fs.readFileSync(path.join(root,'showcase.js'),'utf8');
source=source.replace('  init();',`  day=displayDay(fixtures.day);initialDay=JSON.parse(JSON.stringify(day));seed=fixtures.seed;cachedChains=fixtures.cached;allocationProcess=fixtures.process;contributionStory=fixtures.story;config={activityTypes:[]};ready=true;map={stop(){}};let demoResets=0;convenience={present(){},setVisible(){},prepareDemo(){demoResets++;}};
  const actualUpdateCaption=updateCaption;render=()=>{};renderPanel=()=>{};updateCaption=()=>{};updateMap=()=>{};frameScene=()=>{};resetAllocationMapAnimation=()=>{};syncAllocationMarker=()=>{};syncAllocationMap=()=>{};updateAllocationRibbon=()=>{};completeAllocationMap=()=>{};storyMode=()=>{};
  bindEvents();globalThis.test={state,play,pause,seek,tick,allocationMapClock,allocationHTML,renderResidentCard,updatePlayback,tween,tweenValue,setScene,getEnd:()=>playbackEnd,setEnd:v=>playbackEnd=v,getDemoResets:()=>demoResets,actualUpdateCaption};`);
vm.runInNewContext(source,context);const t=context.test;
const click=id=>element(id).listeners.click();
// The real input handler must capture the target before pause writes the old value back.
t.seek(3);element('timeline').value='15.25';element('timeline').listeners.input({target:element('timeline')});assert.equal(t.state.time,15.25);assert.equal(t.state.allocationStage,2);assert.equal(t.state.playing,false);
t.seek(10);t.play();t.tick(1000);t.tick(1100);assert.equal(t.state.time,10.1);t.pause();const paused=t.state.time;t.tick(5000);assert.equal(t.state.time,paused);assert.equal(t.allocationMapClock(),paused);
// Resume preserves the current day object and does not reset the story.
t.play();t.tick(6000);assert.equal(t.state.time,paused);t.tick(6100);assert.equal(t.state.time,paused+.1);t.pause();
const resetsBefore=t.getDemoResets();t.seek(0);assert.equal(t.getDemoResets(),resetsBefore+1);
for(const time of [0,9.5,10,12.4,12.5,17.9,18,22,27,30,33,38,44,50,5]){t.seek(time);assert.equal(t.state.time,time);assert.equal(t.state.scene,time<10?'convenience':time<18?'allocation':time<27?'chain':time<38?'recommendation':'records');}
for(let i=0;i<3;i++){click('replay');assert.equal(t.state.time,0);assert.equal(t.state.playing,true);click('play-toggle');assert.equal(t.state.playing,false);click('play-toggle');assert.equal(t.state.playing,true);t.pause();}
t.setEnd(18);t.seek(17.95);t.play();t.tick(1000);t.tick(1100);assert.equal(t.state.time,18);assert.equal(t.state.scene,'allocation');assert.equal(t.state.playing,false);
t.setEnd(50);t.seek(50);t.play();assert.equal(t.state.time,0);t.pause();
t.seek(10);const html=t.allocationHTML();assert.match(html,/<details/);assert.match(html,/居住网格/);assert.ok(!html.includes('data-allocation-stage'));
t.setEnd(10);t.seek(9.95);t.play();t.tick(1000);t.tick(1100);assert.equal(t.state.time,10);assert.equal(t.state.scene,'convenience');assert.equal(t.state.playing,false);t.setEnd(50);
console.log('PASS: timeline input, pause/resume, repeated replay, three-scene seeking, scoped replays and end restart.');

// A held mouse press spans many animation frames. The button subtree must never be replaced.
let buttonMarkupWrites=0;
Object.defineProperty(element('play-toggle'),'innerHTML',{configurable:true,get(){return 'persistent icons';},set(){buttonMarkupWrites++;}});
for(let i=0;i<100;i++){
  t.state.playing=true;
  for(let frame=0;frame<6;frame++)t.updatePlayback();
  click('play-toggle');assert.equal(t.state.playing,false);
  assert.equal(element('pause-glyph').hidden,true);assert.equal(element('play-glyph').hidden,false);
  click('play-toggle');assert.equal(t.state.playing,true);t.pause();
}
assert.equal(buttonMarkupWrites,0);
const markup=fs.readFileSync(path.join(root,'index.html'),'utf8');assert.match(markup,/id="play-glyph"/);assert.match(markup,/id="pause-glyph"/);
console.log('PASS: 100 pause/resume clicks across frame updates without replacing button nodes.');

assert.equal(t.getEnd(),50);assert.match(markup,/max="50"/);assert.match(markup,/id="history-frame"/);console.log("PASS: 50-second playback includes the records chapter.");

t.seek(40);assert.doesNotThrow(()=>t.actualUpdateCaption());assert.equal(context.document.body.dataset.scene,'records');console.log('PASS: direct seek to records keeps captions safe.');
