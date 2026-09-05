import {STAGES,tensor} from './stages.js';
import {VOCAB,WEIGHTS} from './model.js';
export class Observatory {
  constructor(host,onStage,onFocus) {
    this.host=host;this.onStage=onStage;this.onFocus=onFocus;this.available=false;
    try {
      const T=window.THREE;if(!T) throw Error('Three.js missing');this.T=T;
      this.renderer=new T.WebGLRenderer({antialias:true,alpha:true});
      this.renderer.setPixelRatio(Math.min(window.devicePixelRatio,2));this.renderer.setClearColor(0x000000,0);host.prepend(this.renderer.domElement);
      this.scene=new T.Scene();this.camera=new T.PerspectiveCamera(38,1,.1,200);
      this.controls=new T.OrbitControls(this.camera,this.renderer.domElement);this.controls.enableDamping=true;this.controls.dampingFactor=.09;this.controls.minDistance=10;this.controls.maxDistance=65;this.controls.maxPolarAngle=Math.PI*.88;
      this.scene.add(new T.AmbientLight(0xffffff,.85));const sun=new T.DirectionalLight(0xe3fffa,.8);sun.position.set(5,18,12);this.scene.add(sun);
      this.geometry=new T.BoxGeometry(.30,1,.30);this.material=new T.MeshStandardMaterial({roughness:.55,metalness:.1});
      this.cells=new T.InstancedMesh(this.geometry,this.material,8*12*16);this.cells.instanceMatrix.setUsage(T.DynamicDrawUsage);this.scene.add(this.cells);
      this.decoration=new T.Group();this.scene.add(this.decoration);
      this.grid=new T.GridHelper(34,34,0x254351,0x172d39);this.grid.position.y=-.50;this.scene.add(this.grid);
      // Neutral arrows denote forward data dependencies, not simulated activations.
      for(let i=0;i<7;i++)this.scene.add(new T.ArrowHelper(new T.Vector3(1,0,0),new T.Vector3(this.x(i)+.95,-.20,0),1.13,0x466976,.20,.13));
      this.raycaster=new T.Raycaster();this.pointer=new T.Vector2();this.labels=[];this.rowLabels=[];
      const names=['01 · EMBED','02 · L1 / Q','03 · L1 / MIX','04 · L1 / OUT','05 · L2 / Q','06 · L2 / MIX','07 · L2 / OUT','08 · P(NEXT)'];
      this.baseStages=[0,2,3,7,9,10,14,17];
      names.forEach((name,i)=>{const el=document.createElement('button');el.className='scene-label';el.textContent=name;el.setAttribute('aria-label',`Inspect ${STAGES[this.baseStages[i]].name}${i>0&&i<7?` in layer ${i<4?1:2}`:''}`);el.addEventListener('click',()=>this.onStage(this.baseStages[i]));document.querySelector('#sceneLabels').append(el);this.labels.push(el);});
      this.renderer.domElement.addEventListener('pointermove',e=>this.hover(e));
      this.renderer.domElement.addEventListener('pointerleave',()=>document.querySelector('#hoverInfo').style.display='none');
      this.renderer.domElement.addEventListener('pointerdown',e=>{this.down=[e.clientX,e.clientY];});
      this.renderer.domElement.addEventListener('pointerup',e=>{if(this.down&&Math.hypot(e.clientX-this.down[0],e.clientY-this.down[1])<5){const hit=this.pick(e);if(hit)this.onFocus(hit.p);}});
      this.resizeObserver=new ResizeObserver(()=>this.resize());this.resizeObserver.observe(host);this.available=true;this.reset();this.resize();this.animate();
    } catch(e) {console.warn('3D unavailable:',e);document.querySelector('#sceneFallback').hidden=false;}
  }
  x(i){return (i-3.5)*3.05;}
  reset(){if(!this.controls)return;this.camera.position.set(8,9,22);this.controls.target.set(0,1.7,0);this.controls.update();}
  resize(){if(!this.renderer)return;const w=this.host.clientWidth,h=this.host.clientHeight;this.renderer.setSize(w,h);this.camera.aspect=w/h;this.camera.fov=w<550?64:w<750?43:32;this.camera.updateProjectionMatrix();}
  clearDecoration(){while(this.decoration.children.length){const o=this.decoration.children[0];this.decoration.remove(o);o.geometry?.dispose();if(Array.isArray(o.material))o.material.forEach(m=>m.dispose());else o.material?.dispose();}}
  update(run,index,focus,head=0,attentionLayer=null){
    this.run=run;this.focus=focus;this.index=index;
    if(!this.available)return;
    const T=this.T,stage=STAGES[index],obj=new T.Object3D(),color=new T.Color(),zero=new T.Color('#263f4b'),pos=new T.Color('#8df0cf'),neg=new T.Color('#edab70');
    this.data=[];this.clearDecoration();let n=0;
    this.rowLabels.forEach(el=>el.remove());this.rowLabels=run.ids.map((id,p)=>{const el=document.createElement('button');el.className='scene-row-label';el.textContent=`p${p}`;el.title=`Inspect position ${p}: ${VOCAB[id]}`;el.setAttribute('aria-label',el.title);if(p===focus)el.classList.add('selected');el.addEventListener('click',()=>this.onFocus(p));document.querySelector('#sceneLabels').append(el);return el;});
    for(let i=0;i<8;i++) {
      const s=i===stage.station?stage:STAGES[this.baseStages[i]],values=tensor(run,s,s.key,WEIGHTS),prob=s.key==='probs';
      this.labels[i].classList.toggle('selected',i===stage.station);
      this.labels[i].textContent=`${s.layer!==undefined?'L'+(s.layer+1)+' · ':''}${({Attention:'ATTN','Q · K · V':'Q','Skip 2':'OUT','Skip 1':'SKIP','Final norm':'NORM'})[s.short]??s.short.toUpperCase()}`;
      this.labels[i].title=`${s.name} · ${values.length} × ${values[0].length}`;
      values.forEach((row,p)=>row.slice(0,16).forEach((value,d)=>{
        const t=prob?value:Math.tanh(value/2),height=.07+Math.abs(t)*(prob?1.8:.40);
        obj.position.set(this.x(i)+(d%4-1.5)*.36,p*.68+height/2,(Math.floor(d/4)-1.5)*.36);obj.scale.set(1,height,1);obj.updateMatrix();this.cells.setMatrixAt(n,obj.matrix);
        color.copy(zero).lerp(t<0?neg:pos,Math.min(1,Math.abs(t)));this.cells.setColorAt(n,color);
        this.data.push({p,d,value,name:s.name,layer:s.layer,prob});n++;
      }));
      // Outlines locate the inspected token; they don't encode additional values.
      const outline=new T.LineSegments(new T.EdgesGeometry(new T.BoxGeometry(1.58,.52,1.58)),new T.LineBasicMaterial({color:i===stage.station?0x8df0cf:0x38505f,transparent:true,opacity:i===stage.station?.8:.32}));outline.position.set(this.x(i),focus*.68+.21,0);this.decoration.add(outline);
      if(i===stage.station){const ring=new T.Mesh(new T.PlaneGeometry(2.1,2.1),new T.MeshBasicMaterial({color:0x8df0cf,transparent:true,opacity:.04,side:T.DoubleSide}));ring.rotation.x=-Math.PI/2;ring.position.set(this.x(i),-.46,0);this.decoration.add(ring);}
    }
    this.cells.count=n;this.cells.instanceMatrix.needsUpdate=true;this.cells.instanceColor.needsUpdate=true;
    if(!this.colorInitialized){this.material.needsUpdate=true;this.colorInitialized=true;}
    // Recompute bounds where supported, otherwise disable frustum culling for dynamic instances.
    this.cells.frustumCulled=false;
    const l=attentionLayer??(stage.short==='Attention'?stage.layer:null);
    if(l!==null){const station=l===0?2:5,a=run.rows[focus].blocks[l].heads[head];for(let j=0;j<=focus;j++){
      const start=new T.Vector3(this.x(station)+.8,j*.68+.18,0),end=new T.Vector3(this.x(station)+.8,focus*.68+.18,.12);
      const mid=start.clone().lerp(end,.5);mid.x+=.5+(focus-j)*.12;mid.z+=.7+(focus-j)*.15;
      const curve=new T.QuadraticBezierCurve3(start,mid,end),geometry=new T.BufferGeometry().setFromPoints(curve.getPoints(24));
      const line=new T.Line(geometry,new T.LineBasicMaterial({color:head===0?0x8df0cf:0x8bafff,transparent:true,opacity:a.probs[j]}));this.decoration.add(line);
    }}
    document.querySelector('#sceneEncoding').textContent=l!==null?`Arcs: layer ${l+1}, head ${head}, key → query ${focus}; opacity = attention weight.`:'Click a tile to select its token. The complete forward trace is shown; stepping highlights one operation.';
  }
  pick(e){if(!this.available)return null;const r=this.renderer.domElement.getBoundingClientRect();this.pointer.set((e.clientX-r.left)/r.width*2-1,-(e.clientY-r.top)/r.height*2+1);this.raycaster.setFromCamera(this.pointer,this.camera);const hit=this.raycaster.intersectObject(this.cells)[0];return hit?this.data[hit.instanceId]:null;}
  hover(e){const item=this.pick(e),el=document.querySelector('#hoverInfo');el.style.display=item?'block':'none';if(item)el.textContent=`${item.layer!==undefined?'L'+(item.layer+1)+' · ':''}${item.name} | p${item.p} “${VOCAB[this.run.ids[item.p]]}” | ${item.prob?'vocab':'feature'} ${item.d} = ${item.value.toPrecision(10)}`;}
  animate(){requestAnimationFrame(()=>this.animate());if(document.hidden)return;this.controls.update();this.renderer.render(this.scene,this.camera);this.rowLabels.forEach((el,p)=>{const v=new this.T.Vector3(this.x(0)-1.2,p*.68+.15,0).project(this.camera);el.style.left=`${(v.x*.5+.5)*this.host.clientWidth}px`;el.style.top=`${(-v.y*.5+.5)*this.host.clientHeight}px`;el.style.visibility=v.z<1&&v.z>-1?'visible':'hidden';});this.labels.forEach((el,i)=>{const v=new this.T.Vector3(this.x(i),-.83,.1).project(this.camera);el.style.left=`${(v.x*.5+.5)*this.host.clientWidth}px`;el.style.top=`${(-v.y*.5+.5)*this.host.clientHeight}px`;el.style.visibility=v.z<1&&v.z>-1?'visible':'hidden';});}
}
