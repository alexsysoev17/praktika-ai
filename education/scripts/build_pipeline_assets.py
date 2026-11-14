"""Authored procedural GLB teaching assets. No image model or AI-generation claims."""
from pathlib import Path
import json,struct,math,hashlib
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
DEST=ROOT/'assets/pipeline';DEST.mkdir(parents=True,exist_ok=True)
PALETTE=['#C4B5FD','#A3E635','#F1EDFF']
def linear(h):
    a=[int(h[i:i+2],16)/255 for i in (1,3,5)]
    return [x/12.92 if x<=.04045 else ((x+.055)/1.055)**2.4 for x in a]+[1]
def sphere(center,scale,material=0,nu=28,nv=16):
    p=[];n=[];idx=[]
    for j in range(nv+1):
        v=math.pi*j/nv
        for i in range(nu+1):
            u=2*math.pi*i/nu;d=np.array([math.sin(v)*math.cos(u),math.cos(v),math.sin(v)*math.sin(u)])
            p.append(np.array(center)+d*np.array(scale));normal=d/np.array(scale);n.append(normal/np.linalg.norm(normal))
    for j in range(nv):
        for i in range(nu):
            a=j*(nu+1)+i;b=a+nu+1
            idx += [a,a+1,b,a+1,b+1,b]
    return {'positions':p,'normals':n,'indices':idx,'material':material}
def box(center,size,r=.12,material=0,segments=8):
    p=[];n=[];idx=[];half=np.array(size)/2;inside=half-r
    faces=[([1,0,0],[0,0,-1],[0,1,0]),([-1,0,0],[0,0,1],[0,1,0]),([0,1,0],[1,0,0],[0,0,-1]),([0,-1,0],[1,0,0],[0,0,1]),([0,0,1],[1,0,0],[0,1,0]),([0,0,-1],[-1,0,0],[0,1,0])]
    for normal,uu,vv in faces:
        normal,uu,vv=map(np.array,(normal,uu,vv));start=len(p)
        for j in range(segments+1):
            for i in range(segments+1):
                q=(normal+uu*(2*i/segments-1)+vv*(2*j/segments-1))*half;c=np.clip(q,-inside,inside);d=q-c;d=d/np.linalg.norm(d);p.append(np.array(center)+c+r*d);n.append(d)
        for j in range(segments):
            for i in range(segments):
                a=start+j*(segments+1)+i;b=a+segments+1;idx += [a,a+1,b,a+1,b+1,b]
    return {'positions':p,'normals':n,'indices':idx,'material':material}
def torus(center,r,t,material=0,start=0,end=2*math.pi):
    p=[];n=[];idx=[];nu=48;nv=12
    for i in range(nu+1):
        u=start+(end-start)*i/nu
        for j in range(nv+1):
            v=2*math.pi*j/nv;d=np.array([math.cos(u)*math.cos(v),math.sin(u)*math.cos(v),math.sin(v)])
            p.append(np.array(center)+np.array([r*math.cos(u),r*math.sin(u),0])+t*d);n.append(d)
    for i in range(nu):
        for j in range(nv):
            a=i*(nv+1)+j;b=a+nv+1;idx += [a,b,a+1,a+1,b,b+1]
    return {'positions':p,'normals':n,'indices':idx,'material':material}
def components(name):
    if name=='cloud':return [box([0,-.32,0],[1.55,.48,.65],.2),sphere([-.43,-.03,0],[.45,.44,.40]),sphere([.05,.22,-.02],[.55,.57,.48]),sphere([.55,-.03,.02],[.4,.4,.36]),sphere([.5,-.55,.44],[.16,.16,.16],1)]
    if name=='folder':return [box([0,-.05,0],[1.65,1.13,.36],.13),box([-.48,.55,0],[.65,.35,.32],.1),box([0,-.13,.25],[1.45,.86,.16],.07,1),box([.35,-.02,.36],[.5,.13,.05],.02,2)]
    if name=='lock':return [torus([0,.28,0],.43,.14,2,0,math.pi),box([0,-.18,0],[1.25,.93,.55],.15),sphere([0,-.12,.3],[.13,.13,.08],1),box([0,-.3,.32],[.11,.25,.09],.04,1)]
    if name=='chart':return [box([0,-.66,0],[1.9,.22,.9],.1,2),box([-.58,-.21,0],[.38,.7,.5],.14),box([0,.02,0],[.38,1.16,.5],.14,1),box([.58,.3,0],[.38,1.72,.5],.14)]
    if name=='stack':return [box([0,-.43,0],[1.55,.32,1.15],.13),box([0,0,0],[1.55,.32,1.15],.13,2),box([0,.43,0],[1.55,.32,1.15],.13),sphere([.55,.61,.35],[.18,.18,.18],1)]
    return [torus([0,0,0],.78,.14),torus([0,0,.04],.44,.12,2),sphere([0,0,.1],[.22,.22,.17],1)]
def glb(name,parts,palette=PALETTE,roughness=.78):
    binary=bytearray();views=[];accessors=[];primitives=[]
    def buffer(values,ctype,gtype,target):
        arr=np.array(values,dtype='<f4' if ctype==5126 else '<u2');raw=arr.tobytes();offset=len(binary);binary.extend(raw);binary.extend(b'\0'*((-len(binary))%4));views.append({'buffer':0,'byteOffset':offset,'byteLength':len(raw),'target':target})
        a={'bufferView':len(views)-1,'componentType':ctype,'count':len(arr),'type':gtype}
        if ctype==5126:a.update(min=arr.min(axis=0).tolist(),max=arr.max(axis=0).tolist())
        accessors.append(a);return len(accessors)-1
    for part in parts:
        pos=buffer(part['positions'],5126,'VEC3',34962);norm=buffer(part['normals'],5126,'VEC3',34962);index=buffer(part['indices'],5123,'SCALAR',34963)
        primitives.append({'attributes':{'POSITION':pos,'NORMAL':norm},'indices':index,'material':part['material'],'mode':4})
    doc={'asset':{'version':'2.0','generator':'Praktika procedural learning fixtures v1'},'scene':0,'scenes':[{'nodes':[0]}],'nodes':[{'mesh':0,'name':name}],'meshes':[{'name':name,'primitives':primitives}],'materials':[{'name':h,'pbrMetallicRoughness':{'baseColorFactor':linear(h),'metallicFactor':0,'roughnessFactor':roughness}} for h in palette],'buffers':[{'byteLength':len(binary)}],'bufferViews':views,'accessors':accessors,'extras':{'provenance':'procedural_authored_not_ai','units':'metres','styleVersion':'clay-orbit-v1'}}
    raw=json.dumps(doc,separators=(',',':')).encode();raw+=b' '*((-len(raw))%4);length=12+8+len(raw)+8+len(binary)
    return struct.pack('<III',0x46546C67,2,length)+struct.pack('<II',len(raw),0x4E4F534A)+raw+struct.pack('<II',len(binary),0x004E4942)+binary
names=[('cloud','Облако'),('folder','Папка'),('lock','Замок'),('chart','Рост'),('stack','Слои'),('target','Цель')];artifacts=[]
for i,(name,title) in enumerate(names):
    raw=glb(name,components(name));p=DEST/(name+'.glb');p.write_bytes(raw)
    artifacts.append({'id':'mesh-'+name,'type':'mesh_3d','path':'assets/pipeline/'+p.name,'role':'pilot' if i<3 else 'batch','seriesId':'clay-orbit-v1','metadata':{'title':title,'provenance':'procedural_authored_not_ai','sha256':hashlib.sha256(raw).hexdigest()}})
for name,palette,roughness in [('bad-palette',['#EF4444',*PALETTE[1:]],.78),('bad-roughness',PALETTE,.1)]:
    raw=glb(name,components('folder'),palette,roughness);p=DEST/(name+'.glb');p.write_bytes(raw);artifacts.append({'id':'mesh-'+name,'type':'mesh_3d','path':'assets/pipeline/'+p.name,'role':'negative','seriesId':'clay-orbit-v1','metadata':{'title':'Ошибка: '+('чужая палитра' if 'palette' in name else 'глянцевый материал'),'provenance':'procedural_authored_not_ai','sha256':hashlib.sha256(raw).hexdigest()}})
# PNG previews are exported through the in-app WebGL viewer at the common camera.
# Preserve their provenance and register only already-exported files; do not fabricate model output.
for i,(name,title) in enumerate(names):
    p=DEST/(name+'.png')
    if p.exists():
        from PIL import Image
        with Image.open(p) as im:assert im.size==(512,512) and im.mode=='RGBA'
        artifacts.append({'id':'png-'+name,'type':'raster_3d','path':'assets/pipeline/'+p.name,'role':'pilot' if i<3 else 'batch','seriesId':'clay-orbit-v1','metadata':{'title':title+' · PNG','provenance':'webgl_render_of_procedural_authored_mesh_not_ai','sourceMesh':'mesh-'+name,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}})
registry={'version':'1','title':'Procedural geometry fixtures','note':'Synthetic shapes for file-validation tests. No model inference was used.','contracts':{'mesh_3d':{'maxTriangles':25000,'allowedBaseColors':PALETTE,'roughnessMin':.6,'metallicMax':.1,'maxExtent':3},'raster_3d':{'width':512,'height':512,'maxBytes':1000000,'requireAlpha':True}},'artifacts':artifacts}
(ROOT/'content/pipeline-artifacts.json').write_text(json.dumps(registry,ensure_ascii=False,indent=2));print({'assets':len(artifacts),'bytes':sum((ROOT/a['path']).stat().st_size for a in artifacts)})
