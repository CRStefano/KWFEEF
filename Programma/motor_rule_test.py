"""Test: la non-monotonicita' della frontiera di Pareto e' un risultato robusto
o un artefatto della regola motoria 'centroide'? Confrontiamo 3 regole motorie
a parita' di tutto il resto (stesso spazio stati, stessa viability, stessa MI).

Regole motorie (azione data la CLASSE percepita del cibo):
  centroid     : muovi verso la cella di griglia piu' vicina al centroide della classe (ATTUALE)
  random       : muovi verso un membro CASUALE della classe (mistura uniforme)
  bayes        : azione che massimizza la prob. attesa di mangiare dato il posterior nella classe
"""
import numpy as np
from core.model import SemanticForagingModel, set_partitions, CELLS, GRID, NO_FOOD, N_FOOD, N_Y, _manh

PARTS = list(set_partitions(N_Y))

def class_of(partition):
    """mappa f -> lista membri-cibo della sua classe"""
    cm = {}
    for cls in partition:
        food = [c for c in cls if c != NO_FOOD]
        for y in cls:
            cm[y] = food
    return cm

def grid_neighbours(pos):
    x,y=pos; nb=[pos]
    for dx,dy in ((1,0),(-1,0),(0,1),(0,-1)):
        nx,ny=x+dx,y+dy
        if 0<=nx<3 and 0<=ny<3: nb.append((nx,ny))
    return nb

def step_toward(pos,tgt):
    if tgt is None or tgt==pos: return pos
    x,y=pos
    if abs(tgt[0]-x)>=abs(tgt[1]-y) and tgt[0]!=x: return (x+(1 if tgt[0]>x else -1),y)
    if tgt[1]!=y: return (x,y+(1 if tgt[1]>y else -1))
    return (x+(1 if tgt[0]>x else -1),y)

def action_probs(m, pos, members, rule):
    """restituisce dict {cella_destinazione: prob} per la regola data."""
    if not members:
        nb=grid_neighbours(pos); return {a:1.0/len(nb) for a in nb}
    if rule=='centroid':
        cx=np.mean([CELLS[c][0] for c in members]); cy=np.mean([CELLS[c][1] for c in members])
        tgt=min(GRID,key=lambda g:(g[0]-cx)**2+(g[1]-cy)**2)
        return {step_toward(pos,tgt):1.0}
    if rule=='random':
        d={}
        for c in members:
            a=step_toward(pos,CELLS[c]); d[a]=d.get(a,0.0)+1.0/len(members)
        return d
    if rule=='bayes':
        r=m.eat_radius; best=None; bestscore=-1; besttie=1e9
        for a2 in grid_neighbours(pos):
            score=np.mean([1.0 if _manh(a2,CELLS[c])<=r else 0.0 for c in members])
            tie=np.mean([_manh(a2,CELLS[c]) for c in members])  # spareggio: vicinanza media
            if score>bestscore+1e-12 or (abs(score-bestscore)<1e-12 and tie<besttie):
                best=a2; bestscore=score; besttie=tie
        return {best:1.0}

def build_T(m, partition, rule):
    cm=class_of(partition); T=np.zeros((m.num_states,m.num_states))
    for si,(pos,f,v) in enumerate(m.states):
        if v==0: T[si,si]=1.0; continue
        members = cm[f] if f!=NO_FOOD else []
        ap = action_probs(m,pos,members,rule) if f!=NO_FOOD else {a:1.0/len(grid_neighbours(pos)) for a in grid_neighbours(pos)}
        for a2,pmove in ap.items():
            ate=(f!=NO_FOOD) and (_manh(a2,CELLS[f])<=m.eat_radius)
            if ate:
                for fn in range(N_FOOD): T[si,m.state2id[(a2,fn,1)]]+=pmove/N_FOOD
            else:
                hz=m.base_hazard
                if f==NO_FOOD:
                    T[si,m.state2id[(a2,NO_FOOD,1)]]+=pmove*(1-hz); T[si,m.state2id[(a2,NO_FOOD,0)]]+=pmove*hz
                else:
                    pd=m.food_disappear_rate
                    T[si,m.state2id[(a2,f,1)]]+=pmove*(1-pd)*(1-hz); T[si,m.state2id[(a2,f,0)]]+=pmove*(1-pd)*hz
                    T[si,m.state2id[(a2,NO_FOOD,1)]]+=pmove*pd*(1-hz); T[si,m.state2id[(a2,NO_FOOD,0)]]+=pmove*pd*hz
    rs=T.sum(1,keepdims=True); rs=np.where(rs<1e-12,1,rs); return T/rs

def envelope(m, rule, alpha=9.5):
    ip=m.get_initial_distribution(2,'uniform','food')
    ks,vs=[],[]
    for part in PARTS:
        ptau=ip@np.linalg.matrix_power(build_T(m,part,rule),m.timescale)
        ks.append(m._mi_partition(part)); vs.append(m._viability(ptau))
    ks,vs=np.array(ks),np.array(vs)
    order=np.argsort(ks); ks,vs=ks[order],vs[order]
    ku,vu=[],[]; i=0
    while i<len(ks):
        k0=ks[i]; j=i; vm=vs[i]
        while j<len(ks) and ks[j]-k0<=1e-9: vm=max(vm,vs[j]); j+=1
        ku.append(k0); vu.append(vm); i=j
    ku,vu=np.array(ku),np.array(vu)
    dips=int(np.sum(np.diff(vu)<-1e-6))
    over=float(vu.max()-vu[-1])
    fs=alpha*ku-vu; istar=float(ku[int(np.argmin(fs))])
    return ku,vu,dips,over,istar

m=SemanticForagingModel(eat_radius=1, timescale=5)
print('regola    | envelope V per livello di I (0,0.72,0.97,1.37,1.52,1.92,2.32) | dips over_pen I*')
for rule in ['centroid','random','bayes']:
    ku,vu,dips,over,istar=envelope(m,rule)
    vs_str=' '.join('%6.2f'%v for v in vu)
    print('%-9s | %s | %d   %5.2f   %.3f'%(rule,vs_str,dips,over,istar))
