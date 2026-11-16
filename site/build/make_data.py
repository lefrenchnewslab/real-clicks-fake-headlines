import json, csv, pandas as pd, collections, re
P='../'  # run from project2/site: python3 build/make_data.py
D={}
# training rollouts
runs={}
for r in ['main_a_s0','main_a_s1','main_b_s0','main_b_s1','main_c_s0','main_c_s1']:
    L=[json.loads(l) for l in open(P+f'runs_main/{r}/reward_log.jsonl')]
    runs[r]=[[x['step'],round(x['reward_mean'],3),round(x['distinct_per_group'],3),round(x['valid_rate'],3),x['examples'][:3]] for x in L]
D['train']=runs
# trajectories (eval checkpoints)
tr=[]
for row in csv.DictReader(open(P+'trajectories.csv')):
    tr.append({k:(row[k] if k in('run','arm') else (float(row[k]) if row[k]!='' else None)) for k in row})
D['traj']=tr
# intermediate labelled samples
im=[]
for row in csv.DictReader(open(P+'intermediate_label_table_joined.csv')):
    im.append({'run':row['source'],'step':int(row['step']),'h':row['headline'],'react':int(row['invented_reaction']),'other':int(row['other_invented_fact']),'viral':int(row['viral_template_style']),'why':row['reason']})
D['ckpt_labels']=im
# final outputs: same article across sources
S=P+'runs_stage3/'
srcs=['base','main_a_s0','main_a_s1','main_b_s0','main_b_s1','main_c_s0','main_c_s1']
F={}
for s in srcs:
    g=pd.read_parquet(S+f'final_{s}.parquet'); m=pd.read_parquet(S+f'final_{s}.metrics.parquet')
    g=pd.concat([g.reset_index(drop=True),m[['fabricated','numbers_only','unsup_ent','unsup_num']].reset_index(drop=True)],axis=1)
    F[s]=g
ids=F['base'].id.unique().tolist()
# choose articles: where b and c differ informative; pick 12 ids deterministic with b flagged
import random
rnd=random.Random(3)
cand=[i for i in ids if F['main_b_s0'][(F['main_b_s0'].id==i)&(F['main_b_s0']['sample']==0)].fabricated.iloc[0]==1]
pick=rnd.sample(cand,6)+rnd.sample([i for i in ids if i not in cand],4)
same=[]
for i in pick:
    row={'id':i}
    for s in srcs:
        x=F[s][(F[s].id==i)&(F[s]['sample']==0)].iloc[0]
        row[s]={'h':x.headline,'ent':json.loads(x.unsup_ent),'num':json.loads(x.unsup_num)}
    same.append(row)
D['same']=same
# collapse: distinct headline counts and top headlines
col={}
for s in srcs:
    vc=F[s].headline.value_counts()
    col[s]={'n':int(len(F[s])),'distinct':int(vc.size),'top':[[h,int(c)] for h,c in vc.head(6).items()]}
D['collapse']=col
# reference labels (final ckpt) with notes, no article text
key={r['item']:r for r in csv.DictReader(open(P+'handlabel/key.csv'))}
labs=[]
for r in csv.DictReader(open(P+'handlabel/labels.csv')):
    k=key[r['item']]
    lab=[c for c in ['faithful','fabricated_fact','misleading_exaggeration','teaser_no_claim','other'] if r[c].strip() in('1','x','X','1.0')]
    labs.append({'src':k['source'],'flag':int(k['flagged']),'h':r['headline'],'label':lab[0] if lab else r.get('claude_label',''),'note':r['note']})
D['labels']=labs
json.dump(D,open('src/data.json','w'),ensure_ascii=False,separators=(',',':'))
print({k:len(v) for k,v in D.items()}); print(collections.Counter(l['label'] for l in labs)); print(col['main_a_s0']['distinct'],col['base']['distinct'],col['main_a_s1']['distinct'],col['main_b_s0']['distinct'])
import os; print(os.path.getsize('src/data.json'))
print(json.dumps(same[0],ensure_ascii=False)[:800])
# scale runs with example headlines per checkpoint
s=json.load(open(P+'scale_E.json'))
sc={}
for k,v in s.items():
    sc[k]=[{'step':int(st),'num':c['numbers_only'],'react':c['react'],'off':c['offarticle'],'distinct':c['distinct_per_600'],'kl':c['kl'],'ex':c.get('examples',[])[:3]} for st,c in sorted(v['curve'].items(),key=lambda t:int(t[0]))]
D['scale']=sc
json.dump(D,open('src/data.json','w'),ensure_ascii=False,separators=(',',':'))
print({k:[(r['step'],round(r['num'],2)) for r in v] for k,v in sc.items()})
