import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
num={'a':[(0,1.5),(50,3.0),(100,5.0),(150,11.0),(200,96.8),(240,97.8)],'b':[(0,1.5),(50,2.2),(100,2.2),(150,3.7),(200,1.7),(250,1.2),(300,1.3),(350,1.7)],'c':[(0,1.5),(50,2.5),(100,3.3),(150,3.2),(200,1.3),(250,1.7),(300,1.3),(350,1.8)],
'L':[(0,1.2),(50,5.8),(100,63.3),(110,63.3)],'Q':[(0,0.2),(50,1.0),(100,2.5),(150,5.7),(200,5.8),(250,2.3)]}
rea={'a':[(0,2),(50,4),(100,32),(150,88),(200,98),(240,98)],'b':[(0,2),(50,3),(100,2),(150,4),(200,4),(250,21),(300,14),(350,21)],'c':[(0,2),(50,2),(100,4),(150,12),(200,11),(250,8),(300,16),(350,4)],
'L':[(0,1.2),(50,7.2),(100,0),(110,0)],'Q':[(0,2.7),(50,3.7),(100,6.0),(150,65.8),(200,92.2),(250,80.3)]}
off={'a':[(0,0),(50,0),(100,0),(150,0),(200,26),(240,26)],'b':[(0,0),(50,0),(100,0),(150,0),(200,0),(250,0),(300,1),(350,0)],'c':[(s,0) for s in (0,50,100,150,200,250,300,350)],
'L':[(0,0),(50,6.2),(100,59.3),(110,59.3)],'Q':[(0,0),(50,0.3),(100,0),(150,0.2),(200,0.2),(250,0.3)]}
st={'a':('(a) click predictor, Qwen-3B, seed 0','#2a6fbb','o','-'),'b':('(b) simulated audience, Qwen-3B, seed 0','#c8553d','s','-'),'c':('(c) click predictor + faithfulness penalty, Qwen-3B, seed 0','#3a8f5c','^','-'),
'L':('(a) click predictor, Llama-8B','#7b4fa0','D','--'),'Q':('(a) click predictor, Qwen-14B','#1aa3b5','v','--')}

S=[0,50,100,150,200,250,300,350]
num1={'a':[1.5,2.5,4.5,13.2,43.0,100,100,100],'b':[1.5,1.2,2.5,2.3,2.3,1.8,2.5,2.2],'c':[1.5,1.3,2.0,2.8,3.0,2.5,4.5,6.2]}
rea1={'a':[2.5,3.8,26.0,89.7,97.7,98,98,98],'b':[2.5,4.7,2.7,7.3,5.0,7.5,15.7,32.3],'c':[2.5,1.7,10.8,9.5,4.5,6.0,7.7,21.3]}
off1={'a':[0,0,0.2,0,1.5,24,24,24],'b':[0,0,0,0.5,0.2,0.2,0.5,0.2],'c':[0,0,0.2,0,0,0,0,0]}
fig,ax=plt.subplots(1,3,figsize=(10.5,3.4),sharey=True)
for k,(d,t,d1) in enumerate([(rea,'Invented reactions',rea1),(num,'Invented numbers',num1),(off,'Off-article headlines',off1)]):
    for a in 'abc':
        ax[k].plot(S,d1[a],lw=0.9,ls='-',color=st[a][1],alpha=0.55)
    for a in 'abcLQ':
        x,y=zip(*d[a]); ax[k].plot(x,y,marker=st[a][2],ms=3.5,lw=1.6,ls=st[a][3],color=st[a][1],label=st[a][0])
    ax[k].set_title(t,fontsize=11); ax[k].set_xlabel('RL training step'); ax[k].set_xticks([0,100,200,300])
    ax[k].grid(alpha=.25); ax[k].spines[['top','right']].set_visible(False)
ax[0].set_ylabel('% of test headlines'); ax[0].set_ylim(-3,103)
h,l=ax[0].get_legend_handles_labels()
from matplotlib.lines import Line2D
h.append(Line2D([0],[0],color='grey',lw=0.9,alpha=0.6)); l.append('thin lines: seed 1')
fig.legend(h,l,loc='lower center',ncol=3,fontsize=8,frameon=False)
fig.tight_layout(rect=(0,0.13,1,1)); fig.savefig('fig_stages.pdf'); fig.savefig('fig_stages.png',dpi=150)
