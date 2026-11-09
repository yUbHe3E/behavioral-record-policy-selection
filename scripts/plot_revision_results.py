"""Publication figures from completed study summaries; no new analyses or calls."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
DEST=ROOT/'generated_figures'
DEST.mkdir(exist_ok=True)
CODE=ROOT
S=json.loads((CODE/'outputs/kbs_revision_r1_20261104/claim_evidence_synthesis_20261105.json').read_text())
U=json.loads((ROOT/'figure_data/usage_figure_data.json').read_text())
OLD=json.loads((ROOT/'figure_data/figure_data_v2.json').read_text())['controlled_outer']
BLUE='#23689B'; ORANGE='#C56630'; INK='#243343'; GREY='#7C8995'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.labelcolor':INK,'text.color':INK,'xtick.color':INK,'ytick.color':INK,'axes.edgecolor':'#B7C0C8','pdf.fonttype':42,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False,'axes.titleweight':'bold','axes.titlesize':11,'savefig.facecolor':'white'})
def save(f,n):
    for ext in ['pdf','png','svg']:f.savefig(DEST/(n+'.'+ext),bbox_inches='tight',dpi=220)
    plt.close(f)

f,axes=plt.subplots(1,2,figsize=(10,4.1),sharey=True,layout='constrained')
for ax,new,title in zip(axes,[False,True],['A  Original evaluations','B  Additional evaluations']):
    for j,(model,color) in enumerate([('MiniMax-M3',BLUE),('deepseek-flash' if new else 'deepseek-v4-flash',ORANGE)]):
        if new:
            d=S['new_matched_contrasts'][model]; vals=[d['by_requirement']['C1'],d['by_requirement']['C2'],d]
        else:
            d=S['original_requirement_effects'][model]['by_requirement']; p=S['original_matched_contrasts_from_submitted_abstract'][model]
            vals=[d['C1'],d['C2'],{'mean':p['effect'],'ci95':p['ci95']}]
        for i,v in enumerate(vals):
            x=100*v['mean']; low,hi=np.array(v['ci95'])*100; y=2-i+(-.13 if j else .13)
            ax.errorbar(x,y,xerr=[[x-low],[hi-x]],fmt='s' if i==2 else 'o',color=color,capsize=3,ms=6,label=('MiniMax' if j==0 else 'DeepSeek') if i==0 else None)
    ax.axvline(0,color=GREY,lw=1,ls='--');ax.set_xlim(-7,52);ax.set_ylim(-.55,2.6)
    ax.set_yticks([2,1,0],['Recovery','Necessity','Pooled']);ax.set_xlabel('Preserved − disrupted (percentage points)');ax.set_title(title,loc='left',pad=14)
    ax.grid(axis='x',alpha=.15);ax.legend(loc='upper right',frameon=False,fontsize=9)
save(f,'figure_03_controlled_static')

arms=['Event-Caption','FullTrace-Control','LLM-Summary','Relational-Linked','Relational-LinkBroken']
labels=['Event descriptions','Complete trajectories','Model summaries','Preserved relations','Disrupted relations']
byarm={r['arm']:r for r in S['new_minimax_baselines']}
f,axes=plt.subplots(1,2,figsize=(10,4.2),gridspec_kw={'width_ratios':[1.25,1]},layout='constrained')
y=np.arange(5)[::-1]
for key,color,mark,label in [('recovery_selection_rate',BLUE,'o','Recovery'),('necessity_selection_rate',ORANGE,'s','Necessity')]:
    axes[0].scatter([byarm[a][key]*100 for a in arms],y+(.10 if mark=='o' else -.10),color=color,marker=mark,s=35,label=label,zorder=3)
axes[0].set_yticks(y,labels);axes[0].set_xlim(65,100);axes[0].set_xlabel('Reference-label selection (%)');axes[0].set_title('A  Selection accuracy',loc='left',pad=15);axes[0].legend(frameon=False,loc='lower left',fontsize=9);axes[0].grid(axis='x',alpha=.16)
vals=[U[a]['input_tokens_mean']/1000 for a in arms]
axes[1].barh(y,vals,color=[BLUE]*3+[ORANGE]*2,height=.52,alpha=.85)
for yy,v in zip(y,vals):axes[1].text(v+.15,yy,f'{v:.2f}',va='center',fontsize=9)
axes[1].set_xlim(0,max(vals)*1.18);axes[1].set_yticks(y,[]);axes[1].set_xlabel('Recorded prompt tokens (thousands)');axes[1].set_title('B  Input per request with usage',loc='left',pad=15);axes[1].grid(axis='x',alpha=.13);axes[1].set_axisbelow(True)
save(f,'figure_04_common_formats')

f,axes=plt.subplots(1,2,figsize=(10,4.4),sharey=True,layout='constrained')
for key,label,color,ls in [('B2-FullTrace-LLM','Complete episodes',BLUE,'-'),('B1-LLM','Aggregates',ORANGE,'-'),('Metric selector','Metric selection','#81939F','--'),('Random selector','Random selection','#B0B8BE',':')]:
    axes[0].plot([0,1,2],np.array(OLD['pass_rates'][key])*100,marker='o',color=color,ls=ls,label=label)
for key,label,color in [('linked_test_pass','Preserved relations',BLUE),('disrupted_test_pass','Disrupted relations',ORANGE)]:
    axes[1].plot([0,1,2],[r[key]*100 for r in S['retention']],marker='o',color=color,label=label)
for ax,title in zip(axes,['A  Complete episodes versus aggregates','B  Preserved versus disrupted relations']):
    ax.set_xticks([0,1,2]);ax.set_xlabel('Generation');ax.set_title(title,loc='left',pad=17,fontsize=10.5);ax.set_ylim(38,102);ax.grid(alpha=.16);ax.legend(frameon=False,fontsize=8.5,loc='center right',bbox_to_anchor=(1,0.43))
axes[0].set_ylabel('Selected-parent requirement-test pass (%)')
axes[0].text(.03,.97,'Final difference: +41.5 pp\n95% interval [37.7, 45.4]',transform=axes[0].transAxes,va='top',fontsize=9)
axes[1].text(.03,.97,'Final difference: +3.79 pp\n95% interval [−1.00, 8.26]',transform=axes[1].transAxes,va='top',fontsize=9)
save(f,'figure_05_repeated_retention')

f,ax=plt.subplots(figsize=(9,4.8),layout='constrained')
series=[('LunarLander: recovery',0,[-17.31,17.39],.71,[-6.67,8.87]),('LunarLander: robust landing',5.26,[-9.48,19.12],2.63,[-10.19,18.42]),('LunarLander: equal-weight mean',2.63,[-10.11,13.78],1.67,[-6.87,11.29]),('Highway: all ten pairs',-5,[-31.25,25],2.5,[0,12.5])]
for i,(label,mm,mmci,ds,dsci) in enumerate(series):
    for j,(val,ci,color,model) in enumerate([(mm,mmci,BLUE,'MiniMax'),(ds,dsci,ORANGE,'DeepSeek')]):
        ax.errorbar(val,3-i+(.13 if j==0 else -.13),xerr=[[val-ci[0]],[ci[1]-val]],fmt='o' if j==0 else 's',color=color,capsize=3,label=model if i==0 else None)
ax.axvline(0,color=GREY,ls='--',lw=1);ax.set_yticks([3,2,1,0],[x[0] for x in series]);ax.set_ylim(-.6,3.6);ax.set_xlim(-37,32);ax.set_xlabel('Preserved − disrupted selection (percentage points)');ax.grid(axis='x',alpha=.15);ax.legend(frameon=False,loc='lower left',ncol=2)
save(f,'figure_07_external_validation')
print('Updated four result figures from archived summaries.')
