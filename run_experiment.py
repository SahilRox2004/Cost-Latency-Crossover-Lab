import argparse,os,json
from dataclasses import asdict
import matplotlib.pyplot as plt
from src.experiment import Config,run,summarize,crossover,DEFAULT_RATES,MODES
p=argparse.ArgumentParser(); p.add_argument("--repeats",type=int,default=5); p.add_argument("--requests",type=int,default=250)
p.add_argument("--workload",default="Small tabular classification"); a=p.parse_args()
cfg=Config(workload=a.workload,repeats=max(3,a.repeats),requests_per_trial=max(50,a.requests))
raw=run(cfg); s=summarize(raw); os.makedirs("results",exist_ok=True); os.makedirs("figures",exist_ok=True)
raw.to_csv("results/raw_runs.csv",index=False); s.to_csv("results/summary_statistics.csv",index=False); crossover(s).to_csv("results/crossover_summary.csv",index=False)
json.dump(asdict(cfg),open("results/experiment_config.json","w"),indent=2)
for metric,title,ylabel,filename in [("cost_per_1000_usd","Cost vs request rate","Cost / 1,000 inferences (illustrative $)","cost_vs_rate_ci.png"),("p99_latency_ms","p99 latency vs request rate","p99 latency (ms)","p99_latency_ci.png")]:
    fig,ax=plt.subplots(figsize=(8.5,5))
    for mode in MODES:
        g=s[s.deployment_mode==mode].sort_values("request_rate_rps")
        ax.errorbar(g.request_rate_rps,g[metric+"_mean"],yerr=g[metric+"_ci95_halfwidth"],marker="o",capsize=3,label=mode)
    ax.set_xscale("log"); ax.set_xlabel("Request rate (rps, log scale)"); ax.set_ylabel(ylabel); ax.set_title(title)
    ax.grid(True,alpha=.3); ax.legend(); fig.tight_layout(); fig.savefig("figures/"+filename,dpi=180); plt.close(fig)
# Cold-start ablation, holding the other parameters fixed.
parts=[]
for penalty in [0,100,250,450,700]:
    cfg2=Config(**{**asdict(cfg),"cold_start_ms":float(penalty)})
    x=run(cfg2); x=x[x.deployment_mode=="Serverless"].copy(); x["cold_start_penalty_ms"]=penalty; parts.append(x)
abl=__import__("pandas").concat(parts,ignore_index=True); abl.to_csv("results/ablation_runs.csv",index=False)
fig,ax=plt.subplots(figsize=(8.5,5))
for pen,g in abl.groupby("cold_start_penalty_ms"):
    q=g.groupby("request_rate_rps").p99_latency_ms.mean()
    ax.plot(q.index,q.values,marker="o",label=f"{pen} ms")
ax.set_xscale("log"); ax.set_xlabel("Request rate (rps, log scale)"); ax.set_ylabel("p99 latency (ms)"); ax.set_title("Cold-start penalty ablation")
ax.grid(True,alpha=.3); ax.legend(title="Penalty"); fig.tight_layout(); fig.savefig("figures/cold_start_ablation.png",dpi=180); plt.close(fig)
print("Done. See results/ and figures/."); print(crossover(s).to_string(index=False))
