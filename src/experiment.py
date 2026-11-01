from dataclasses import dataclass
import time
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer, load_digits, load_diabetes
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error

MODES=["Serverless","Continuous container","Managed endpoint"]
DEFAULT_RATES=[0.1,0.25,0.5,1,2,5,10]

@dataclass
class Config:
    workload:str="Small tabular classification"
    repeats:int=5
    requests_per_trial:int=250
    seed:int=1379
    serverless_request_price_per_million:float=2.0
    serverless_gb_second_price:float=0.0000166667
    serverless_memory_gb:float=1.0
    container_hourly_cost:float=0.06
    managed_hourly_cost:float=0.12
    managed_premium:float=1.15
    serverless_base_ms:float=8.0
    cold_start_ms:float=450.0
    cold_probability_at_0_1_rps:float=0.25
    container_base_ms:float=12.0
    managed_base_ms:float=25.0
    saturation_rps:float=8.0

def workload(name,seed):
    if name=="Small tabular classification":
        d=load_breast_cancer(); X,y=d.data,d.target
        model=make_pipeline(StandardScaler(),LogisticRegression(max_iter=1000,random_state=seed)); task="classification"
    elif name=="Medium image-like classification":
        d=load_digits(); X,y=d.data,d.target
        model=make_pipeline(StandardScaler(),LogisticRegression(max_iter=1500,random_state=seed)); task="classification"
    else:
        d=load_diabetes(); X,y=d.data,d.target
        model=make_pipeline(StandardScaler(),Ridge(alpha=1)); task="regression"
    Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.25,random_state=seed,stratify=y if task=="classification" else None)
    model.fit(Xtr,ytr)
    quality=accuracy_score(yte,model.predict(Xte)) if task=="classification" else mean_squared_error(yte,model.predict(Xte))**.5
    return model,Xte[:1],float(quality),"holdout accuracy" if task=="classification" else "holdout RMSE",task

def cold_p(rate,cfg):
    b=cfg.cold_probability_at_0_1_rps
    return 0.0 if b<=0 else float(min(.85,max(.01,b*(.1/rate)**.35)))

def cost(mode,rate,cfg):
    if mode=="Serverless":
        p=cold_p(rate,cfg)
        duration=(cfg.serverless_base_ms+p*cfg.cold_start_ms)/1000
        return 1000*cfg.serverless_request_price_per_million/1e6+1000*cfg.serverless_memory_gb*duration*cfg.serverless_gb_second_price
    hourly=cfg.container_hourly_cost if mode=="Continuous container" else cfg.managed_hourly_cost*cfg.managed_premium
    return hourly*(1000/rate)/3600

def trial(model,sample,mode,rate,cfg,rng):
    n=cfg.requests_per_trial
    times=[]
    for _ in range(n):
        t=time.perf_counter(); model.predict(sample); times.append((time.perf_counter()-t)*1000)
    if mode=="Serverless":
        cold=rng.random(n)<cold_p(rate,cfg)
        overhead=cfg.serverless_base_ms*rng.lognormal(0,.20,n)+cold*cfg.cold_start_ms
        cr=float(cold.mean())
    elif mode=="Continuous container":
        overload=max(0,rate/cfg.saturation_rps-1)
        overhead=cfg.container_base_ms*rng.lognormal(0,.18,n)*(1+4*overload); cr=0.0
    else:
        overload=max(0,rate/(cfg.saturation_rps*.8)-1)
        overhead=cfg.managed_base_ms*rng.lognormal(0,.20,n)*(1+3*overload); cr=0.0
    lat=overhead+np.asarray(times)
    return dict(cost_per_1000_usd=cost(mode,rate,cfg),p50_latency_ms=float(np.percentile(lat,50)),
                p99_latency_ms=float(np.percentile(lat,99)),cold_start_rate=cr,
                local_inference_median_ms=float(np.median(times)))

def run(cfg=None,rates=None):
    cfg=cfg or Config(); rates=rates or DEFAULT_RATES
    model,sample,q,qlabel,task=workload(cfg.workload,cfg.seed); rows=[]
    for ri,rate in enumerate(rates):
        for mi,mode in enumerate(MODES):
            for rep in range(max(3,cfg.repeats)):
                rng=np.random.default_rng(cfg.seed+ri*10000+mi*100+rep)
                rows.append(dict(workload=cfg.workload,task=task,request_rate_rps=float(rate),deployment_mode=mode,
                    repeat=rep+1,requests_per_trial=cfg.requests_per_trial,model_quality_metric=qlabel,
                    model_quality_value=q,**trial(model,sample,mode,float(rate),cfg,rng),
                    evidence_type="local inference timing + simulated serving overhead and illustrative prices"))
    return pd.DataFrame(rows)

def summarize(raw):
    metrics=["cost_per_1000_usd","p50_latency_ms","p99_latency_ms","cold_start_rate"]
    rows=[]
    for keys,g in raw.groupby(["workload","request_rate_rps","deployment_mode"]):
        row=dict(zip(["workload","request_rate_rps","deployment_mode"],keys))
        for m in metrics:
            vals=g[m].astype(float); row[m+"_mean"]=vals.mean(); row[m+"_std"]=vals.std(ddof=1)
            row[m+"_count"]=len(vals); row[m+"_ci95_halfwidth"]=1.96*vals.std(ddof=1)/np.sqrt(len(vals)) if len(vals)>1 else 0
        rows.append(row)
    return pd.DataFrame(rows)

def crossover(summary):
    out=[]
    for rate,g in summary.groupby("request_rate_rps"):
        r=g.loc[g.cost_per_1000_usd_mean.idxmin()]
        out.append(dict(request_rate_rps=rate,least_cost_mode=r.deployment_mode,
                        estimated_cost_per_1000_usd=r.cost_per_1000_usd_mean))
    return pd.DataFrame(out)
