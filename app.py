import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import asdict
from src.experiment import Config,run,summarize,crossover,DEFAULT_RATES,MODES

st.set_page_config(page_title="ML Inference Cost-Latency Lab",layout="wide")
st.title("ML Inference Cost–Latency Crossover Lab")
st.caption("Sahil Parija • 23BCE1379 • Faculty: Dr. Indra Priyadarshini S")
st.warning("Evidence boundary: local model inference is timed on this machine. Serving overheads, cold starts and costs are illustrative simulations, not cloud-provider measurements.")
with st.sidebar:
    st.header("Experiment controls")
    w=st.selectbox("Workload",["Small tabular classification","Medium image-like classification","Regression workload"])
    reps=st.slider("Independent repeats",3,10,5)
    n=st.slider("Requests per trial",50,1000,250,step=50)
    rates=st.multiselect("Request rates (rps)",DEFAULT_RATES,default=DEFAULT_RATES)
    st.subheader("Cold-start assumptions")
    cold=st.slider("Cold-start penalty (ms)",0,1000,450,step=50)
    prob=st.slider("Cold-start probability at 0.1 rps",0.0,.8,.25,step=.05)
    st.subheader("Illustrative cost inputs")
    req=st.number_input("Serverless request price per million ($)",0.0,100.0,2.0,step=.1)
    cc=st.number_input("Container hourly cost ($)",.001,10.0,.06,step=.01)
    mc=st.number_input("Managed endpoint hourly cost ($)",.001,10.0,.12,step=.01)
    go=st.button("Run experiment",type="primary",use_container_width=True)
if "raw" not in st.session_state: st.session_state.raw=None
if go:
    if not rates: st.error("Select at least one rate.")
    else:
        cfg=Config(workload=w,repeats=reps,requests_per_trial=n,cold_start_ms=cold,cold_probability_at_0_1_rps=prob,
                   serverless_request_price_per_million=req,container_hourly_cost=cc,managed_hourly_cost=mc)
        with st.spinner("Running repeated trials..."): st.session_state.raw=run(cfg,rates)
if st.session_state.raw is None:
    st.info("Choose settings and click Run experiment.")
    st.markdown("### Live demo sequence\n1. Run defaults.\n2. Inspect cost and p99 latency.\n3. Increase cold-start penalty and rerun.\n4. Change workload.\n5. Export results and discuss validity.")
    st.stop()
raw=st.session_state.raw; summary=summarize(raw); cross=crossover(summary)
c=st.columns(4); c[0].metric("Workload",raw.workload.iloc[0]); c[1].metric("Independent repeats",int(raw.repeat.max()))
c[2].metric("Rates tested",raw.request_rate_rps.nunique()); c[3].metric("Requests per trial",int(raw.requests_per_trial.iloc[0]))
t1,t2,t3,t4=st.tabs(["Cost","Latency","Crossover","Data & validity"])
with t1:
    fig,ax=plt.subplots(figsize=(9,4.5))
    for mode in MODES:
        g=summary[summary.deployment_mode==mode].sort_values("request_rate_rps")
        ax.errorbar(g.request_rate_rps,g.cost_per_1000_usd_mean,yerr=g.cost_per_1000_usd_ci95_halfwidth,marker="o",capsize=3,label=mode)
    ax.set_xscale("log"); ax.set_xlabel("Request rate (rps, log scale)"); ax.set_ylabel("Cost / 1,000 inferences (illustrative $)")
    ax.grid(True,alpha=.3); ax.legend(); fig.tight_layout(); st.pyplot(fig)
with t2:
    fig,ax=plt.subplots(figsize=(9,4.5))
    for mode in MODES:
        g=summary[summary.deployment_mode==mode].sort_values("request_rate_rps")
        ax.errorbar(g.request_rate_rps,g.p99_latency_ms_mean,yerr=g.p99_latency_ms_ci95_halfwidth,marker="o",capsize=3,label=mode)
    ax.set_xscale("log"); ax.set_xlabel("Request rate (rps, log scale)"); ax.set_ylabel("p99 latency (ms)")
    ax.grid(True,alpha=.3); ax.legend(); fig.tight_layout(); st.pyplot(fig)
with t3:
    st.dataframe(cross,use_container_width=True,hide_index=True)
    st.caption("Least-cost option is conditional on the current parameter choices; not a universal cloud-pricing conclusion.")
with t4:
    st.subheader("Run-level data"); st.dataframe(raw,use_container_width=True,hide_index=True)
    st.subheader("Summary statistics"); st.dataframe(summary,use_container_width=True,hide_index=True)
    st.download_button("Download raw CSV",raw.to_csv(index=False),"raw_runs.csv","text/csv")
    st.download_button("Download summary CSV",summary.to_csv(index=False),"summary_statistics.csv","text/csv")
    st.download_button("Download crossover CSV",cross.to_csv(index=False),"crossover.csv","text/csv")
    st.markdown("**Threats to validity:** serving overhead and cold starts are simulated; prices are illustrative; datasets do not represent all production models; local timings depend on machine load. Provider-specific claims require controlled cloud deployments and billing evidence.")
