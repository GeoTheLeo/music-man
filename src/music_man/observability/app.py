"""
Agent observability dashboard: token usage, estimated cost, latency, failures,
and retries per model, across every recorded run, with a drill-down into any
run's plan and tool-call trace. Built for the moment something breaks: recent
failures come first.

Run with: streamlit run src/music_man/observability/app.py
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import altair as alt
import pandas as pd
import streamlit as st

from music_man.observability.data import load

st.set_page_config(page_title="Music Man - Agent Observability", page_icon=":material/monitoring:", layout="wide")

# Validated dark-mode categorical slots (dataviz reference palette, checked against #0e1117).
SERIES = ["#3987e5", "#d95926", "#199e70", "#c98500"]
# Fixed entity -> colour so a model keeps its colour when filters change.
MODEL_ORDER = ["claude-opus-5", "claude-haiku-4-5", "claude-sonnet-5"]
TOKEN_ORDER = ["uncached input", "cache read", "cache write", "output"]
# Failure kinds are states, so they take the reserved status palette, always with a text label.
STATUS = {
    "application error": "#d03b3b",
    "billing": "#d03b3b",
    "provider error": "#ec835a",
    "timeout": "#ec835a",
    "rate limit": "#fab219",
    "budget exhausted": "#fab219",
    "refusal": "#fab219",
}
INK, MUTED, GRID, AXIS = "#c3c2b7", "#898781", "#2c2c2a", "#383835"


@st.cache_data(ttl=60, show_spinner=False)
def load_cached():
    return load()


def style(chart: alt.Chart, height: int = 260) -> alt.Chart:
    return (
        chart.properties(height=height)
        .configure(background="transparent")
        .configure_view(stroke=None)
        .configure_axis(labelColor=MUTED, titleColor=INK, gridColor=GRID, domainColor=AXIS, tickColor=AXIS,
                        labelFontSize=12, titleFontWeight="normal")
        .configure_legend(labelColor=INK, titleColor=INK, orient="top", labelFontSize=12)
    )


runs, model_usage, records = load_cached()

st.title("Agent observability")
st.caption("Every Music Man run - live, webhook-triggered, and eval - in one place. Costs are estimates from list prices.")

if runs.empty:
    st.info("No run records yet. Run the agent or the evals and they will appear here.", icon=":material/info:")
    st.stop()

# ---------- one filter row, scoping everything below ----------
with st.container(horizontal=True, vertical_alignment="bottom"):
    window = st.segmented_control("Time range", ["24 h", "7 days", "30 days", "All"], default="All")
    sources = st.pills("Source", sorted(runs["source"].unique()), default=sorted(runs["source"].unique()),
                       selection_mode="multi")
    engines = st.pills("Engine", sorted(runs["engine"].unique()), default=sorted(runs["engine"].unique()),
                       selection_mode="multi")
    if st.button("Refresh", icon=":material/refresh:"):
        load_cached.clear()
        st.rerun()

since = {"24 h": timedelta(days=1), "7 days": timedelta(days=7), "30 days": timedelta(days=30)}.get(window or "All")
view = runs[runs["source"].isin(sources or []) & runs["engine"].isin(engines or [])]
if since is not None:
    view = view[view["started_at"] >= datetime.now(timezone.utc) - since]
if view.empty:
    st.warning("No runs match these filters.", icon=":material/filter_alt_off:")
    st.stop()
usage = model_usage[model_usage["run_id"].isin(view["run_id"])]

# ---------- KPI tiles ----------
failed = view[view["outcome"] != "completed"]
measured_retries = view["retries"].dropna()
with st.container(horizontal=True):
    st.metric("Runs", f"{len(view)}", border=True)
    st.metric("Success rate", f"{(1 - len(failed) / len(view)):.0%}", border=True)
    st.metric("Failures", f"{len(failed)}", border=True)
    st.metric("Mean cost / run", f"${view['cost_usd'].mean():.3f}", border=True)
    st.metric("Total cost", f"${view['cost_usd'].sum():.2f}", border=True)
    st.metric("p95 latency", f"{view['duration_s'].quantile(0.95):.0f} s", border=True)
    st.metric("Retries", f"{int(measured_retries.sum())}" if len(measured_retries) else "n/a", border=True,
              help="HTTP attempts beyond the first, counted at the client. Not measured for older records "
                   "or the LangGraph engine.")

# ---------- incident view first: recent failures ----------
with st.container(border=True):
    st.subheader(":material/error: Recent failures")
    if failed.empty:
        st.success("No failed runs in this slice.", icon=":material/check_circle:")
    else:
        failure_table = failed[["started_at", "engine", "error_kind", "error", "run_id"]].rename(
            columns={"started_at": "time", "error_kind": "kind"})
        st.dataframe(
            failure_table, hide_index=True,
            column_config={
                "time": st.column_config.DatetimeColumn(format="YYYY-MM-DD HH:mm"),
                "error": st.column_config.TextColumn(width="large"),
            },
        )
        by_kind = failed.groupby("error_kind").size().reset_index(name="runs")
        kinds = [k for k in STATUS if k in set(by_kind["error_kind"])]
        bars = (
            alt.Chart(by_kind)
            .mark_bar(cornerRadiusEnd=4, size=18)
            .encode(
                y=alt.Y("error_kind:N", sort=kinds, title=None),
                x=alt.X("runs:Q", title="Failed runs", axis=alt.Axis(tickMinStep=1),
                        scale=alt.Scale(domainMax=int(by_kind["runs"].max() * 1.2) + 1)),
                color=alt.Color("error_kind:N", legend=None,
                                scale=alt.Scale(domain=kinds, range=[STATUS[k] for k in kinds])),
                tooltip=[alt.Tooltip("error_kind:N", title="Kind"), alt.Tooltip("runs:Q", title="Failed runs")],
            )
        )
        counts = bars.mark_text(align="left", dx=6).encode(text="runs:Q", color=alt.value(INK))
        st.altair_chart(style(bars + counts, 40 + 32 * len(kinds)))

# ---------- cost and tokens by model ----------
models = [m for m in MODEL_ORDER if m in set(usage["model"])] + sorted(set(usage["model"]) - set(MODEL_ORDER))
model_colors = alt.Scale(domain=models, range=SERIES[: len(models)])

left, right = st.columns(2)
with left, st.container(border=True):
    st.subheader("Estimated cost by model")
    cost = usage.groupby("model", as_index=False)["cost_usd"].sum()
    cost_chart = (
        alt.Chart(cost)
        .mark_bar(cornerRadiusEnd=4, size=22)
        .encode(
            y=alt.Y("model:N", sort=models, title=None),
            x=alt.X("cost_usd:Q", title="USD", scale=alt.Scale(domainMax=float(cost["cost_usd"].max()) * 1.18)),
            color=alt.Color("model:N", scale=model_colors, legend=None),
            tooltip=[alt.Tooltip("model:N", title="Model"), alt.Tooltip("cost_usd:Q", title="Cost", format="$.3f")],
        )
    )
    labels = cost_chart.mark_text(align="left", dx=6, color=INK).encode(
        text=alt.Text("cost_usd:Q", format="$.2f"), color=alt.value(INK))
    st.altair_chart(style(cost_chart + labels, 60 + 40 * len(models)))
    with st.expander("Table view"):
        st.dataframe(cost, hide_index=True, column_config={"cost_usd": st.column_config.NumberColumn("Cost", format="$%.4f")})

with right, st.container(border=True):
    st.subheader("Tokens by model")
    if {"input_tokens", "output_tokens"}.issubset(usage.columns):
        tokens = usage.dropna(subset=["input_tokens"]).groupby("model", as_index=False)[
            ["input_tokens", "cache_read_input_tokens", "cache_creation_input_tokens", "output_tokens"]].sum()
        long = tokens.melt("model", var_name="type", value_name="tokens").replace({"type": {
            "input_tokens": "uncached input", "cache_read_input_tokens": "cache read",
            "cache_creation_input_tokens": "cache write", "output_tokens": "output"}})
        token_chart = (
            alt.Chart(long)
            .mark_bar(size=22, stroke="#0e1117", strokeWidth=2)
            .encode(
                y=alt.Y("model:N", sort=models, title=None),
                x=alt.X("tokens:Q", title="Tokens"),
                color=alt.Color("type:N", title="Token type", sort=TOKEN_ORDER,
                                scale=alt.Scale(domain=TOKEN_ORDER, range=SERIES)),
                order=alt.Order("type_order:Q"),
                tooltip=[alt.Tooltip("model:N", title="Model"), alt.Tooltip("type:N", title="Type"),
                         alt.Tooltip("tokens:Q", title="Tokens", format=",")],
            )
            .transform_calculate(type_order=f"indexof({TOKEN_ORDER}, datum.type)")
        )
        st.altair_chart(style(token_chart, 80 + 40 * len(models)))
        with st.expander("Table view"):
            st.dataframe(tokens, hide_index=True)
    else:
        st.caption("Per-model token breakdown is recorded for runs made after the observability update.")

# ---------- latency ----------
with st.container(border=True):
    st.subheader("Run latency")
    p = view.groupby("engine")["duration_s"].quantile([0.5, 0.95]).unstack()
    st.caption(" · ".join(f"**{e}** p50 {row[0.5]:.0f} s, p95 {row[0.95]:.0f} s" for e, row in p.iterrows()))
    latency_chart = (
        alt.Chart(view)
        .mark_circle(size=80, opacity=0.8, color=SERIES[0], stroke="#0e1117", strokeWidth=2)
        .encode(
            x=alt.X("duration_s:Q", title="Seconds per run"),
            y=alt.Y("engine:N", title=None),
            yOffset=alt.YOffset("jitter:Q"),
            tooltip=[alt.Tooltip("run_id:N", title="Run"), alt.Tooltip("duration_s:Q", title="Seconds", format=".1f"),
                     alt.Tooltip("outcome:N", title="Outcome"), alt.Tooltip("cost_usd:Q", title="Cost", format="$.3f")],
        )
        .transform_calculate(jitter="random()")
    )
    st.altair_chart(style(latency_chart, 60 + 50 * view["engine"].nunique()))

# ---------- drill-down: one run's trace ----------
with st.container(border=True):
    st.subheader(":material/account_tree: Run trace")
    options = list(failed["run_id"]) + [r for r in view["run_id"] if r not in set(failed["run_id"])]
    run_id = st.selectbox("Run", options, help="Failed runs are listed first.")
    record = records[run_id]
    row = view[view["run_id"] == run_id].iloc[0]
    with st.container(horizontal=True):
        st.metric("Outcome", record.get("outcome", "?"), border=True)
        st.metric("Engine", row["engine"], border=True)
        st.metric("Cost", f"${row['cost_usd']:.3f}", border=True)
        st.metric("Duration", f"{row['duration_s']:.1f} s", border=True)
        st.metric("Tool calls", f"{row['tool_calls']} ({row['tool_errors']} errors)", border=True)
    if record.get("error"):
        st.error(record["error"], icon=":material/error:")
    st.markdown(f"**Request:** {record.get('user_message', '')}")
    if record.get("plan"):
        st.markdown("**Plan**")
        st.dataframe(pd.DataFrame(record["plan"]), hide_index=True)
    calls = pd.DataFrame(record.get("tool_calls") or [])
    if not calls.empty:
        calls["target"] = calls["input"].apply(lambda i: " / ".join(str(v) for k, v in i.items() if k in ("artist_id", "period")))
        calls["result"] = calls["result"].str.slice(0, 160)
        st.markdown("**Tool calls**")
        st.dataframe(calls[["name", "target", "is_error", "result"]], hide_index=True,
                     column_config={"is_error": st.column_config.CheckboxColumn("error"),
                                    "result": st.column_config.TextColumn(width="large")})
    with st.expander("Usage detail"):
        st.json(record.get("usage") or {})
