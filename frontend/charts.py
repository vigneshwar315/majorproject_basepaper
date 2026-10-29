"""Plotly figure builders (pure functions, no Streamlit import)."""

from __future__ import annotations

from typing import Optional

import pandas as pd
import plotly.graph_objects as go

from paper_replication.constraints import plan_route_results
from paper_replication.schemas import PICKUP, Instance, Plan

PALETTE = ["#1F4E79", "#0F6E56", "#8A5200", "#5B3FA6", "#A31F34", "#4A6B8A", "#6B7F2A", "#8C4B7E"]
SEL = "#A31F34"
BASE = "#4A6B8A"


def _layout(fig: go.Figure, title: str = "", height: int = 380) -> go.Figure:
    fig.update_layout(title=dict(text=title, font=dict(size=14)), height=height, margin=dict(l=10, r=10, t=44, b=10), paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="#FFFFFF", font=dict(color="#1B2430"), legend=dict(orientation="h", y=-0.2))
    fig.update_xaxes(gridcolor="#E3E8EE", zeroline=False)
    fig.update_yaxes(gridcolor="#E3E8EE", zeroline=False)
    return fig


def metric_scatter(df: pd.DataFrame, x: str, y: str, selected: Optional[str]) -> go.Figure:
    d = df[df["feasible"]]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=d[x], y=d[y], mode="markers", text=d["plan"], name="feasible plans", marker=dict(size=9, color=BASE, opacity=0.8),
                             hovertemplate="%{text}<br>" + x + "=%{x}<br>" + y + "=%{y}<extra></extra>"))
    if selected is not None and selected in set(d["plan"]):
        s = d[d["plan"] == selected]
        fig.add_trace(go.Scatter(x=s[x], y=s[y], mode="markers+text", text=s["plan"], textposition="top center", name="selected by MCDM",
                                 marker=dict(size=14, color=SEL, symbol="diamond", line=dict(width=1, color="#fff"))))
    fig.update_xaxes(title=x)
    fig.update_yaxes(title=y)
    return _layout(fig, f"{y} against {x}")


def metric_bar(df: pd.DataFrame, metric: str, selected: Optional[str]) -> go.Figure:
    d = df[df["feasible"]]
    colors = [SEL if p == selected else BASE for p in d["plan"]]
    fig = go.Figure(go.Bar(x=d["plan"], y=d[metric], marker_color=colors, hovertemplate="%{x}<br>" + metric + "=%{y}<extra></extra>"))
    fig.update_yaxes(title=metric)
    return _layout(fig, f"{metric} per candidate plan" + (" (selected plan in red)" if selected else ""))


def route_figure(inst: Instance, plan: Plan, day: int) -> go.Figure:
    """Route sequences on the instance's planar coordinates. NOT a geographic map: coordinates are abstract (synthetic)."""
    reqs = inst.request_map()
    results = plan_route_results(inst, plan)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[inst.depot.x], y=[inst.depot.y], mode="markers+text", text=["depot"], textposition="bottom center", name="depot",
                             marker=dict(size=14, color="#1B2430", symbol="square")))
    i = 0
    for rt in sorted((r for r in plan.routes if r.day == day), key=lambda r: r.vehicle_id):
        xs, ys, labels, hov = [inst.depot.x], [inst.depot.y], [""], ["depot"]
        res = results[(rt.day, rt.vehicle_id)]
        for st, tm in zip(rt.stops, res.stops):
            r = reqs[st.request_id]
            loc = r.pickup if st.kind == PICKUP else r.delivery
            xs.append(loc.x)
            ys.append(loc.y)
            labels.append(f"{st.request_id}{st.kind}")
            hov.append(f"{st.request_id} {'pickup' if st.kind == PICKUP else 'delivery'}<br>service start {int(tm.start) // 60:02d}:{int(tm.start) % 60:02d}")
        xs.append(inst.depot.x)
        ys.append(inst.depot.y)
        labels.append("")
        hov.append("depot")
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines+markers+text", text=labels, textposition="top center", name=f"vehicle {rt.vehicle_id}", hovertext=hov,
                                 hoverinfo="text", line=dict(color=PALETTE[i % len(PALETTE)], width=2), marker=dict(size=8)))
        i += 1
    rejected = [reqs[rid] for rid in plan.rejected_ids() if rid in reqs]
    if rejected:
        fig.add_trace(go.Scatter(x=[r.pickup.x for r in rejected] + [r.delivery.x for r in rejected], y=[r.pickup.y for r in rejected] + [r.delivery.y for r in rejected],
                                 mode="markers", name="rejected request (not served)", marker=dict(size=9, color="#9AA5B1", symbol="x"),
                                 hovertext=[f"{r.id} pickup (rejected)" for r in rejected] + [f"{r.id} delivery (rejected)" for r in rejected], hoverinfo="text"))
    fig.update_xaxes(title="x (planar coordinate, not geographic)")
    fig.update_yaxes(title="y (planar coordinate, not geographic)", scaleanchor="x", scaleratio=1)
    return _layout(fig, f"Routes of plan {plan.plan_id} on day {day} (labels: request id + P for pickup, D for delivery)", height=560)


def _hhmm(tw) -> str:
    if tw is None:
        return "none"
    return f"{int(tw.earliest) // 60:02d}:{int(tw.earliest) % 60:02d}-{int(tw.latest) // 60:02d}:{int(tw.latest) % 60:02d}"


def instance_figure(inst: Instance, max_links: int = 400) -> go.Figure:
    """Depot, every pickup (circle) and delivery (triangle), with a dotted line joining each pair. Planar coordinates."""
    fig = go.Figure()
    reqs = inst.requests
    for r in reqs[:max_links]:
        fig.add_trace(go.Scatter(x=[r.pickup.x, r.delivery.x], y=[r.pickup.y, r.delivery.y], mode="lines", showlegend=False, hoverinfo="skip",
                                 line=dict(color="#9DB3C8", width=1, dash="dot")))
    high = [r for r in reqs if r.priority == 1]
    norm = [r for r in reqs if r.priority != 1]

    def hover(r, kind):
        tw = r.pickup_tw if kind == "pickup" else r.delivery_tw
        return (f"<b>{r.id}</b> {kind}<br>arrives day {r.arrival_day}, demand {r.demand}<br>priority {'high' if r.priority == 1 else 'normal'}"
                f"<br>time window {_hhmm(tw)}<br>ZIP {(r.pickup if kind == 'pickup' else r.delivery).zip}")

    for group, label, color in ((norm, "normal", "#1F4E79"), (high, "high priority", "#A31F34")):
        if not group:
            continue
        fig.add_trace(go.Scatter(x=[r.pickup.x for r in group], y=[r.pickup.y for r in group], mode="markers", name=f"pickup ({label})",
                                 marker=dict(size=9, color=color, symbol="circle"), hovertext=[hover(r, "pickup") for r in group], hoverinfo="text"))
        fig.add_trace(go.Scatter(x=[r.delivery.x for r in group], y=[r.delivery.y for r in group], mode="markers", name=f"delivery ({label})",
                                 marker=dict(size=10, color=color, symbol="triangle-up", line=dict(width=1, color="#FFFFFF")),
                                 hovertext=[hover(r, "delivery") for r in group], hoverinfo="text"))
    tw = [r for r in reqs if r.pickup_tw or r.delivery_tw]
    if tw:
        fig.add_trace(go.Scatter(x=[(r.pickup if r.pickup_tw else r.delivery).x for r in tw], y=[(r.pickup if r.pickup_tw else r.delivery).y for r in tw],
                                 mode="markers", name="has a time window", hoverinfo="skip",
                                 marker=dict(size=17, color="rgba(0,0,0,0)", line=dict(width=1.5, color="#8A5200"))))
    fig.add_trace(go.Scatter(x=[inst.depot.x], y=[inst.depot.y], mode="markers+text", text=["depot"], textposition="bottom center", name="depot",
                             marker=dict(size=16, color="#1B2430", symbol="square")))
    fig.update_xaxes(title="x (planar coordinate, not geographic)")
    fig.update_yaxes(title="y", scaleanchor="x", scaleratio=1)
    return _layout(fig, f"{len(reqs)} requests: pickups (circles), deliveries (triangles), dotted line = same request", height=560)


def parallel_figure(df: pd.DataFrame, metric_ids: list[str], selected: Optional[str]) -> go.Figure:
    d = df[df["feasible"]].reset_index(drop=True)
    dims = [dict(label=m, values=d[m]) for m in metric_ids]
    color = [1.0 if p == selected else 0.0 for p in d["plan"]]
    fig = go.Figure(go.Parcoords(line=dict(color=color, colorscale=[[0, "#9DB3C8"], [1, SEL]]), dimensions=dims))
    return _layout(fig, "All feasible plans across metrics (selected plan in red)", height=420)


def bar_pair(labels: list[str], values: list[Optional[float]], title: str, ytitle: str, color: str = BASE) -> go.Figure:
    fig = go.Figure(go.Bar(x=labels, y=[v if v is not None else 0 for v in values], marker_color=color, text=[("n/a" if v is None else f"{v:.3g}") for v in values], textposition="outside"))
    fig.update_yaxes(title=ytitle)
    return _layout(fig, title, height=330)


def grouped_bars(categories: list[str], series: dict[str, list[Optional[float]]], title: str, ytitle: str) -> go.Figure:
    fig = go.Figure()
    for i, (name, vals) in enumerate(series.items()):
        fig.add_trace(go.Bar(name=name, x=categories, y=[v if v is not None else 0 for v in vals], marker_color=PALETTE[i % len(PALETTE)],
                             text=[("n/a" if v is None else f"{v:.3g}") for v in vals], textposition="outside"))
    fig.update_layout(barmode="group")
    fig.update_yaxes(title=ytitle)
    return _layout(fig, title, height=360)
