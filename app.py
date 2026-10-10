"""Executive overview page for RetailLens-AI."""

import streamlit as st

from src.dashboard import daily_from_transactions, empty_state, filters, load_data, metric_strip, page_shell, revenue


page_shell("Overview", "A clear view of historical sales performance and the signals that deserve attention.")
transactions = load_data()[0]
scoped = filters(transactions, "overview")

if scoped.empty:
    empty_state()
    st.stop()

metric_strip(scoped, "Selected-period")
series = daily_from_transactions(scoped)
left, right = st.columns([1.65, 1])
with left:
    st.subheader("Recorded revenue trend")
    st.caption("Daily net recorded revenue from transaction rows matching the current sidebar filters.")
    if len(series):
        st.line_chart(series.rename("Recorded revenue"), color="#148a7c", height=300)
with right:
    st.subheader("Executive summary")
    total = float(scoped["Revenue"].sum())
    active_days = max(scoped["Date"].nunique(), 1)
    st.markdown(f"**{len(scoped):,}** transaction lines across **{active_days:,}** active dates contributed **{revenue(total)}** recorded revenue in this selection.")
    best = series.idxmax() if not series.empty else None
    if best is not None:
        st.info(f"Highest selected daily total: **{revenue(series.loc[best])}** on **{best:%d %b %Y}**.")
    st.caption("Filters scope the historical transaction view. They do not alter the packaged forecast.")

st.subheader("At a glance")
if not series.empty:
    latest = series.sort_index().tail(7).mean()
    previous = series.sort_index().iloc[-14:-7].mean()
    trend = "—" if len(series) < 14 or previous == 0 else f"{(latest-previous)/abs(previous)*100:+.1f}%"
    k = st.columns(3)
    k[0].metric("Average per active date", revenue(scoped.groupby("Date")["Revenue"].sum().mean()))
    k[1].metric("Latest 7 active dates average", revenue(latest))
    k[2].metric("Change vs prior 7 active dates", trend, help="Computed from selected transaction dates; this is descriptive, not predictive.")
st.markdown("#### Review cues")
cues = []
cancel = scoped["IsCancellation"]
if cancel.any():
    cues.append(f"{int(cancel.sum()):,} selected transaction lines are marked as cancellation invoices.")
if not series.empty and series.nunique() > 1:
    q1, q3 = series.quantile([.25, .75])
    flagged = int((series > q3 + 1.5 * (q3-q1)).sum())
    if flagged:
        cues.append(f"{flagged} selected date(s) exceed the historical upper IQR threshold; inspect context in Decision Center.")
if cues:
    for cue in cues:
        st.markdown(f"- {cue}")
else:
    st.write("No cancellation or upper-threshold review cues were identified in this selection.")
st.caption("Historical data: 2010–2011. Recorded revenue is not profit; zeros and missing dates follow the project’s existing data conventions.")
