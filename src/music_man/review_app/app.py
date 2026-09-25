"""
Streamlit approval queue for proposed royalty holds. Approving is the only
place anything "real" happens - it flips status and appends a line to
audit_log.csv. Never a real payment action, same principle as NorthStar's
audit-log-only "execution."

Run with: streamlit run src/music_man/review_app/app.py
"""

from __future__ import annotations

import csv

import streamlit as st

from music_man.agent.queue import Hold, decide_hold, list_holds
from music_man.paths import AUDIT_LOG_PATH

st.set_page_config(page_title="Music Man - Royalty Hold Review", layout="centered")

st.title("Royalty Hold Approval Queue")
st.caption(
    "Nothing here was withheld automatically - the agent can only queue a proposal. "
    "Approving or rejecting is the only thing that decides what actually happens."
)


def _append_audit_log(hold: Hold) -> None:
    is_new = not AUDIT_LOG_PATH.exists()
    AUDIT_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(AUDIT_LOG_PATH, "a", newline="") as f:
        writer = csv.writer(f)
        if is_new:
            writer.writerow(["hold_id", "artist_id", "period", "decided_at", "decided_by", "status"])
        writer.writerow(
            [hold.id, hold.artist_id, hold.period, hold.decided_at, hold.decided_by, hold.status]
        )


pending = list_holds(status="pending_approval")
st.subheader(f"Pending ({len(pending)})")

if not pending:
    st.info("No holds are waiting on review.")

for hold in pending:
    with st.container(border=True):
        st.markdown(f"**Artist `{hold.artist_id}`** · period `{hold.period}`")
        st.caption(hold.created_at)
        # st.write()/st.markdown() render $...$ as LaTeX - the agent's
        # rationale is plain prose (and may contain literal dollar
        # amounts), so use st.text() to avoid mangling it.
        st.text(hold.rationale)
        col1, col2 = st.columns(2)
        if col1.button("Approve", key=f"approve-{hold.id}"):
            decided = decide_hold(hold.id, "executed", "staff-demo-user")
            _append_audit_log(decided)
            st.rerun()
        if col2.button("Reject", key=f"reject-{hold.id}"):
            decide_hold(hold.id, "rejected", "staff-demo-user")
            st.rerun()

st.subheader("Recent decisions")
decided = [h for h in list_holds() if h.status != "pending_approval"][:10]
if not decided:
    st.caption("No decisions yet.")
for hold in decided:
    st.write(f"Artist `{hold.artist_id}` · {hold.period} — **{hold.status}**")
