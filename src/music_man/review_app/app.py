"""
Streamlit approval queue for proposed royalty holds. Approving is the only
place anything "real" happens - it flips status and appends a line to
audit_log.csv. Never a real payment action, same principle as NorthStar's
audit-log-only "execution."

Executed holds can be released (the rollback path) with a required reason.
Every decision goes through the queue's compare-and-set state machine, so a
double click or two reviewers deciding at once can't apply twice.

Run with: streamlit run src/music_man/review_app/app.py
"""

from __future__ import annotations

import csv

import streamlit as st

from music_man.agent.queue import (
    EXECUTED,
    PENDING,
    Hold,
    InvalidTransition,
    decide_hold,
    list_events,
    list_holds,
    release_hold,
)
from music_man.paths import AUDIT_LOG_PATH

REVIEWER = "staff-demo-user"

st.set_page_config(page_title="Music Man - Royalty Hold Review", layout="centered")

st.title("Royalty Hold Approval Queue")
st.caption(
    "Nothing here was withheld automatically - the agent can only queue a proposal. "
    "Approving or rejecting is the only thing that decides what actually happens."
)


def _append_audit_log(hold: Hold, action: str, at: str | None, by: str | None) -> None:
    is_new = not AUDIT_LOG_PATH.exists()
    AUDIT_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(AUDIT_LOG_PATH, "a", newline="") as f:
        writer = csv.writer(f)
        if is_new:
            writer.writerow(["hold_id", "artist_id", "period", "decided_at", "decided_by", "status"])
        writer.writerow([hold.id, hold.artist_id, hold.period, at, by, action])


def _apply(action) -> None:
    try:
        action()
    except InvalidTransition as exc:
        st.warning(f"Not applied: {exc}. The queue has been refreshed.")
        return
    st.rerun()


pending = list_holds(status=PENDING)
st.subheader(f"Pending ({len(pending)})")

if not pending:
    st.info("No holds are waiting on review.")

for hold in pending:
    with st.container(border=True):
        st.markdown(f"**Artist `{hold.artist_id}`** · period `{hold.period}`")
        st.caption(f"{hold.created_at} · proposed by run `{hold.run_id or 'unknown'}`")
        # st.write()/st.markdown() render $...$ as LaTeX - the agent's
        # rationale is plain prose (and may contain literal dollar
        # amounts), so use st.text() to avoid mangling it.
        st.text(hold.rationale)
        col1, col2 = st.columns(2)
        if col1.button("Approve", key=f"approve-{hold.id}"):
            def approve(h=hold):
                decided = decide_hold(h.id, EXECUTED, REVIEWER)
                _append_audit_log(decided, decided.status, decided.decided_at, decided.decided_by)
            _apply(approve)
        if col2.button("Reject", key=f"reject-{hold.id}"):
            _apply(lambda h=hold: decide_hold(h.id, "rejected", REVIEWER))

executed = list_holds(status=EXECUTED)
st.subheader(f"Active holds ({len(executed)})")
if not executed:
    st.caption("No payouts are currently held.")
for hold in executed:
    with st.container(border=True):
        st.markdown(f"**Artist `{hold.artist_id}`** · period `{hold.period}` — held")
        st.caption(f"approved {hold.decided_at} by {hold.decided_by}")
        reason = st.text_input("Release reason", key=f"reason-{hold.id}")
        if st.button("Release hold", key=f"release-{hold.id}", disabled=not reason.strip()):
            def release(h=hold, r=reason):
                released = release_hold(h.id, REVIEWER, r)
                _append_audit_log(released, released.status, released.released_at, released.released_by)
            _apply(release)

st.subheader("Recent decisions")
decided = [h for h in list_holds() if h.status not in (PENDING, EXECUTED)][:10]
if not decided:
    st.caption("No decisions yet.")
for hold in decided:
    with st.expander(f"Artist {hold.artist_id} · {hold.period} — {hold.status}"):
        for event in list_events(hold.id):
            st.text(
                f"{event['at']}  {event['from_status'] or '-'} -> {event['to_status']}"
                f"  by {event['actor']}" + (f"  ({event['reason']})" if event["reason"] else "")
            )
