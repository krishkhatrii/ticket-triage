import json

from app.db import SessionLocal
from app.llm import classify_ticket, embed
from app.models import Classification as ClassificationRow
from app.retrieval import find_similar

HAND_PATH = "data/eval_hand_labels.json"
DATASET_PATH = "data/eval_dataset_labels.json"


def main():
    with open(HAND_PATH) as f:
        hand = json.load(f)
    with open(DATASET_PATH) as f:
        dataset = json.load(f)

    db = SessionLocal()
    # ticket_id -> category label, for every stored ticket
    label_of = {c.ticket_id: c.category for c in db.query(ClassificationRow)}

    hand_cat_hits = 0
    hand_urg_hits = 0
    ds_cat_hits = 0
    ds_urg_hits = 0
    retrieval_hits = 0

    for i, (h, d) in enumerate(zip(hand, dataset, strict=True), start=1):
        assert h["subject"] == d["subject"], f"ticket {i}: files are out of order"
        pred = classify_ticket(h["subject"], h["body"])

        vector = embed(h["subject"] + "\n" + h["body"])
        rows = find_similar(db, vector, k=3)
        retrieved = [label_of[r.Ticket.id] for r in rows]
        hit = d["category"] in retrieved

        hand_cat_ok = pred.category.value == h["category"]
        hand_urg_ok = pred.urgency.value == h["urgency"]
        ds_cat_ok = pred.category.value == d["category"]
        ds_urg_ok = pred.urgency.value == d["urgency"]

        hand_cat_hits += hand_cat_ok
        hand_urg_hits += hand_urg_ok
        ds_cat_hits += ds_cat_ok
        ds_urg_hits += ds_urg_ok
        retrieval_hits += hit

        print(
            f"[{i:2}/{len(hand)}] hand cat {'✓' if hand_cat_ok else '✗'} urg {'✓' if hand_urg_ok else '✗'} "
            f"| ds cat {'✓' if ds_cat_ok else '✗'} urg {'✓' if ds_urg_ok else '✗'} "
            f"| hit@3 {'✓' if hit else '✗'} "
            f"| hand={h['category']} / {h['urgency']} "
            f"| pred={pred.category.value} / {pred.urgency.value}"
        )

    db.close()

    n = len(hand)
    print(f"\n{'':10}{'hand':>8}{'dataset':>10}")
    print(f"{'Category':10}{hand_cat_hits / n:>8.0%}{ds_cat_hits / n:>10.0%}")
    print(f"{'Urgency':10}{hand_urg_hits / n:>8.0%}{ds_urg_hits / n:>10.0%}")
    print(f"Retrieval hit@3 (dataset labels): {retrieval_hits}/{n} = {retrieval_hits / n:.0%}")


if __name__ == "__main__":
    main()
