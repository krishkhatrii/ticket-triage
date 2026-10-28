import json, random
from collections import defaultdict
from datasets import load_dataset
from app.db import SessionLocal, init_db
from app.models import Ticket, Classification, Embedding
from app.llm import embed

def clean(text: str) -> str:
    return text.replace("\\n", "\n").strip()

def main():
    init_db()
    ds = load_dataset("Tobi-Bueck/customer-support-tickets", split="train")
    ds = ds.filter(lambda r: r["language"] == "en" and r["subject"] and r["body"]
               and r["queue"] and r["priority"])

    by_queue = defaultdict(list)
    for row in ds:
        by_queue[row["queue"]].append(row)

    rng = random.Random(42)
    seed_rows, eval_rows = [], []
    for queue, rows in by_queue.items():
        rng.shuffle(rows)
        seed_rows += rows[:15]
        eval_rows += rows[15:17]

    db = SessionLocal()
    if db.query(Ticket).count() > 0:
        print("Database already has tickets. Wipe it first (docker compose down -v).")
        return
    for row in seed_rows:
        subject, body = clean(row["subject"]), clean(row["body"])
        ticket = Ticket(subject=subject, body=body, status="resolved")
        db.add(ticket)
        db.flush()  # assigns ticket.id without committing yet
        db.add(Classification(ticket_id=ticket.id, category=row["queue"],
                              urgency=row["priority"], confidence=1.0))
        db.add(Embedding(ticket_id=ticket.id, vector=embed(subject + "\n" + body)))
    db.commit()

    with open("data/eval_dataset_labels.json", "w") as f:
        json.dump([{"subject": clean(r["subject"]), "body": clean(r["body"]),
                    "category": r["queue"], "urgency": r["priority"]}
                   for r in eval_rows], f, indent=2)

if __name__ == "__main__":
    main()