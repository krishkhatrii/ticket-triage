import json

from app.db import SessionLocal                                  
from app.llm import classify_ticket, embed                       
from app.models import Classification as ClassificationRow      
from app.retrieval import find_similar                          

LABELS_PATH = "data/eval_dataset_labels.json"


def main():
    with open(LABELS_PATH) as f:
        cases = json.load(f)

    db = SessionLocal()                                        
    # ticket_id -> category label, for every stored ticket       
    label_of = {c.ticket_id: c.category for c in db.query(ClassificationRow)}

    category_hits = 0
    urgency_hits = 0
    retrieval_hits = 0                                          

    for i, case in enumerate(cases, start=1):
        pred = classify_ticket(case["subject"], case["body"])

        vector = embed(case["subject"] + "\n" + case["body"])    
        rows = find_similar(db, vector, k=3)                     
        retrieved = [label_of[r.Ticket.id] for r in rows]        
        hit = case["category"] in retrieved                      

        cat_ok = pred.category.value == case["category"]
        urg_ok = pred.urgency.value == case["urgency"]
        category_hits += cat_ok
        urgency_hits += urg_ok
        retrieval_hits += hit                                    

        print(
            f"[{i:2}/{len(cases)}] cat {'✓' if cat_ok else '✗'} urg {'✓' if urg_ok else '✗'} "
            f"hit@3 {'✓' if hit else '✗'} "                                     
            f"| true={case['category']} / {case['urgency']} "
            f"| pred={pred.category.value} / {pred.urgency.value}"
        )

    db.close()                                                   

    n = len(cases)
    print(f"\nCategory accuracy: {category_hits}/{n} = {category_hits / n:.0%}")
    print(f"Urgency accuracy:  {urgency_hits}/{n} = {urgency_hits / n:.0%}")
    print(f"Retrieval hit@3:   {retrieval_hits}/{n} = {retrieval_hits / n:.0%}")   


if __name__ == "__main__":
    main()
