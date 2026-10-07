"""Decision records for the spike: 50 records, varied question types and sizes.

Accuracy isn't measured; these only need to exercise the model so the
OpenVINO path can be compared against the PyTorch reference.
"""

STATES = [
    "I was charged twice for my order and want my money back.",
    "My card was declined at the supermarket even though I have funds.",
    "How do I change the PIN on my debit card?",
    "I lost my card on the train this morning, please freeze it.",
    "The app crashes every time I open the transfers screen on Android 15.",
    "Can I get a statement for my business account for last quarter?",
    "Someone I don't know made three payments from my account overnight.",
    "Why is the exchange rate on my euro payment worse than the website said?",
    "I'd like to close my savings account and move the balance to checking.",
    "Your support agent was rude and hung up on me twice today.",
    {"ticket": "Build on main fails after merging PR 412", "failing": "crate runner tests time out",
     "since": "2 hours", "blocking_release": True},
    {"customer_message": "I was charged twice for order ord_7429.", "duplicate_charge_usd": 680,
     "customer_identity_verified": True, "policy": "Refunds above USD 500 require human approval."},
    {"post": "Selling cheap followers, DM me now!!! 10k for $5", "account_age_days": 1, "links": 3},
    {"post": "Does anyone know a good pediatrician in Lalitpur?", "account_age_days": 900, "links": 0},
    {"tool_request": "rm -rf /var/lib/postgres", "requested_by": "agent", "environment": "production"},
    {"tool_request": "ls -la ./src", "requested_by": "agent", "environment": "dev sandbox"},
    "Please book a table for four at 7pm tomorrow and send me the confirmation.",
    "What's the weather going to be like in Kathmandu this weekend?",
    "Translate 'where is the train station' into Japanese.",
    "Summarize the attached quarterly report in three bullet points.",
    "The checkout page returns HTTP 500 for every customer since the 14:05 deploy.",
    "A typo on the pricing page says 'monthy' instead of 'monthly'.",
    "Login is slow, about 4 seconds, for some users in Europe.",
    "All customer data in the EU region appears to be readable without authentication.",
    "Can you recommend a laptop under $1000 for programming?",
]

ROUTING = {"type": "choice", "instructions": "Which team should handle this request?",
           "criteria": {"billing": "Charges, refunds, fees and exchange rates.",
                        "cards": "Card loss, PINs, declines and freezing.",
                        "fraud": "Unauthorized activity on the account.",
                        "accounts": "Opening, closing and statements.",
                        "technical": "App and website problems.",
                        "complaints": "Service quality complaints."}}
SEVERITY = {"type": "score", "instructions": "How severe is the impact?",
            "criteria": ["No impact", "Minor", "Major", "Critical"]}
URGENT = {"type": "noul", "instructions": "Does this need action within the hour?"}
MODERATE = {"type": "choice", "instructions": "Moderation decision for this content.",
            "criteria": {"allow": "Benign content.", "review": "Unclear; a human should look.",
                         "remove": "Spam, scams or abuse."}}
AGENT = {"type": "choice", "instructions": "Which tool should the assistant use next?",
         "criteria": {"calendar": "Create or change bookings and events.",
                      "search": "Look up live information on the web.",
                      "translate": "Translate text between languages.",
                      "summarize": "Condense a provided document.",
                      "none": "Answer directly without a tool."}}
APPROVE = {"type": "noul", "instructions": "Does this action require human confirmation?"}

QUESTION_SETS = [
    {"team": ROUTING},
    {"team": ROUTING, "severity": SEVERITY, "urgent": URGENT},
    {"severity": SEVERITY},
    {"moderation": MODERATE, "urgent": URGENT},
    {"next_tool": AGENT},
    {"approval": APPROVE, "severity": SEVERITY},
]


def records(n=50):
    out = []
    for i in range(n):
        out.append({"id": f"r{i:02d}", "state": STATES[i % len(STATES)],
                    "questions": QUESTION_SETS[(i * 7 + i // len(STATES)) % len(QUESTION_SETS)]})
    return out
