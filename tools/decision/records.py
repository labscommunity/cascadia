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


# ---------------------------------------------------------------------------------------------
# Wide JEV choices (Task 38): choice questions past JEV's 16 trained slots, kept apart from the 50
# parity records above (whose saved spike references cover only those). Four requests of 17, 24,
# 40 + 100 and 256 options, each asked once per multi-pass strategy: 12 records, 17 questions.
# Two native questions ride along in the `single` requests only (asked once each, so their flips
# do not count three times), so those requests mix native and multi-pass lines in their dump.

WIDE_STRATEGIES = ("single", "tournament", "permute")

PRODUCT = {"type": "choice", "instructions": "Which product is this ticket about?", "criteria": {
    "checking": "Everyday checking account.", "savings": "Savings and deposit accounts.",
    "debit_card": "Debit cards and PINs.", "credit_card": "Credit cards, limits and statements.",
    "mortgage": "Home loans and remortgages.", "personal_loan": "Unsecured personal loans.",
    "auto_loan": "Car and vehicle finance.", "student_loan": "Student and education loans.",
    "business_account": "Accounts for companies and sole traders.", "payroll": "Payroll and salary runs.",
    "fx_transfer": "Foreign currency transfers and exchange rates.", "wire": "Domestic and international wires.",
    "mobile_app": "The mobile banking app.", "online_banking": "The web banking site.",
    "insurance": "Home, travel and life insurance.", "investments": "Brokerage and investment funds.",
    "pension": "Retirement and pension plans."}}

AIRPORTS = ["KTM", "DEL", "BOM", "DXB", "DOH", "IST", "LHR", "CDG", "FRA", "AMS", "ZRH", "VIE",
            "JFK", "ORD", "SFO", "LAX", "YYZ", "GRU", "NRT", "HND", "ICN", "SIN", "BKK", "SYD"]
CITIES = ["Kathmandu", "Delhi", "Mumbai", "Dubai", "Doha", "Istanbul", "London", "Paris", "Frankfurt",
          "Amsterdam", "Zurich", "Vienna", "New York", "Chicago", "San Francisco", "Los Angeles", "Toronto",
          "Sao Paulo", "Tokyo Narita", "Tokyo Haneda", "Seoul", "Singapore", "Bangkok", "Sydney"]
DESTINATION = {"type": "choice", "instructions": "Which airport is the traveller flying to?",
               "criteria": {code: f"{city} airport." for code, city in zip(AIRPORTS, CITIES)}}

COUNTRIES = ["Nepal", "India", "Bangladesh", "Bhutan", "Sri Lanka", "Pakistan", "China", "Japan",
             "South Korea", "Vietnam", "Thailand", "Malaysia", "Singapore", "Indonesia", "Philippines",
             "Australia", "New Zealand", "United Arab Emirates", "Qatar", "Saudi Arabia", "Turkey", "Egypt",
             "Kenya", "Nigeria", "South Africa", "Morocco", "United Kingdom", "Ireland", "France", "Germany",
             "Netherlands", "Spain", "Portugal", "Italy", "Poland", "Sweden", "Canada", "United States",
             "Mexico", "Brazil"]
# null instructions: the prompt shows the question id and the permute seed is the id
COUNTRY = {"type": "choice", "instructions": None, "criteria": {c.lower().replace(" ", "_"): c for c in COUNTRIES}}

LANGUAGES = ["Nepali", "Hindi", "Bengali", "Urdu", "Punjabi", "Marathi", "Gujarati", "Tamil", "Telugu",
             "Kannada", "Malayalam", "Odia", "Assamese", "Sinhala", "Maithili", "Bhojpuri", "Newari", "Tibetan",
             "Dzongkha", "Burmese", "Thai", "Lao", "Khmer", "Vietnamese", "Malay", "Indonesian", "Javanese",
             "Sundanese", "Tagalog", "Cebuano", "Mandarin", "Cantonese", "Hakka", "Japanese", "Korean",
             "Mongolian", "Kazakh", "Uzbek", "Kyrgyz", "Tajik", "Turkmen", "Pashto", "Dari", "Persian",
             "Kurdish", "Arabic", "Hebrew", "Turkish", "Azerbaijani", "Armenian", "Georgian", "Russian",
             "Ukrainian", "Belarusian", "Polish", "Czech", "Slovak", "Slovenian", "Croatian", "Serbian",
             "Bosnian", "Macedonian", "Bulgarian", "Romanian", "Hungarian", "Greek", "Albanian", "Italian",
             "Spanish", "Catalan", "Basque", "Galician", "Portuguese", "French", "Breton", "German", "Dutch",
             "Frisian", "Danish", "Swedish", "Norwegian", "Icelandic", "Finnish", "Estonian", "Latvian",
             "Lithuanian", "Irish", "Welsh", "Scottish Gaelic", "English", "Swahili", "Amharic", "Somali",
             "Yoruba", "Igbo", "Hausa", "Zulu", "Xhosa", "Afrikaans", "Malagasy"]
LANGUAGE = {"type": "choice", "instructions": "Which language should the reply be written in?",
            "criteria": {lang.lower().replace(" ", "_"): "" for lang in LANGUAGES}}

TEAMS = ["billing", "cards", "fraud", "accounts", "loans", "mortgages", "payments", "fx", "app", "web",
         "security", "compliance", "complaints", "business", "wealth", "insurance"]
REGIONS = ["nepal", "india", "gulf", "uk", "ireland", "france", "germany", "benelux", "nordics", "iberia",
           "italy", "us_east", "us_west", "canada", "latam", "apac"]
QUEUE = {"type": "choice", "instructions": "Which support queue should receive this ticket?",
         "criteria": {f"{t}_{r}": "" for t in TEAMS for r in REGIONS}}

# (id, state, questions, native questions asked in the `single` request only)
WIDE_CASES = [
    ("w0", "My credit card statement shows an annual fee I was told would be waived.",
     {"product": PRODUCT}, {"team": ROUTING}),
    ("w1", {"traveller": "Asha Gurung", "booking": "PNR 7XK2QD", "message": "Please rebook me onto tomorrow's "
            "first flight; my connection through Doha was cancelled.", "loyalty_tier": "gold"},
     {"destination": DESTINATION}, {"urgent": URGENT}),
    ("w2", "Bonjour, je n'arrive pas a me connecter a l'application depuis Lyon depuis hier soir.",
     {"country": COUNTRY, "language": LANGUAGE}, {}),
    ("w3", "Business customer in Toronto: two payroll transfers to our staff in Mexico were rejected today.",
     {"queue": QUEUE}, {}),
]


def _wide_records():
    return [{"id": f"{cid}-{s}", "state": state, "strategy": s,
             "questions": {**native, **qs} if s == "single" else dict(qs)}
            for cid, state, qs, native in WIDE_CASES for s in WIDE_STRATEGIES]


WIDE_RECORDS = _wide_records()
