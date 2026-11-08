"""Writes data/labeled_prompts.jsonl - the hand-labeled complexity dataset.

Tier 1 (simple): reformatting, extraction, basic Q&A from provided context.
Tier 2 (moderate): summarization, classification, structured analysis.
Tier 3 (complex): multi-step reasoning, creative generation, nuanced judgment.

Every row ships with verified=false. Drafted to cover the edge cases (short
hard prompts, long easy prompts, context-heavy moderate prompts); a person
should check each label before treating it as ground truth.
"""
import json
from pathlib import Path

EMAIL = ("Email: Hi team, our invoice #INV-20931 for $4,812.50 was charged twice on "
         "October 3. Please refund the duplicate to the Visa ending 4417. You can reach me "
         "at dana.ruiz@northwind.io or 713-555-0182. Thanks, Dana Ruiz, Accounts Payable, Northwind Ltd.")
ARTICLE = ("Article: The city council voted 7-2 on Tuesday to approve a $38 million bond for "
           "repairing the Harbor Street bridge, which was closed in March after inspectors found "
           "cracked welds. Supporters said the closure has cut foot traffic to east-bank shops by "
           "nearly a third. Opponents argued the bond adds to an already high debt load and that the "
           "state should cover the cost, since the welds were flagged in a 2019 audit. Construction "
           "is expected to start in January and take eight months, with a temporary ferry running "
           "in the meantime.")
TICKET = ("Ticket: After updating to version 5.2 the app crashes whenever I open the reports tab. "
          "I've reinstalled twice. This is blocking our month-end close, please help ASAP.")
LOG = ("Log:\n2026-10-02 10:14:03 ERROR db pool exhausted (max=20)\n2026-10-02 10:14:03 WARN "
       "request /api/orders took 9123ms\n2026-10-02 10:14:05 ERROR db pool exhausted (max=20)\n"
       "2026-10-02 10:14:09 INFO autoscaler added 2 instances\n2026-10-02 10:15:30 ERROR db pool "
       "exhausted (max=20)")
MEETING = ("Transcript: Priya: we're two weeks behind on the migration. Tom: mostly because the "
           "vendor API changed auth. Priya: can we ship the read path first? Tom: yes if QA signs off "
           "by Friday. Lena: QA can do Thursday if we get the build Wednesday. Priya: ok, Tom owns the "
           "build, Lena owns QA, I'll tell the client the write path slips to the 30th.")
WELL = ("Data: Well A-12 | spud 2024-03-11 | TD 3,412 m | formation Mishrif | initial rate 1,850 bopd | "
        "water cut 4% | Well A-14 | spud 2024-05-02 | TD 3,390 m | formation Mishrif | initial rate "
        "1,210 bopd | water cut 11%")
CODE = ("```python\ndef avg(xs):\n    total = 0\n    for x in xs:\n        total += x\n    return total / len(xs)\n```")
REVIEW = ("Review: Battery life is excellent and the screen is gorgeous, but the keyboard started "
          "double-typing after three weeks and support took ten days to answer.")

TIER1 = [
    f"Extract the invoice number, amount and customer email from the text below as JSON.\n\n{EMAIL}",
    f"What is the phone number in this email?\n\n{EMAIL}",
    f"Who sent this email?\n\n{EMAIL}",
    f"Pull out every dollar amount mentioned.\n\n{ARTICLE}",
    f"How long is construction expected to take, according to the article?\n\n{ARTICLE}",
    f"What was the council vote count?\n\n{ARTICLE}",
    f"List the well names in this data.\n\n{WELL}",
    f"Convert this data into a CSV with columns well, spud_date, td_m, initial_rate_bopd.\n\n{WELL}",
    f"What formation are these wells in?\n\n{WELL}",
    f"Which version did the user update to?\n\n{TICKET}",
    f"Extract all timestamps from this log.\n\n{LOG}",
    f"Count how many ERROR lines are in this log.\n\n{LOG}",
    f"Who owns QA according to the transcript?\n\n{MEETING}",
    f"List the names of everyone who spoke.\n\n{MEETING}",
    f"Rename the function in this code from avg to mean and return the code.\n\n{CODE}",
    "Convert 72 degrees Fahrenheit to Celsius.",
    "Reformat this date to ISO 8601: March 5th, 2026.",
    "Uppercase this sentence: the quick brown fox jumps over the lazy dog.",
    "Fix the spelling: 'Teh reciept was recieved on wendsday.'",
    "Sort these words alphabetically: pear, apple, mango, banana, cherry.",
    "Translate 'good morning' into Spanish.",
    "What is the capital of Canada?",
    "Convert this list to a JSON array: red, green, blue.",
    "Rewrite this in the past tense: 'She walks to the office and opens the door.'",
    "Format this phone number as (XXX) XXX-XXXX: 7135550182",
    "Remove the duplicates from this list: 3, 7, 3, 9, 7, 1.",
    "What is 15% of 240?",
    "Turn this into title case: annual report on regional energy demand",
    "Extract the domain from this URL: https://www.terra-example.com/products/vpe?ref=home",
    "Convert these key=value pairs to YAML: name=router, port=8000, debug=false",
    "Who wrote Pride and Prejudice?",
    "Lowercase every word in: THE MEETING IS MOVED TO FRIDAY",
    "Convert 3.2 kilometers to miles.",
    "Change the delimiter in this line from semicolons to commas: a;b;c;d",
    "Extract the hashtags from: Loving the #EAGE2026 talks on #FWI and #CCS today!",
    "Give me the file extension of report_final_v3.xlsx",
    f"Extract the reviewer's complaints as a bullet list.\n\n{REVIEW}",
    "Strip the HTML tags from: <p>Hello <b>world</b></p>",
    "Convert this markdown table row to CSV: | Alice | 34 | Houston |",
    "Return only the numbers from: 'Order 66 shipped 3 boxes weighing 12.5 kg.'",
    "What day of the week was January 1, 2026?",
    "Put these numbers in descending order: 14, 2, 88, 37, 5.",
    "Spell out the number 1,204 in words.",
    "Replace every instance of 'colour' with 'color' in: The colour of the colour chart.",
    "Wrap each item in double quotes: alpha, beta, gamma",
    f"What is the water cut of Well A-14?\n\n{WELL}",
    f"Extract the requested refund action and the card's last four digits.\n\n{EMAIL}",
    f"When did the bridge close?\n\n{ARTICLE}",
    f"Copy out the line that mentions the autoscaler.\n\n{LOG}",
    f"What date does the write path slip to?\n\n{MEETING}",
    "Convert this JSON to a Python dict literal: {\"a\": 1, \"b\": true}",
    "Give the plural of 'analysis'.",
    "Trim the whitespace and return: '   padded value   '",
    "What is the chemical symbol for sodium?",
    "Abbreviate these US states: Texas, California, New York.",
    "Convert 0.375 to a fraction.",
    "Extract the year from: 'Founded in 1998 in Calgary.'",
    "Join these lines into one sentence separated by commas: apples\noranges\npears",
    "Change this sentence to passive voice: 'The team fixed the bug.'",
    "Convert 5 PM Central Time to UTC during standard time.",
    "Number these lines: buy milk / call Sam / send invoice",
    "Format 1234567.891 with thousands separators and two decimals.",
    "Find the email address in: 'Questions? Write to support@example.org any time.'",
    "Convert this CSV header to snake_case: First Name,Last Name,Date Of Birth",
    "What is the square root of 144?",
    f"Extract the formation, TD and initial rate for Well A-12 as JSON.\n\n{WELL}",
    "Rewrite 'u r gr8 thx' in standard English.",
    "Convert the Roman numeral XLII to a number.",
    "Remove all punctuation from: Hello, world! How's it going?",
    "Give the ISO country code for Libya.",
    "Make this a bulleted list: eggs, flour, sugar, butter",
    "Convert 250 grams to ounces.",
]

TIER2 = [
    f"Summarize this article in three sentences.\n\n{ARTICLE}",
    f"Summarize the meeting transcript into decisions and action items.\n\n{MEETING}",
    f"Classify this support ticket as one of: billing, technical, account, other.\n\n{TICKET}",
    f"Classify the sentiment of this review as positive, negative or mixed and give a one-line reason.\n\n{REVIEW}",
    f"Classify the urgency of this ticket as low, medium or high.\n\n{TICKET}",
    f"Summarize what is going wrong in this log for an on-call engineer.\n\n{LOG}",
    f"Compare the two wells in a short table: rate, water cut, depth, and note which looks better.\n\n{WELL}",
    f"Categorize this email: refund request, complaint, inquiry, or spam.\n\n{EMAIL}",
    f"Explain what this function does and point out any edge case it misses.\n\n{CODE}",
    f"Give the key points of the article as 4 bullets, each under 15 words.\n\n{ARTICLE}",
    f"Identify the arguments for and against the bond in this article, as two lists.\n\n{ARTICLE}",
    f"Draft a short, polite reply to this email confirming we'll process the refund.\n\n{EMAIL}",
    f"Tag this review with up to three topics from: battery, display, keyboard, support, price, shipping.\n\n{REVIEW}",
    "Summarize the main differences between TCP and UDP in a short table.",
    "Explain what a p-value is to a product manager in one paragraph.",
    "Classify each of these as fruit or vegetable: tomato, carrot, apple, spinach, cucumber.",
    "Outline the main sections a post-incident report should contain.",
    "Explain the difference between supervised and unsupervised learning with one example each.",
    "Compare Python lists and tuples: mutability, performance, typical use.",
    "Summarize the plot of Romeo and Juliet in five sentences.",
    "Group these tasks by priority (high/medium/low): fix prod outage, update README, renew SSL cert next month, refactor tests.",
    "Explain how a hash map works to a junior developer.",
    "Describe the water cycle in four steps.",
    "Classify these job titles by department: Geophysicist, Account Executive, DevOps Engineer, Controller, Recruiter.",
    "Rank these programming languages by typical startup speed and explain briefly: Go, Java, Python, Rust.",
    "Summarize the pros and cons of remote work in two short lists.",
    "Explain what the Section 45Q tax credit is in plain English, in one paragraph.",
    "Review this SQL for obvious problems: SELECT * FROM orders WHERE created_at > '2026-01-01' ORDER BY id",
    "Describe what an API rate limit is and why services use one.",
    "Explain the difference between revenue and profit with a simple example.",
    "Categorize these expenses: AWS bill, team lunch, laptop, conference ticket, office rent.",
    "Summarize the key ideas of agile development in five bullets.",
    "Compare SQLite and PostgreSQL for a small internal tool.",
    "Explain what seismic inversion is to a non-geophysicist in a paragraph.",
    "Identify the parts of speech in: 'The quick fox jumped gracefully.'",
    "Draft a two-sentence LinkedIn update announcing a new hire on the data team.",
    "Classify these log levels by severity order and explain each in one line: WARN, DEBUG, ERROR, INFO, FATAL.",
    "Explain why the sky is blue in a short paragraph for a teenager.",
    "Summarize the benefits of containerizing an application with Docker.",
    "Organize these notes into headings: budget approved; hire 2 engineers; launch moved to May; budget is $200k; QA lead leaving.",
    "Explain what a confusion matrix shows, with a 2x2 example.",
    "Compare logistic regression and random forest for a small tabular dataset.",
    "Describe the difference between latency and throughput.",
    "Classify these emails as internal or external based on domain: a@ourco.com, b@gmail.com, c@ourco.com, d@vendor.io.",
    "Summarize the GDPR's main principles in six bullets.",
    "Explain what an index does in a database and when not to add one.",
    "Draft a polite meeting decline that suggests next Tuesday instead.",
    "Outline a one-day agenda for onboarding a new data analyst.",
    "Identify which of these are renewable energy sources and explain why: coal, solar, geothermal, natural gas, wind.",
    "Explain what JSON Web Tokens are used for, briefly.",
    "Summarize what a balance sheet shows.",
    "Explain the difference between git merge and git rebase.",
    "Compare three ways to cache API responses: in-memory, Redis, CDN.",
    "Classify each customer comment as praise, complaint or suggestion: 'love it', 'too slow', 'add dark mode', 'crashes daily'.",
    "Describe the stages of a typical sales funnel.",
    "Explain porosity versus permeability with an everyday analogy.",
    "Summarize the main causes of World War I in a short paragraph.",
    "Review this commit message and suggest a clearer one: 'fixed stuff'",
    "Explain what a vector database is and a typical use case.",
    "Outline the steps to set up a Python virtual environment and explain each.",
    "Turn these meeting notes into a structured status update with sections Done, Next, Blocked: shipped login; next is billing; blocked on API keys from vendor.",
    "Classify these sentences as fact or opinion: 'Water boils at 100C at sea level', 'Blue is the best color', 'Houston is in Texas'.",
    "Explain what overfitting is and two ways to reduce it.",
    "Summarize the differences between EB-2 and EB-1 visa categories at a high level.",
    "Describe how HTTPS protects data in transit.",
    "Compare Streamlit and Grafana for an internal metrics dashboard.",
    "Explain inflation to a 12-year-old in one paragraph.",
    "Identify the main stakeholders in a hospital software rollout and what each cares about.",
    "Extract the action items and owners from this transcript and format them as a table with a due-date column.\n\n" + MEETING,
    "Summarize the advantages of LoRA fine-tuning over full fine-tuning.",
    "Explain the CAP theorem with a short example.",
]

TIER3 = [
    "Design a routing strategy for an LLM gateway that minimizes cost while keeping quality parity, and justify each tradeoff.",
    "A well's production dropped 40% in a month while water cut rose from 5% to 30%. Reason step by step through the likely causes and how you'd diagnose each.",
    "Write a short story about a lighthouse keeper who discovers the light has been signaling to someone.",
    "Our startup has 9 months of runway and two product lines. Analyze whether to cut one, and recommend a plan with risks.",
    "Prove that the square root of 2 is irrational.",
    "Critique this business strategy: give away the software free and charge for training. What are the second-order effects?",
    "Write a persuasive speech for a city council meeting arguing for a temporary ferry while the bridge is repaired.",
    "Evaluate whether a mid-size operator should build or buy a subsurface ML platform. Consider cost, talent, IP and time to value.",
    "Design the database schema and API for a multi-tenant feature-flag service, and explain how you'd handle gradual rollouts.",
    "A train leaves at 3pm going 80 km/h; another leaves the same station at 4pm going 100 km/h on the same track. When and where does the second catch up? Show your reasoning.",
    "Write a poem about the Sahara at night in the style of a classical Arabic qasida, in English.",
    "Debug this: our API p99 latency jumped from 300ms to 4s after a deploy that only changed logging. Walk through hypotheses in order of likelihood and how to test each.",
    "Assess the ethical implications of using AI to screen job applicants, and propose safeguards a mid-size company could actually implement.",
    "Forecast the impact on our cloud bill if traffic triples and we move from on-demand to reserved instances. State your assumptions and give a range.",
    f"Analyze this log and propose a root cause, a short-term mitigation and a long-term fix, with the reasoning for each.\n\n{LOG}",
    f"Given this transcript, assess whether the project is at risk, what the client should be told, and draft the message.\n\n{MEETING}",
    f"Based on these two wells, reason about what might explain the difference in water cut and what data you'd collect next to decide where to drill A-16.\n\n{WELL}",
    f"Evaluate the arguments in this article: which side has the stronger case, what's missing, and what would change your mind?\n\n{ARTICLE}",
    "Explain why the halting problem is undecidable and what that implies for static analysis tools.",
    "Design an evaluation plan for a RAG system over legal documents, including metrics, datasets and failure modes.",
    "Negotiate on my behalf: draft a counter-offer email for a job offer that's 15% below market, keeping the relationship warm.",
    "What would happen to global shipping if the Suez Canal closed for six months? Reason through first and second-order effects.",
    "Write a screenplay scene where two rival geologists must share a tent during a sandstorm.",
    "Compare three architectures for real-time fraud detection, recommend one for a bank with 10M transactions/day, and justify it.",
    "Optimize this function for large inputs and explain the complexity before and after, then discuss numerical stability.\n\n" + CODE,
    "Invent a board game that teaches carbon capture concepts to high-school students. Describe rules, components and a sample turn.",
    "Plan a 12-week curriculum to take a petroleum geologist to production ML engineering, with weekly milestones and projects.",
    "Is it better for a small team to adopt microservices or a modular monolith? Weigh the tradeoffs for a 6-engineer team and recommend.",
    "Derive the formula for the sum of the first n squares and verify it for n=4.",
    "Analyze the strategic risks of depending on a single LLM provider and propose a mitigation strategy with costs.",
    "Write a eulogy for a beloved retired professor who mentored hundreds of students.",
    "Our model's accuracy is 94% but customers say it's useless. Diagnose what might be going on and propose how to find out.",
    "Recommend a pricing strategy for a B2B SaaS that currently charges per seat but whose value scales with data volume. Justify it.",
    "A patient-facing chatbot gave a wrong dosage answer. Walk through the incident review: root causes, accountability, and process changes.",
    "Reason about whether a 2x faster but 5% less accurate model is the right choice for a customer-support triage system.",
    "Design a carbon MMV (monitoring, measurement, verification) plan for a saline aquifer CO2 injection site, with justification for each measurement.",
    "Write a haiku sequence (five haiku) that tells the story of an oil field from discovery to abandonment.",
    "Critique the claim that 'AI will replace geophysicists within ten years'. Steelman it, then evaluate it.",
    "If interest rates rise 2 points, how should a startup with venture debt adjust its plan? Think through cash, hiring and fundraising.",
    "Architect a system that verifies an ML model's subsurface predictions against physical constraints, and explain where it can fail.",
    "Solve: three friends split a bill unevenly so that A pays twice B and C pays $10 more than B; the bill is $130. How much does each pay? Explain the steps.",
    "Draft a strategy memo on whether to enter the Libyan upstream services market in 2027, covering political risk, partners and timing.",
    "Write a persuasive op-ed arguing that carbon capture deserves more public funding, then list the strongest counterarguments.",
    "Design an A/B test to measure whether a new onboarding flow increases 30-day retention, including sample size reasoning and pitfalls.",
    "Explain the tradeoffs between consistency models in distributed databases and recommend one for a payments ledger.",
    "Imagine a city in 2060 designed around water scarcity. Describe how daily life works there in a vivid short piece.",
    "Evaluate this argument: 'Our churn rose after the price increase, so the price increase caused churn.' What else could explain it and how would you test?",
    "Plan the migration of a 2TB PostgreSQL database to a new cloud region with under 5 minutes of downtime. Step by step, with rollback.",
    "Assess the tradeoffs of fine-tuning versus retrieval for a domain-specific assistant with frequently changing documents.",
    "Write a children's story that explains how earthquakes happen, with a memorable character.",
    "Reason through whether this is a good hire: strong technical interview, weak references, and a 6-month gap. What would you ask next?",
    "Analyze how a 20% drop in oil prices would ripple through a national budget that is 90% oil-funded, and suggest policy responses.",
    "Design a self-healing documentation system that detects when code changes make docs stale. Explain the detection logic and its limits.",
    "Why do large language models hallucinate? Explain the mechanisms and evaluate three mitigation strategies.",
    "Write a dialogue between Socrates and a modern data scientist about whether a model can 'know' anything.",
    "Propose a strategy to reduce our LLM API spend by 50% without hurting quality, with a rollout plan and how we'd measure success.",
    "Given conflicting reports from two teams about the cause of an outage, how should a VP decide what to tell the board? Reason through it.",
    "Develop a risk-ranked plan for a first exploration well in a frontier basin with sparse 2D seismic.",
    "Explain Gödel's incompleteness theorems informally and why they matter for mathematics.",
    "Create a fictional company's founding story and mission statement for an AI geoscience startup, with three core values and why each matters.",
    "Compare the long-term economics of solar plus storage versus gas peakers for a utility in Texas, stating assumptions.",
    "Our classifier routes 30% of prompts wrong. Propose a systematic process to find out why and fix it, without hand-labeling everything.",
    "Write a cover letter for a senior reservoir geophysicist moving into AI product leadership, emphasizing a non-obvious through-line.",
    "Think step by step: if every request to a model costs $0.01 and verification samples 25% of requests at $0.03 each, at what cheap-model price does routing stop saving money versus always using the $0.01 model?",
    "Evaluate whether a hospital should deploy an AI triage tool that is more accurate on average but worse for one demographic group.",
    "Design a curriculum module that teaches full-waveform inversion intuitively before the math, and justify the sequence.",
    "Analyze the failure modes of LLM-as-judge evaluation and design a protocol that controls for each.",
    "Write a satirical news article about a meeting that could have been an email.",
    "Recommend how to structure equity for three co-founders with unequal time commitment and capital contributions. Explain the reasoning.",
    "Reason about whether this conclusion follows: all reservoirs in the basin are carbonates; some carbonates are fractured; therefore some reservoirs in the basin are fractured.",
    "Plan a go-to-market for a physics-verified AI product aimed at national oil companies, including sales cycle realities.",
]


def main():
    out = Path(__file__).resolve().parents[1] / "data" / "labeled_prompts.jsonl"
    out.parent.mkdir(exist_ok=True)
    rows = [(p, 1) for p in TIER1] + [(p, 2) for p in TIER2] + [(p, 3) for p in TIER3]
    seen = set()
    with out.open("w", encoding="utf-8") as f:
        for p, t in rows:
            assert p not in seen, f"duplicate prompt: {p[:60]}"
            seen.add(p)
            f.write(json.dumps({"prompt": p, "tier": t, "verified": False}) + "\n")
    print(f"wrote {len(rows)} prompts ({len(TIER1)}/{len(TIER2)}/{len(TIER3)}) -> {out}")


if __name__ == "__main__":
    main()
