<!-- Numbers below are from a SIMULATED run. Replace them with a real run before publishing. -->
# I built a system that cut LLM API costs by 47.58% while keeping 92.31% quality parity

**The problem.** Most teams send every request to their best model. Most requests don't need it:
reformatting a date or pulling an invoice number out of an email does not need the same model as
reasoning through an outage. That gap is pure spend.

**What I built.** A routing layer in front of OpenAI, Anthropic and a local Llama (Ollama). Callers
hit one OpenAI-compatible endpoint and never choose a model. For each request:

1. A complexity classifier (scikit-learn: hand-built features + TF-IDF, 0.903 5-fold
   CV accuracy on 216 labeled prompts, including routing-failure feedback) assigns tier 1, 2 or 3.
2. A YAML routing map sends the tier to the cheapest model allowed to handle it. Low-confidence
   predictions are bumped up a tier, because under-routing is the expensive mistake.
3. A sample of answers is re-run on the top model in a background worker and scored with a check
   that fits the task: field recall for extraction, label match for classification, LLM-as-judge for
   summaries and open-ended work.
4. A failed check is logged as a routing failure, the better answer is stored (or returned inline
   when the caller's latency budget allows), and the prompt becomes a training example. The
   classifier retrains weekly on those failures, so it gets better where it was wrong.

**The number.** Over 1000 requests the router spent $1.00 against a
$1.91 all-baseline bill: a 47.58% net reduction *after* paying for verification
(gross 56.75%). 92.31% of verified answers matched the reference model's quality bar;
the rest were escalated.

**What I'd tell a VP of Engineering.** The savings come from the long tail of easy requests, and they
only hold if you measure quality continuously. Verification is the cost of trusting the router, so
it's sampled, and its cost is subtracted from the headline number rather than hidden.
