# fraud-pipeline

A real-time card fraud detection pipeline. Transactions flow through Kafka, get scored by a machine learning model and explainable rules, and suspicious ones go to an analyst review dashboard.

**What's real and what's simulated:** the pipeline and services are real, and the model was also tested on real public fraud data. The live pipeline runs on simulated transactions, because public datasets remove the card IDs needed for per-card history. No real card data or payments are involved.

## How it works

```
 producer ──> Kafka ──> consumer ──> PostgreSQL ──> FastAPI ──> React dashboard
                           │                                    analyst reviews
                     rolling features (Redis)
                     model + rules
```

| Part | What it does |
|---|---|
| `simulator.py` | Generates transactions with four fraud patterns, plus legitimate behavior that looks suspicious |
| `features.py` | Per-card rolling features, such as uses in the last hour, spend compared with average, and travel speed |
| `train.py` | Trains a gradient-boosted model (scikit-learn), splits data by time, and picks thresholds |
| `rules.py` | Named rules a person can read, and how they combine with the model score |
| `stream.py` | Kafka producer and consumer, with a dead-letter topic for bad messages |
| `api.py`, `dashboard/` | REST API and a React/TypeScript review queue |

Each decision is one of three: approve, send to review, or decline.

## Results

**Real data:** the [ULB credit card dataset](https://www.openml.org/d/1597), 284,807 transactions, tested on the latest 20%, which contains 75 frauds.

| Measure | Result |
|---|---|
| PR-AUC | 0.746 (random guessing gets 0.0013) |
| Precision | 88% |
| Recall | 71% |

**Simulated data, offline:** 301,352 transactions, tested on the latest 20%.

| Measure | Model | Logistic regression | Rules only |
|---|---|---|---|
| PR-AUC | 0.916 | 0.669 | n/a |
| Precision when flagging | 27% | n/a | 25% |
| Recall when flagging | 94% | n/a | 56% |

This score is higher than on real data because I wrote the simulator, so its fraud is easier to spot. By pattern, flagged fraud was 100% of card testing, bursts, and big purchases, and 18% of "quiet" fraud, which is built to look normal.

**Live run:** 30,000 unseen transactions through real Kafka, Redis, and PostgreSQL on one laptop.

| Measure | Result |
|---|---|
| Throughput | 353 transactions per second, on one consumer |
| Scoring time | 1.95 ms median, 6.3 ms at the 99th percentile |
| Stored | 30,000, with none lost or duplicated |
| Fraud flagged | 89% of 53 frauds, with 15% precision |
| Declines | 90% precision, 81% recall |

Live recall is a little below the offline number. The likely cause is that Redis starts empty, so cards have no history at first, but with only 53 frauds I haven't confirmed this.

## Design decisions

**Accuracy is the wrong measure.** Fraud here is under 1% of transactions, so a model that always says "not fraud" is over 99% accurate and useless. I report precision, recall, and PR-AUC.

**Split by time, not randomly.** A random split lets the model learn from the future and inflates results. Thresholds come from a middle period, and scores from a later one.

**One feature code path.** Models often fail in production because features are computed differently in training and live. Here both use `features.compute`, with only the storage differing. A test runs the same transactions through both stores and requires identical results.

**Safe redelivery.** Kafka can deliver a message twice, so the offset is committed only after the decision is stored. Card history and the decisions table both ignore repeats. A redelivered transaction also gets the same features, because each only looks at events from before it.

**Per-card ordering.** Messages are keyed by card ID, so one card's transactions stay in order.

**Rules can't decline alone.** Real travel and big purchases trip the rules, and declining a real customer has a cost. A rule only sends a transaction to review. Declining takes a high model score, tuned to 90% precision on the validation period.

## Running it

Requires Docker, Python 3.12, and Node 20.

```sh
docker compose up -d --wait
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python -m fraud.train                  # Train on simulated data
python -m fraud.benchmark              # Test the method on real data (downloads about 150 MB)
RUN_INTEGRATION=1 python -m pytest     # Tests

python -m fraud.stream produce --count 30000
python -m fraud.stream consume --exit-when-idle 10
python -m fraud.live_eval              # Compare decisions with the answers

uvicorn fraud.api:app --port 8000
cd dashboard && npm install && npm run dev
```

Then open http://localhost:5173. The database password in `docker-compose.yml` is only a local default for a throwaway container.

## Known limits

- The live model trains on simulated data, so its scores don't predict real-world performance. The real-data result is the better guide.
- The model is never retrained on analyst decisions. They're stored, but closing that loop is the next step.
- The real-data test has only 75 frauds, so small differences in those numbers don't mean much.
- History starts empty on a fresh run, and old events are only trimmed relative to the newest transaction seen for a card.
- The API has no login, because it's built for local use. It isn't deployed anywhere.
