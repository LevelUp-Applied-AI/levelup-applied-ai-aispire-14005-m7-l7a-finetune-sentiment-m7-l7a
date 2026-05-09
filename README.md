# Module 7 Week A — Lab: Fine-Tune DistilBERT for App-Review Sentiment

Fine-tune DistilBERT on a curated 7,472-review subset of the AARSynth app-review corpus (9 apps × 3 sentiment classes — negative / neutral / positive), evaluate, push the model to Hugging Face Hub, and produce an evaluation report.

Dataset:
- `data/app_reviews_train.csv` — 7,472 reviews (lab uses this; internally splits 80/20).
- `data/app_reviews_eval.csv` — 1,867 reviews (independent holdout for instructor review; not consumed by the lab pipeline by default).
- `fixtures/tiny_app_reviews.csv` — 60-row CI smoke fixture.

Full instructions: see the **Applied Lab guide** linked in TalentLMS.

## Quick start

```bash
pip install -r requirements.txt
huggingface-cli login        # one-time; needs a write-scoped token
python lab.py
```

Outputs (committed):
- `metrics.json`
- `predictions.csv`
- `evaluation-report.md` (you write this)

Local-only (gitignored):
- `model/` (~265 MB) — also pushed to your HF Hub.

## Submission

Open a PR from `lab-7a-finetune-sentiment` into `main`. Paste the PR URL into TalentLMS → Module 7 → Applied Lab 7A.

---

## License

This repository is provided for educational use only. See [LICENSE](LICENSE) for terms.

You may clone and modify this repository for personal learning and practice, and reference code you wrote here in your professional portfolio. Redistribution outside this course is not permitted.
