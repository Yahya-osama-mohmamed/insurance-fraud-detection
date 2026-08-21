---
title: Insurance Claims Fraud Detection
emoji: 🚨
colorFrom: indigo
colorTo: red
sdk: docker
app_port: 7860
pinned: false
license: mit
---

# Insurance Claims Fraud Detection

A fraud-scoring API with a built-in analyst UI. Submit a vehicle insurance claim
and get back a probability, a flag, and the reasons behind it in plain language.

**Open [`/ui/`](./ui/) for the analyst interface**, or `POST /predict` for the API.

At a 5.99% fraud rate, accuracy is a trap — a model that flags nothing scores
94% and catches no fraud. This one is selected on PR-AUC and tuned on F2, and at
its operating threshold of 0.036 it catches about 89% of fraud, with roughly 1
in 6 flagged claims being genuine against a 1-in-17 base rate.

Source: https://github.com/Yahya-osama-mohmamed/insurance-fraud-detection
