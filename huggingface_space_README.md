---
title: Flight Price Prediction
emoji: ✈️
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 6.26.0
app_file: app_gradio.py
pinned: false
---

# Flight Price Prediction

Fare estimator trained on 271,888 historical bookings (XGBoost (tuned),
R² 1.0000).

**To publish:** create a new Space at huggingface.co/new-space (SDK = Gradio), then upload
`app_gradio.py`, `requirements.txt`, this file renamed to `README.md`, and the whole `model/` folder.
