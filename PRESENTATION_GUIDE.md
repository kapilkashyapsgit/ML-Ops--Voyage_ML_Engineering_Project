# Flight Price Prediction — Presentation Guide

## 1. Before the presentation (do this once)

1. Open [colab.research.google.com](https://colab.research.google.com) → **File → Upload notebook** → `Flight_Price_Prediction_Complete.ipynb`
2. Left sidebar → **📁 folder icon → Upload** → `flights.csv`, `hotels.csv`, `users.csv`
3. **Runtime → Run all** (~3 minutes). Everything is already executed in the file you received, so even if the Wi-Fi dies the outputs are visible.
4. **Section 14 launches the Gradio web app automatically** and prints a public `*.gradio.live` link — open it on your phone or share it with the class. The app also appears inline inside the notebook.

## 1b. The live demo (Section 14) — how to run it in front of the class

| Tab | What to do on screen | What to say |
|---|---|---|
| **💰 Predict a fare** | Sao Paulo → Rio de Janeiro, Economy, CloudFy → **Predict fare** | "The model returns R$ 592 — and the chart on the right instantly re-prices the same flight in Premium and First Class, so you can see the class effect we found in the EDA." |
| | switch cabin class to First Class → Predict again | "Same route, same date, only the cabin changed — the fare jumps to R$ 1,093. Cabin class is the strongest feature." |
| **📊 Compare every option** | Sao Paulo → Salvador → **Compare all options** | "Here the model runs 9 predictions at once — every agency × cabin class — and ranks them. Cheapest is Rainbow Economy at R$ 744, most expensive R$ 1,427. This is what a fare-comparison site does internally." |
| **🧠 How it works** | just scroll | "This tab documents the pipeline, the test metrics, and why the accuracy is this high — no hand-waving." |

Backup plan if the Colab session or Wi-Fi dies: the executed notebook already contains every chart and result, and `flight_price_app.zip` runs the exact same app offline with `python app_gradio.py`.

## 2. Suggested slide flow (each notebook section = one slide)

| # | Section | What to say (30–60 sec) |
|---|---|---|
| 1 | Problem | Predict a flight fare from route, cabin class, agency and date. Regression problem, 271,888 bookings, 3 related tables. |
| 2 | Data | users → flights → hotels joined by `userCode` and `travelCode`. Zero missing values, zero orphan keys. |
| 3 | EDA | Price is spiky, not bell-shaped. Cabin class explains most of it. `time` and `distance` correlate 1.000 → multicollinearity. |
| 4 | **Reality check** | **The killer slide.** 490 combinations of route × class × agency, each with exactly ONE price, spread = 0.0000. The fare is `rate(class, agency) × distance`. So a high R² here is rule-discovery, not luck. |
| 5 | Feature engineering | Calendar features + customer attributes; IDs dropped on purpose. |
| 6 | Pipeline | One-hot inside a `ColumnTransformer`, fitted only on the training fold → no leakage, one artifact for deployment. |
| 7 | Model comparison | Linear Regression R² 0.92 (one straight line). Trees ≈ 1.00, because the true rule is piecewise if/else — exactly what trees do. |
| 8 | CV + tuning | 5-fold CV, RandomizedSearchCV over 8 configurations. |
| 9 | Evaluation | Residuals centred at 0, no bias. Permutation importance shows age/gender/company/date ≈ 0 → the model found the real rule. |
| 10 | **Dynamic market test** | We simulated real airline pricing (seasonality + weekend premium + 12% random shocks). XGBoost still reaches **R² ≈ 0.91** against a theoretical ceiling of 0.93, Linear Regression collapses to 0.78. This proves the *model* works, not just the dataset. |
| 11 | **Live demo** | Open the Gradio app and run the three tabs above. This is the part the class remembers. |
| 12 | Deployment | Saved pipeline → Gradio app + Flask REST API → Docker → Kubernetes (deployment + service with health probes) → MLflow tracking. `app_gradio.py` is generated from the notebook's own source with `inspect.getsource`, so the deployed app cannot drift from the demo. |
| 13 | Conclusion | Insights, limitations, future scope. |

## 3. Likely questions and answers

**Q: Your R² is 1.00 — isn't that overfitting?**
No. Section 5 proves the fare is a deterministic price list: every route × class × agency combination has exactly one price, with a spread of 0.0000. There is nothing to overfit to. The test set is untouched data and cross-validation gives the same answer across all 5 folds. Section 11 shows what happens when randomness *is* present: R² drops to ~0.91, close to the theoretical noise ceiling.

**Q: Why not LabelEncoder like most tutorials?**
LabelEncoder invents a fake ordering (Recife = 5 > Natal = 4 is meaningless) and, if fitted on the whole dataset, leaks test information into training. One-hot inside a `ColumnTransformer` is fitted on the training fold only, handles unseen categories at inference, and ships as a single artifact.

**Q: Why does Linear Regression do so badly?**
It fits one global straight line. The true rule is conditional — "if firstClass and Rainbow then 2.35 per km". Trees split on exactly those conditions; a linear model cannot express them without explicit interaction terms.

**Q: Why did you keep age, gender and company if they don't matter?**
As a control. Their near-zero permutation importance is evidence that the pipeline is honest and the model is not picking up spurious correlations.

**Q: `time` correlates 0.64 with price but has zero importance — why?**
It is redundant, not useless. `distance` carries the same information (speed is constant ~385 km/h), so shuffling `time` alone changes nothing. That is textbook multicollinearity.

**Q: Would this work for real airlines?**
Not as-is. Real fares depend on seat availability, days-to-departure, competitor prices and demand — none of which exist in this dataset. Section 11 is the closest approximation, and the future-scope slide lists the features I would add.

## 4. Numbers to memorise

- 271,888 flight bookings · 1,340 users · 40,552 hotel stays · 9 cities · 3 agencies · 3 cabin classes
- 490 unique price combinations — the entire fare table
- Best model: XGBoost (tuned) / Random Forest → R² ≈ 1.000, MAE ≈ 0
- Linear Regression baseline → R² 0.919, MAPE 10.4%
- Dynamic-market stress test → XGBoost R² 0.909 vs a theoretical ceiling of 0.926
- Flights are ~78% of total business-travel spend; hotels ~22%

## 5. Extra questions the app invites

**Q: Is the app calling the same model as the notebook?**
Yes. The app loads `model/flight_price_pipeline.joblib` — one file containing the one-hot encoder *and* the trained model — and does no preprocessing of its own. The standalone `app_gradio.py` is generated from the notebook's own function source with `inspect.getsource`, so the code cannot drift.

**Q: Why does FlyingDrops show Economy and Premium fares in the compare tab?**
In the historical data FlyingDrops sells only First Class. Those two rows are the model *extrapolating* to a product that agency does not actually offer — a nice example of a model answering a question outside its training distribution. The app labels this under the tab heading.

**Q: Can this go online permanently?**
Yes — `README_huggingface.md` in the app folder has the Hugging Face Spaces header. Create a Space with SDK = Gradio, upload `app_gradio.py`, `requirements.txt`, that file renamed to `README.md`, and the `model/` folder. The Colab `*.gradio.live` link is temporary (72 hours); a Space is permanent and free.

## 6. Deploying the app after the presentation

```bash
unzip flight_price_app.zip -d flight_price_app
cd flight_price_app
pip install -r requirements.txt
python app_gradio.py          # http://localhost:7860
```
