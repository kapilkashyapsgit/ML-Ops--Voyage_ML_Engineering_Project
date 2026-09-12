# Flight Price Prediction - Gradio web app (auto-generated from the notebook)
# Run:  pip install -r requirements.txt  &&  python app_gradio.py
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import gradio as gr
import joblib
import matplotlib.pyplot as plt
import pandas as pd

MODEL  = joblib.load(Path(__file__).parent / "model/flight_price_pipeline.joblib")
META   = json.loads((Path(__file__).parent / "model/metadata.json").read_text())
ROUTES = pd.read_csv(Path(__file__).parent / "model/routes.csv")
metadata = META

FEATURES      = META["features"]["categorical"] + META["features"]["numeric"]
CITIES        = sorted(set(ROUTES["from"]) | set(ROUTES["to"]))
AGENCIES      = META["categories"]["agency"]
ROUTE_LOOKUP  = {(r["from"], r["to"]): (r["distance"], r["time"]) for _, r in ROUTES.iterrows()}
CURRENCY      = 'R$'
CLASSES       = ['economic', 'premium', 'firstClass']
CLASS_LABEL   = {'economic': 'Economy', 'premium': 'Premium', 'firstClass': 'First Class'}
CLASS_COLOURS = {'economic': '#1baf7a', 'premium': '#2a78d6', 'firstClass': '#4a3aa7'}
PAL           = {'blue': '#2a78d6', 'orange': '#eb6834', 'aqua': '#1baf7a', 'yellow': '#eda100', 'magenta': '#e87ba4', 'green': '#008300', 'violet': '#4a3aa7', 'red': '#e34948'}
INK, MUTED    = '#0b0b0b', '#52514e'
CSS           = '\n.gradio-container {max-width: 1180px !important;}\n#fp-header {background: linear-gradient(120deg, #2a78d6 0%, #4a3aa7 100%);\n  padding: 26px 30px; border-radius: 16px; margin-bottom: 18px;}\n#fp-header h1, #fp-header p, #fp-header span {color: #ffffff !important;}\n#fp-header h1 {margin: 0 0 6px 0; font-size: 27px; font-weight: 700;}\n#fp-header p {margin: 0; font-size: 14.5px; opacity: .93;}\n#fp-header .fp-badges {margin-top: 14px; display: flex; gap: 10px; flex-wrap: wrap;}\n#fp-header .fp-badge {background: rgba(255,255,255,.17); border: 1px solid rgba(255,255,255,.35);\n  padding: 5px 12px; border-radius: 999px; font-size: 12.5px;}\n.fp-card {background: linear-gradient(135deg, #2a78d6 0%, #4a3aa7 100%);\n  border-radius: 16px; padding: 24px 26px; text-align: center;}\n.fp-card, .fp-card div, .fp-card b, .fp-card span {color: #ffffff !important;}\n.fp-card-label {text-transform: uppercase; letter-spacing: .09em; font-size: 11.5px; opacity: .85;}\n.fp-card-price {font-size: 46px; font-weight: 800; line-height: 1.15; margin: 4px 0 2px;}\n.fp-card-route {font-size: 15px; opacity: .95;}\n.fp-chips {margin-top: 12px; display: flex; gap: 8px; justify-content: center; flex-wrap: wrap;}\n.fp-chip {background: rgba(255,255,255,.18); border: 1px solid rgba(255,255,255,.3);\n  border-radius: 999px; padding: 4px 12px; font-size: 12.5px;}\n.fp-facts {margin-top: 18px; display: flex; justify-content: space-around;\n  border-top: 1px solid rgba(255,255,255,.25); padding-top: 14px;}\n.fp-facts div {display: flex; flex-direction: column;}\n.fp-facts b {font-size: 17px;}\n.fp-facts span {font-size: 11.5px; opacity: .85;}\n.fp-error {background: #fdecec; border: 1px solid #e34948; color: #8a1f1f;\n  border-radius: 12px; padding: 18px 20px; font-size: 14px;}\n.fp-note {background: #f3f7fd; border-left: 4px solid #2a78d6; border-radius: 8px;\n  padding: 12px 16px; font-size: 14px; color: #0b0b0b;}\n.fp-placeholder {border: 2px dashed #d8d8d4; border-radius: 16px; padding: 40px 20px;\n  text-align: center; color: #8a8a85; font-size: 14.5px;}\nfooter {display: none !important;}\n'
HEADER        = '\n<div id="fp-header">\n  <h1>✈️ Flight Price Prediction</h1>\n  <p>Machine-learning fare estimator trained on 217,510 historical bookings</p>\n  <div class="fp-badges">\n    <span class="fp-badge">Model: XGBoost (tuned)</span>\n    <span class="fp-badge">R² 1.0000</span>\n    <span class="fp-badge">MAE 0.00</span>\n    <span class="fp-badge">9 cities · 3 agencies · 3 cabin classes</span>\n  </div>\n</div>'
PLACEHOLDER   = '<div class="fp-placeholder">Choose a route and press <b>Predict fare</b> to see the estimate.</div>'
ABOUT         = '\n### How the prediction is made\n\n```\nraw booking  ->  feature engineering  ->  one-hot encoding  ->  XGBoost (tuned)  ->  fare\n```\n\n**1. Feature engineering** - the travel date is split into year / month / day / day-of-week /\nquarter / weekend-flag, the route supplies `distance` and `time`, and customer attributes\n(`age`, `gender`, `company`) are joined from the users table.\n\n**2. Preprocessing** - every categorical column is one-hot encoded inside a scikit-learn\n`ColumnTransformer`. The encoder is fitted on the **training data only**, so there is no leakage,\nand it ships inside the same `.joblib` file as the model - this app does no preprocessing of its own.\n\n**3. Model** - XGBoost (tuned), trained on 217,510 bookings.\n\n### Test-set performance\n\n| Metric | Value | Meaning |\n|---|---|---|\n| R² | 1.00000 | share of price variance explained |\n| RMSE | 0.025 | typical error, punishes big mistakes |\n| MAE | 0.004 | average absolute error |\n| MAPE | 0.0006% | average error as a % of the fare |\n\n### Why the accuracy is so high\n\nThe fares in this dataset follow a fixed rate card: for every *(route x cabin class x agency)*\ncombination there is exactly **one** price, equal to `rate(cabin class, agency) x distance`.\nThere are only ~490 such combinations, so a tree-based model reconstructs the rule almost\nperfectly. The notebook proves this in Section 5 and then re-tests the same pipeline on a\nsimulated dynamic-pricing market - where the answer is *not* memorisable - and still reaches\n**R² ≈ 0.91**.\n\n### What the model actually uses\n\n`distance`, `flightType`, `from`, `to` and `agency` carry all the signal.\n`age`, `gender`, `company` and the calendar features score ≈ 0 on permutation importance:\nin this dataset pricing is product-based, not customer-based.\n'


def predict_price(origin, destination, flight_type, agency, travel_date="2023-06-15",
                  age=35, gender="male", company="4You"):
    # Predict one fare. Returns (price, distance, flight_time).
    if origin == destination:
        raise ValueError("Origin and destination must be different.")
    if (origin, destination) not in ROUTE_LOOKUP:
        raise ValueError(f"No historical data for {origin} -> {destination}.")

    distance, duration = ROUTE_LOOKUP[(origin, destination)]
    d = pd.to_datetime(travel_date)
    row = pd.DataFrame([{
        "from": origin, "to": destination, "flightType": flight_type, "agency": agency,
        "gender": gender, "company": company,
        "distance": distance, "time": duration,
        "year": d.year, "month": d.month, "day": d.day,
        "day_of_week": d.dayofweek, "quarter": d.quarter,
        "is_weekend": int(d.dayofweek >= 5), "age": int(age),
    }])[FEATURES]
    return float(MODEL.predict(row)[0]), distance, duration


def result_card(price, origin, destination, flight_type, agency, travel_date, distance, duration):
    return f'''
<div class="fp-card">
  <div class="fp-card-label">Predicted fare</div>
  <div class="fp-card-price">{CURRENCY} {price:,.2f}</div>
  <div class="fp-card-route">{origin} &rarr; {destination}</div>
  <div class="fp-chips">
    <span class="fp-chip">{CLASS_LABEL[flight_type]}</span>
    <span class="fp-chip">{agency}</span>
    <span class="fp-chip">{pd.to_datetime(travel_date).strftime("%d %b %Y")}</span>
  </div>
  <div class="fp-facts">
    <div><b>{distance:,.0f} km</b><span>distance</span></div>
    <div><b>{duration:.2f} h</b><span>flight time</span></div>
    <div><b>{CURRENCY} {price / distance:.2f}</b><span>per km</span></div>
  </div>
</div>'''


def error_card(message):
    return f'<div class="fp-error"><b>Cannot predict.</b><br>{message}</div>'


def class_chart(origin, destination, agency, travel_date, chosen):
    prices = {c: predict_price(origin, destination, c, agency, travel_date)[0] for c in CLASSES}

    fig, ax = plt.subplots(figsize=(6.2, 3.1))
    fig.patch.set_facecolor("white")
    bars = ax.bar([CLASS_LABEL[c] for c in CLASSES], [prices[c] for c in CLASSES],
                  color=[CLASS_COLOURS[c] for c in CLASSES], width=0.55)
    for c, bar in zip(CLASSES, bars):
        bar.set_alpha(1.0 if c == chosen else 0.42)
    ax.bar_label(ax.containers[0], fmt=f"{CURRENCY} %.0f", padding=3,
                 fontsize=9.5, color=MUTED, fontweight="bold")
    ax.set_title(f"What the other cabin classes cost  ({agency})",
                 fontsize=11, fontweight="bold", color=INK, pad=10)
    ax.set_ylabel("Fare", color=MUTED, fontsize=9)
    ax.set_ylim(0, max(prices.values()) * 1.22)
    ax.grid(axis="y", color="#ececea", linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(colors=MUTED, labelsize=9.5)
    fig.tight_layout()
    return fig


def on_predict(origin, destination, flight_type, agency, travel_date, age):
    try:
        price, distance, duration = predict_price(origin, destination, flight_type,
                                                  agency, travel_date, age)
        card = result_card(price, origin, destination, flight_type, agency,
                           travel_date, distance, duration)
        return card, class_chart(origin, destination, agency, travel_date, flight_type)
    except Exception as exc:
        return error_card(exc), None


def on_compare(origin, destination, travel_date):
    try:
        rows = []
        for agency in AGENCIES:
            for cls in CLASSES:
                price, distance, _ = predict_price(origin, destination, cls, agency, travel_date)
                rows.append({"Agency": agency, "Cabin class": CLASS_LABEL[cls],
                             "Fare": round(price, 2), "Per km": round(price / distance, 2)})
        df = pd.DataFrame(rows).sort_values("Fare").reset_index(drop=True)
        df.insert(0, "Rank", range(1, len(df) + 1))

        fig, ax = plt.subplots(figsize=(7.4, 4.9))
        fig.patch.set_facecolor("white")
        labels = df["Agency"] + " · " + df["Cabin class"]
        colours = [PAL["aqua"] if i == 0 else PAL["blue"] for i in range(len(df))]
        ax.barh(labels[::-1], df["Fare"][::-1], color=colours[::-1], height=0.68)
        ax.bar_label(ax.containers[0], fmt=f"{CURRENCY} %.0f", padding=4, fontsize=9, color=MUTED)
        ax.set_title(f"Every option on {origin} → {destination}",
                     fontsize=11.5, fontweight="bold", color=INK, pad=10)
        ax.set_xlabel("Predicted fare", color=MUTED, fontsize=9)
        ax.set_xlim(0, df["Fare"].max() * 1.18)
        ax.grid(axis="x", color="#ececea", linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.tick_params(colors=MUTED, labelsize=9.5)
        fig.tight_layout()

        best = df.iloc[0]
        summary = (f'<div class="fp-note"><b>Cheapest:</b> {best["Agency"]} · '
                   f'{best["Cabin class"]} at {CURRENCY} {best["Fare"]:,.2f} &nbsp;|&nbsp; '
                   f'<b>Most expensive:</b> {CURRENCY} {df["Fare"].max():,.2f} &nbsp;|&nbsp; '
                   f'<b>Spread:</b> {CURRENCY} {df["Fare"].max() - df["Fare"].min():,.2f}</div>')
        return summary, df, fig
    except Exception as exc:
        return error_card(exc), pd.DataFrame(), None


def build_demo():
    with gr.Blocks(title="Flight Price Prediction", css=CSS,
                   theme=gr.themes.Soft(primary_hue="blue")) as demo:
        gr.HTML(HEADER)

        with gr.Tab("💰 Predict a fare"):
            with gr.Row():
                with gr.Column(scale=2):
                    with gr.Row():
                        origin = gr.Dropdown(CITIES, value="Sao Paulo (SP)", label="From")
                        destination = gr.Dropdown(CITIES, value="Rio de Janeiro (RJ)", label="To")
                    flight_type = gr.Radio([(CLASS_LABEL[c], c) for c in CLASSES],
                                           value="economic", label="Cabin class")
                    agency = gr.Dropdown(AGENCIES, value=AGENCIES[0], label="Agency")
                    travel_date = gr.Textbox(value="2023-06-15", label="Travel date (YYYY-MM-DD)")
                    age = gr.Slider(18, 80, value=35, step=1, label="Passenger age")
                    predict_btn = gr.Button("Predict fare", variant="primary", size="lg")
                with gr.Column(scale=3):
                    result = gr.HTML(PLACEHOLDER)
                    chart = gr.Plot(show_label=False)

            gr.Examples(
                examples=[
                    ["Sao Paulo (SP)", "Rio de Janeiro (RJ)", "economic", "CloudFy", "2023-06-15", 35],
                    ["Recife (PE)", "Florianopolis (SC)", "firstClass", "FlyingDrops", "2023-12-24", 42],
                    ["Brasilia (DF)", "Salvador (BH)", "premium", "Rainbow", "2023-08-05", 29],
                    ["Natal (RN)", "Campo Grande (MS)", "economic", "Rainbow", "2023-01-02", 55],
                ],
                inputs=[origin, destination, flight_type, agency, travel_date, age],
                label="Try one of these",
            )

            predict_btn.click(on_predict,
                              inputs=[origin, destination, flight_type, agency, travel_date, age],
                              outputs=[result, chart])

        with gr.Tab("📊 Compare every option"):
            gr.Markdown("Predict the fare for **all agency x cabin-class combinations** on one "
                        "route, so you can see the cheapest way to fly it.<br>"
                        "<sub>Note: in the historical data FlyingDrops sells only First Class - "
                        "its Economy/Premium rows are the model extrapolating.</sub>")
            with gr.Row():
                c_origin = gr.Dropdown(CITIES, value="Sao Paulo (SP)", label="From")
                c_dest = gr.Dropdown(CITIES, value="Salvador (BH)", label="To")
                c_date = gr.Textbox(value="2023-06-15", label="Travel date (YYYY-MM-DD)")
            compare_btn = gr.Button("Compare all options", variant="primary")
            c_summary = gr.HTML()
            with gr.Row():
                c_table = gr.Dataframe(label="All options, cheapest first", interactive=False)
                c_chart = gr.Plot(show_label=False)
            compare_btn.click(on_compare, inputs=[c_origin, c_dest, c_date],
                              outputs=[c_summary, c_table, c_chart])

        with gr.Tab("🧠 How it works"):
            gr.Markdown(ABOUT)

    return demo


if __name__ == "__main__":
    build_demo().launch(server_name="0.0.0.0", server_port=7860)
