# Peltier–Evaporative Cooling Recommender

This Streamlit application recommends among the 48 experimentally evaluated packing configurations using the fuzzy preference-space framework developed in the accompanying study.

## Capabilities

- Balanced and specialist preference presets
- Custom cooling, pressure-control and IAQ priorities
- Automatic weight normalization
- Optional limits for all six measured responses
- Pareto-efficient recommendation set
- Explicit near-tie/indifference warning
- Runner-up and score-margin reporting
- Plain-language explanation
- Counterfactual preference change
- Comparison of measured responses
- Downloadable recommendation record
- Closest tested configuration when no design satisfies all limits

## Run in Google Colab

1. Open `Launch_Peltier_Recommender_in_Colab.ipynb` in Google Colab.
2. Upload `peltier_recommender.zip` when requested.
3. Follow the notebook cells to install the packages and enter a free ngrok authentication token.
4. Open the temporary application link printed by the final cell.

The link remains active while the Colab runtime is running. Local execution or Streamlit Community Cloud is preferable for routine or public use.

## Run on your computer

Install Python 3.10 or newer. Extract the ZIP, open a terminal in its parent directory and run:

```bash
python -m venv .venv
```

Activate the environment:

- Windows: `.venv\Scripts\activate`
- macOS/Linux: `source .venv/bin/activate`

Install and launch:

```bash
pip install -r peltier_recommender/requirements.txt
streamlit run peltier_recommender/app.py
```

The browser should open at `http://localhost:8501`.

## Deploy with Streamlit Community Cloud

1. Create a GitHub repository.
2. Upload the complete `peltier_recommender` folder.
3. Sign in at [share.streamlit.io](https://share.streamlit.io/).
4. Select the repository and set the main file to `peltier_recommender/app.py`.
5. Deploy.

## Scientific scope

The tool recommends only among the 48 observed configurations and the operating conditions represented by the experiments. It does not predict the performance of untested materials, thicknesses or densities. The bundled dimension scores use CRITIC within-group weighting, interval-fuzzy satisfaction, a risk penalty of 0.15 and an indifference margin of 0.001.
