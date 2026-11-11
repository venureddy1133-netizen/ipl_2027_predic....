# IPL 2027 Machine Learning Predictor & Auction Simulator

## Files Included
1. `ipl_2027_ml_pipeline.py` — Complete, self-contained Python Machine Learning & Monte Carlo simulation script.
   - Builds a local SQLite database (`ipl_2027_analytics.db`) with 120 IPL players and 10 franchise stadium profiles.
   - Computes 12-man Impact Player phase-split team vectors (Powerplay, Middle, Death batting SR & bowling economy, Batting RAA, Bowling WAR, Spin matchup, Dew/Toss bias).
   - Trains a calibrated `GradientBoostingClassifier` (`ROC-AUC: 0.7532`, `Accuracy: 71.25%`).
   - Runs a 5,000-season Monte Carlo simulation of the 70-match league stage + IPL Playoff Bracket (Qualifier 1, Eliminator, Qualifier 2, Final).
   - Includes a dynamic `simulate_player_transfer()` function to test 2027 Mini-Auction trades.

2. `ipl_2027_simulator.html` — Single-file interactive web dashboard (works on macOS desktop browsers & iPhone Safari) to test player transfers and run 2,500 Monte Carlo seasons in real time.

## How to Run on macOS (Apple Silicon / M1)
```bash
python3 -m venv venv
source venv/bin/activate
pip install numpy pandas scikit-learn
python3 ipl_2027_ml_pipeline.py
```
