# IPL 2027 Prediction & Simulation Platform 🏏

This project explores how machine learning and historical cricket data can be used to estimate team performance in the 2027 Indian Premier League (IPL).

I built this project to combine historical IPL statistics, player performance metrics, and match simulations into one platform. It includes a Python-based prediction pipeline and an interactive dashboard where users can explore team rankings, compare match probabilities, and experiment with player transfers.

The predictions are intended for analysis and experimentation rather than as guaranteed outcomes.

## What the Project Does

- Uses historical IPL data covering the 2008–2026 seasons.
- Stores and organizes historical cricket statistics in SQLite.
- Evaluates team and player performance using different statistical features.
- Applies a Gradient Boosting classification model to estimate match outcomes.
- Uses Monte Carlo simulations to explore possible IPL 2027 tournament scenarios.
- Provides an interactive dashboard to compare teams and explore championship probabilities.
- Includes a player transfer sandbox to examine how changes in team composition may affect projected performance.

## Project Structure

```text
ipl_2027_predic/
│
├── ipl_2027_ml_pipeline.py
│   └── Python prediction and simulation pipeline
│
├── ipl_2027_simulator.html
│   └── Interactive IPL dashboard
│
├── ipl_2008_2026_historical.db
│   └── Historical IPL SQLite database
│
├── README.md
│   └── Project documentation
│
└── .gitignore
```

## Technologies Used

- **Python** – Main programming language
- **NumPy & Pandas** – Data processing and numerical operations
- **Scikit-learn** – Machine learning and model evaluation
- **SQLite** – Historical data storage
- **HTML, CSS & JavaScript** – Interactive dashboard
- **Monte Carlo Simulation** – Tournament scenario analysis

## Machine Learning Approach

The Python pipeline uses historical performance information and engineered features to build a match prediction model.

Some of the factors considered include:

- Historical team performance and Elo ratings
- Batting performance across different match phases
- Bowling performance and wicket-taking ability
- Powerplay, middle-over, and death-over statistics
- Recent player performance and team composition

A Gradient Boosting classifier is used for match outcome estimation. Model evaluation includes accuracy, ROC-AUC, and Brier score.

The project also uses Monte Carlo simulation to explore different tournament outcomes and estimate playoff and championship probabilities.

## Interactive Dashboard

The HTML dashboard provides a visual way to explore the prediction results without having to run Python commands.

It includes:

- Projected IPL 2027 team standings
- Playoff qualification probabilities
- Final and championship probabilities
- Head-to-head match probability comparisons
- Player transfer experiments
- Option to rerun tournament simulations

To open the dashboard, simply open the HTML file in Chrome, Safari, or another modern browser.

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/venureddy1133-netizen/ipl_2027_predic.git
cd ipl_2027_predic
```

### 2. Create a virtual environment

```bash
python3 -m venv venv
```

Activate it on macOS:

```bash
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install numpy pandas scikit-learn
```

### 4. Run the prediction pipeline

```bash
python3 ipl_2027_ml_pipeline.py
```

### 5. Open the dashboard

Open `ipl_2027_simulator.html` in your browser.

## Future Improvements

There are several areas I would like to improve as the project develops:

- Integrate updated player statistics and confirmed auction information.
- Improve model performance through additional feature engineering.
- Incorporate venue conditions, toss outcomes, and player availability.
- Compare multiple machine learning algorithms.
- Add more detailed player-level performance predictions.
- Build a live dashboard with updated match and tournament information.

## Note on Predictions

IPL 2027 projections are experimental estimates based on historical data, model assumptions, and simulated scenarios. Actual team squads, player availability, match conditions, and tournament results may differ from the assumptions used by the model.

This project is intended for learning, experimentation, and cricket analytics.

---

**Developed as a personal machine learning and sports analytics project.**
