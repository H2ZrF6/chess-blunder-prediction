# Chess Outcome & Blunder Prediction

Two questions: can you predict who wins a chess game before it starts, and can you predict when a player's about to blunder mid-game?

Short answer: not really, and yes.

## Results

| Task | Result |
|---|---|
| Predict winner from Elo + opening + time control | ~52% accuracy just beating the Elo formula alone (51%) |
| Same, but only large Elo gaps (400+) | Accuracy jumps up as expected |
| Predict blunders from clock time, move number, eval | ROC-AUC ~0.75, stable across 10 folds |
| Upset rate vs blunder differential | Upsets track blunders closely and favours blunder more|

Turns out single games are noisy and close to a coin flip once the ratings are close. But blunders themselves are predictable from stuff like time pressure, and blunders predictably cause upsets.

## Notebooks

- `Lichess Data Collection.ipynb` - Loads data from LiChess and stores it into pickle files after some cleanup. We only pull games with stockfish evaluations attached.
- `Exploratory Data Analysis.ipynb` — Elo distributions, draw rates, game length by event
- `modeling.ipynb` — win/loss/draw model, Elo baseline comparison, calibration


## Data

- `df` — one row per game
- `df_moves` — one row per move, with clock time and engine eval before/after

## A few things worth knowing before you dig in

- `df_moves` splits use `GroupKFold`/`GroupShuffleSplit` keyed on game, not random row splits — otherwise moves from the same game leak across train/test.
- `Eval_After` and `Eval_Drop` aren't used as model inputs for blunder prediction since they're derived from the blunder move itself (that's leakage).
- `Opening` got collapsed from ~2000 raw values down to the top 50 + "Other" since gradient boosting caps out at 255 categories.
- Mate scores show up as huge  values (~20000) in the raw eval data clipped to ±1000.

```
pandas
numpy
scikit-learn
matplotlib
seaborn
```
