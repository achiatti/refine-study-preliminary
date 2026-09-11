# AIxIA 2026 workshop-paper reproduction package

This compact package reproduces the empirical results reported in
`AIxIA_2026_XAI_it_workshop_version.pdf`: the pooled cumulative-logit
random-intercept model in Table 2 and the numerical contents of Figures 3 and
4. It intentionally excludes unrelated analyses, manuscripts, presentations,
raw exports, consent material, participant lists, dependencies, and web-app
data.

## Run

```bash
python3 -m pip install -r requirements.txt
./run_all.sh
```

Results are written to `results/`:

- `summary.json`, `coefficients.csv`, and `report.txt` reproduce the model.
- `figure_3_joint_distribution.csv` and `figure_4_quadrants.csv` reproduce
  the figures' data.
- `paper_descriptives.txt` reports the counts quoted in Section 4.1.

## Privacy

`data/` is copied solely from `../zenodo_data_and_analyses/anonymised-study-data`.
It uses release-local `anon_*` participant labels only. No real-person
identifiers or original raw-study exports are included.
