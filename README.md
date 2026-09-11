# ReFiNe project: preliminary analyses

This codebase reproduces the empirical results reported in
the paper "Following Delegation Traces to Study Human Oversight in Interaction with Multimodal AI", accepted at the 6th Workshop on Explainable AI, Generative and Agentic Systems: Trust, Transparency, and Human Oversight, @ AIxIA 2026.

The anonymised data collected in our preliminary (N=56) user study can be found on Zenodo: https://zenodo.org/records/21790700
You may download them and copy them under a ```./data``` sub-folder following the main file tree of this repo.

## Citation

If you use this code or the associated dataset, please cite our paper:

```bibtex
@inproceedings{chiatti2026delegationtraces,
  author    = {Chiatti, Agnese and Matteucci, Matteo and Schiaffonati, Viola},
  title     = {Following Delegation Traces to Study Human Oversight in Interaction with Multimodal {AI}},
  booktitle = {Proceedings of the 6th Workshop on Explainable AI, Generative and Agentic Systems: Trust, Transparency, and Human Oversight},
  year      = {2026},
  publisher = {CEUR-WS.org},
  note      = {Accepted for publication}
}
``` 

## How to install requirements and run this code

```bash
python3 -m pip install -r requirements.txt
./run_all.sh
```

Results are written to `results/`:

- `summary.json`, `coefficients.csv`, and `report.txt` reproduce the model.
- `figure_3_joint_distribution.csv` and `figure_4_quadrants.csv` reproduce
  the figures' data.
- `paper_descriptives.txt` reports the counts quoted in Section 4.1.

