#!/usr/bin/env python3
from __future__ import annotations

import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import OrdinalRandomInterceptModel, build_lr_test, build_task_observations, write_csv, write_json


OUTPUT_DIR = Path(__file__).resolve().parents[1] / "results"


def build_h2_rows() -> tuple[list[dict], dict]:
    task_rows, meta = build_task_observations(first_survey_per_user=True)
    trust_by_user: dict[str, list[int]] = defaultdict(list)
    for row in task_rows:
        trust_by_user[row.username].append(row.trust_after)
    means = {user: float(sum(values) / len(values)) for user, values in trust_by_user.items()}
    rows = []
    for row in task_rows:
        rows.append(
            {
                "username": row.username,
                "survey_id": row.survey_id,
                "cohort": row.cohort,
                "cohort_code": row.cohort_code,
                "variant": row.variant,
                "graph": row.graph,
                "ai_error_exposed": row.ai_error_exposed,
                "step": row.step,
                "task_order": row.task_order,
                "video_id": row.video_id,
                "delegation_raw": row.delegation_raw,
                "delegation_label": row.delegation_label,
                "post_trust_raw": row.trust_after,
                "post_trust_reported": row.trust_after + 1,
                "post_trust_within": float(row.trust_after - means[row.username]),
                "mean_post_trust": means[row.username],
            }
        )
    meta["trust_row_count"] = len(rows)
    return rows, meta


def fit_model(rows: list[dict], coefficient_names: list[str], x_builder, lr_predictor_idx: int, additional_lr_tests: dict[str, list[int]] | None = None) -> dict:
    users = [row["username"] for row in rows]
    y = np.array([row["delegation_raw"] for row in rows], dtype=int)
    x = np.array([x_builder(row) for row in rows], dtype=float)
    model = OrdinalRandomInterceptModel(y=y, x=x, users=users, coefficient_names=coefficient_names)
    full_opt, full_params = model.fit()
    full_fit = model.summarize_fit(full_opt, full_params)
    active = [idx for idx in range(len(coefficient_names)) if idx != lr_predictor_idx]
    null_opt, null_params = model.fit(active_predictor_idx=active)
    null_fit = model.summarize_fit(null_opt, null_params)
    lr_tests = {"primary": build_lr_test(full_fit["log_likelihood"], null_fit["log_likelihood"], 1, f"{coefficient_names[lr_predictor_idx]} = 0")}
    lr_tests["primary"]["predictor"] = coefficient_names[lr_predictor_idx]
    if additional_lr_tests:
        for label, retained_predictors in additional_lr_tests.items():
            opt, params = model.fit(active_predictor_idx=retained_predictors)
            restricted = model.summarize_fit(opt, params)
            df = len(coefficient_names) - len(retained_predictors)
            lr_tests[label] = build_lr_test(full_fit["log_likelihood"], restricted["log_likelihood"], df, label)
    return {"full_model": full_fit, "null_model": null_fit, "lr_tests": lr_tests}


def main() -> None:
    rows, meta = build_h2_rows()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    write_csv(
        OUTPUT_DIR / "h2_task_level_long.csv",
        [
            "username", "survey_id", "cohort", "cohort_code", "variant", "graph", "ai_error_exposed",
            "step", "task_order", "video_id", "delegation_raw", "delegation_label",
            "post_trust_raw", "post_trust_reported", "post_trust_within", "mean_post_trust",
        ],
        [
            [
                row["username"], row["survey_id"], row["cohort"], row["cohort_code"], row["variant"], row["graph"], row["ai_error_exposed"],
                row["step"], row["task_order"], row["video_id"], row["delegation_raw"], row["delegation_label"],
                row["post_trust_raw"], row["post_trust_reported"], row["post_trust_within"], row["mean_post_trust"],
            ]
            for row in rows
        ],
    )

    early_rows = [row for row in rows if row["cohort"] == "early"]
    late_rows = [row for row in rows if row["cohort"] == "late"]

    separate_names = ["post_trust_within", "mean_post_trust", "task_order"]
    separate_builder = lambda row: [row["post_trust_within"], row["mean_post_trust"], row["task_order"]]
    pooled_names = ["post_trust_within", "mean_post_trust", "cohort", "graph", "ai_error_exposed", "task_order"]
    pooled_builder = lambda row: [row["post_trust_within"], row["mean_post_trust"], row["cohort_code"], row["graph"], row["ai_error_exposed"], row["task_order"]]

    early_fit = fit_model(early_rows, separate_names, separate_builder, lr_predictor_idx=0)
    late_fit = fit_model(late_rows, separate_names, separate_builder, lr_predictor_idx=0)
    pooled_fit = fit_model(
        rows,
        pooled_names,
        pooled_builder,
        lr_predictor_idx=0,
        additional_lr_tests={
            "drop_graph_and_ai_error": [0, 1, 2, 5],
            "drop_cohort_graph_ai_error": [0, 1, 5],
        },
    )

    summary = {
        "hypothesis": "H2",
        "dataset": meta,
        "models": {"early": early_fit, "late": late_fit, "pooled": pooled_fit},
        "equations": {
            "early_late": "logit(P(Delegation_it <= k)) = alpha_k - (beta_W * PostTrustWithin_it + beta_B * MeanPostTrust_i + beta_T * TaskOrder_t + u_0i)",
            "pooled": "logit(P(Delegation_it <= k)) = alpha_k - (beta_W * PostTrustWithin_it + beta_B * MeanPostTrust_i + beta_C * Cohort_i + beta_G * Graph_i + beta_E * AIErrorExposed_i + beta_T * TaskOrder_t + u_0i)",
        },
    }
    write_json(OUTPUT_DIR / "summary.json", summary)

    coef_rows = []
    for fit_name, fit_payload in [("early", early_fit), ("late", late_fit), ("pooled", pooled_fit)]:
        for predictor, stats in fit_payload["full_model"]["coefficients"].items():
            coef_rows.append([fit_name, predictor, stats["beta"], stats["se"], stats["z"], stats["p_value"], stats["odds_ratio"], stats["ci95_low"], stats["ci95_high"]])
    write_csv(
        OUTPUT_DIR / "coefficients.csv",
        ["fit", "predictor", "beta", "se", "z", "p_value", "odds_ratio", "ci95_low", "ci95_high"],
        coef_rows,
    )

    report_lines = [
        "H2: Trust Predicts Delegation",
        "=============================",
        "",
        "Hypothesis:",
        "H2: Trust predicts delegation choices within individuals.",
        "",
        "Coding choices:",
        f"- First survey per user only, so the analysis uses n={meta['n_users']} users.",
        "- Delegation ordering: never < verify (stored in the webapp as oversee) < totally.",
        "- Graph_i = 1 for A/C and 0 for B/D.",
        "- AIErrorExposed_i = 1 only for late-cohort A/B users; 0 otherwise.",
        "- Cohort_i = 0 for the early cohort and 1 for the late cohort.",
    ]
    for note in meta["cohort_metadata"]["notes"]:
        report_lines.append(f"- Cohort note: {note}")
    report_lines += [
        "",
        "Model equations:",
        "- Early and late cohorts separately: logit(P(Delegation_it <= k)) = alpha_k - (beta_W * PostTrustWithin_it + beta_B * MeanPostTrust_i + beta_T * TaskOrder_t + u_0i)",
        f"- Pooled n={meta['n_users']} model: logit(P(Delegation_it <= k)) = alpha_k - (beta_W * PostTrustWithin_it + beta_B * MeanPostTrust_i + beta_C * Cohort_i + beta_G * Graph_i + beta_E * AIErrorExposed_i + beta_T * TaskOrder_t + u_0i)",
        "",
    ]
    for fit_name, fit_payload, n_users in [("Early cohort", early_fit, len({row['username'] for row in early_rows})), ("Late cohort", late_fit, len({row['username'] for row in late_rows})), ("Pooled sample", pooled_fit, len({row['username'] for row in rows}))]:
        full = fit_payload["full_model"]
        report_lines += [
            fit_name,
            "-" * len(fit_name),
            f"- Users: {n_users}",
            f"- Task observations: {full['n_observations']}",
            f"- Log-likelihood: {full['log_likelihood']:.6f}",
            f"- AIC: {full['aic']:.6f}",
            f"- BIC: {full['bic']:.6f}",
            f"- Thresholds: {', '.join(f'{value:.6f}' for value in full['thresholds_raw'])}",
            f"- sd(user intercept): {full['sd_user_intercept']:.6f}",
            "Fixed effects:",
        ]
        for predictor, stats in full["coefficients"].items():
            report_lines.append(
                f"- {predictor}: beta={stats['beta']:.6f}, SE={stats['se'] if stats['se'] is not None else 'NA'}, z={stats['z'] if stats['z'] is not None else 'NA'}, p={stats['p_value'] if stats['p_value'] is not None else 'NA'}, OR={stats['odds_ratio']:.6f}"
            )
        for test_name, result in fit_payload["lr_tests"].items():
            if "predictor" in result:
                report_lines.append(f"- LR test for {result['predictor']} = 0: statistic={result['lr_statistic']:.6f}, df={result['df']}, p={result['p_value']:.6g}")
            else:
                report_lines.append(f"- LR test {test_name}: statistic={result['lr_statistic']:.6f}, df={result['df']}, p={result['p_value']:.6g}")
        report_lines.append("")
    (OUTPUT_DIR / "report.txt").write_text("\n".join(report_lines), encoding="utf-8")
    print(f"Wrote H2 outputs to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
