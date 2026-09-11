#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import math
import os
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
import re
import unicodedata

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logsumexp
from scipy.stats import norm
from scipy.stats import chi2


ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parent
DATA_ROOT = REPO_ROOT / "data"


def _first_existing_path(*candidates: Path) -> Path:
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


def _env_or_path(env_var: str, fallback: Path) -> Path:
    value = os.environ.get(env_var)
    return Path(value).expanduser() if value else fallback


SURVEY_DIR = _first_existing_path(
    DATA_ROOT / "survey-results",
)
EDITED_GRAPHS_DIR = _first_existing_path(
    DATA_ROOT / "graphs-edited-by-the-users",
)
USERSBYDATE_CSV = _first_existing_path(
    DATA_ROOT / "usersbydate_anonymised.csv",
)
USER_TABLE_CSV = _first_existing_path(
    DATA_ROOT / "user_table_anonymised.csv",
)
GENERATED_DATASET = _first_existing_path(
    _env_or_path("REFINE_GENERATED_DATASET", REPO_ROOT / "anonymised-study-data" / "graph_gen_sequence.min.json"),
    REPO_ROOT / "anonymised-study-data" / "graph_gen_sequence.min.json",
)
GENERATED_DATASET_ERROR = _first_existing_path(
    _env_or_path("REFINE_GENERATED_DATASET_ERROR", REPO_ROOT / "anonymised-study-data" / "graph_gen_sequence.error_injection.min.json"),
    REPO_ROOT / "anonymised-study-data" / "graph_gen_sequence.error_injection.min.json",
)
GROUND_TRUTH_DATASET = _first_existing_path(
    _env_or_path("REFINE_GROUND_TRUTH_DATASET", REPO_ROOT / "anonymised-study-data" / "graph_sequence.min.json"),
    REPO_ROOT / "anonymised-study-data" / "graph_sequence.min.json",
)

REQUESTED_COHORT_CUTOFF = "2026-05-06"
REQUESTED_COHORT_COUNTS = {"early": 31, "late": 25}
AGREEMENT_ELIGIBLE_CUTOFF = "2026-05-01"

DELEGATION_ORDER = {"never": 0, "oversee": 1, "totally": 2}
DELEGATION_LABELS = {0: "never", 1: "verify", 2: "totally"}
GRAPH_BY_VARIANT = {"A": 1, "B": 0, "C": 1, "D": 0}


@dataclass(frozen=True)
class TaskObservation:
    username: str
    survey_id: int
    cohort: str
    cohort_code: int
    last_survey_date: str
    variant: str
    graph: int
    ai_error_exposed: int
    step: int
    task_order: int
    video_id: str
    trust_before: int
    trust_after: int
    trust_delta: int
    delegation_raw: int
    delegation_label: str
    watch_time_s: float
    replay_count: float
    graph_click_count: float
    graph_zoom_count: float
    graph_drag_count: float
    graph_total_interaction_count: float
    graph_edit_count: float
    graph_delete_count: float
    graph_interactions_augmented: float


@dataclass(frozen=True)
class AgreementObservation:
    username: str
    survey_id: int
    cohort: str
    cohort_code: int
    last_survey_date: str
    agreement_eligible: int
    variant: str
    graph: int
    ai_error_exposed: int
    step: int
    task_order: int
    video_id: str
    trust_before: int
    trust_after: int
    trust_delta: int
    delegation_raw: int
    delegation_label: str
    agreement_text: int
    disagreement_text: int
    answer_user_before: str
    shown_answer: str
    watch_time_s: float
    replay_count: float
    graph_interactions_augmented: float


def normalize_username(value: str) -> str:
    return str(value or "").strip().lower()


def load_user_variants() -> dict[str, str]:
    by_user: dict[str, str] = {}
    if not USER_TABLE_CSV.exists():
        return by_user
    with USER_TABLE_CSV.open(newline="", encoding="utf-8-sig") as fh:
        for row in csv.DictReader(fh):
            username = normalize_username(row.get("username", ""))
            variant = str(row.get("variant", "")).strip().upper()
            if username and variant:
                by_user[username] = variant
    return by_user


def load_cohort_assignments() -> tuple[dict[str, dict], dict]:
    assignments: dict[str, dict] = {}
    with USERSBYDATE_CSV.open(newline="", encoding="utf-8-sig") as fh:
        for row in csv.DictReader(fh):
            username = normalize_username(row.get("user_username", ""))
            last_date = str(row.get("last_survey_date", "")).strip()
            if not username or not last_date:
                continue
            cohort = "early" if last_date < REQUESTED_COHORT_CUTOFF else "late"
            assignments[username] = {
                "username": username,
                "last_survey_date": last_date,
                "cohort": cohort,
                "cohort_code": 0 if cohort == "early" else 1,
            }

    counts = Counter(item["cohort"] for item in assignments.values())
    notes: list[str] = []
    if dict(counts) != REQUESTED_COHORT_COUNTS:
        notes.append(
            f"Literal cutoff {REQUESTED_COHORT_CUTOFF} produced counts {dict(counts)} instead of {REQUESTED_COHORT_COUNTS}."
        )

    metadata = {
        "requested_cutoff": REQUESTED_COHORT_CUTOFF,
        "requested_counts": REQUESTED_COHORT_COUNTS,
        "resolved_counts": dict(counts),
        "notes": notes,
    }
    return assignments, metadata


def load_graph_edit_counts() -> dict[tuple[str, int, str], tuple[int, int]]:
    counts: dict[tuple[str, int, str], list[int]] = defaultdict(lambda: [0, 0])
    for path in sorted(EDITED_GRAPHS_DIR.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        for row in data:
            key = (
                normalize_username(row.get("user_username", "")),
                int(row["survey_id"]),
                str(row.get("video_id") or "").strip(),
            )
            new_label = str(row.get("graph_element_newlabel") or "").strip()
            if new_label:
                counts[key][0] += 1
            else:
                counts[key][1] += 1
    return {key: (vals[0], vals[1]) for key, vals in counts.items()}


def load_payloads(first_survey_per_user: bool = True) -> tuple[list[dict], dict]:
    cohort_map, cohort_meta = load_cohort_assignments()
    variant_map = load_user_variants()
    payloads: list[dict] = []
    for path in sorted(SURVEY_DIR.glob("**/*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        answers = sorted(payload.get("answers", []), key=lambda row: int(row["step"]))
        if not answers:
            continue
        username = normalize_username(answers[0].get("user_username", ""))
        if username not in cohort_map:
            continue
        variant = str(payload.get("variant") or variant_map.get(username) or "").strip().upper()
        if variant not in GRAPH_BY_VARIANT:
            continue
        metrics_by_step = {
            int(item.get("step", 0)): item.get("task_metrics") or {}
            for item in payload.get("metrics", [])
            if item.get("task_metrics")
        }
        payloads.append(
            {
                "path": path,
                "username": username,
                "survey_id": int(answers[0]["survey_id"]),
                "variant": variant,
                "answers": answers,
                "metrics_by_step": metrics_by_step,
                **cohort_map[username],
            }
        )

    if first_survey_per_user:
        by_user: dict[str, dict] = {}
        for payload in payloads:
            current = by_user.get(payload["username"])
            if current is None or payload["survey_id"] < current["survey_id"]:
                by_user[payload["username"]] = payload
        payloads = sorted(by_user.values(), key=lambda item: (item["username"], item["survey_id"]))
    return payloads, cohort_meta


def build_task_observations(first_survey_per_user: bool = True) -> tuple[list[TaskObservation], dict]:
    payloads, cohort_meta = load_payloads(first_survey_per_user=first_survey_per_user)
    graph_edits = load_graph_edit_counts()
    rows: list[TaskObservation] = []
    skipped: list[str] = []
    for payload in payloads:
        for answer in payload["answers"]:
            delegation_key = str(answer.get("delegate_summary") or "").strip()
            trust_before = answer.get("answer_trustmeter_before")
            trust_after = answer.get("answer_trustmeter_after")
            if delegation_key not in DELEGATION_ORDER or trust_before is None or trust_after is None:
                skipped.append(f"{payload['username']}:{payload['survey_id']}:{answer.get('step')}")
                continue
            step = int(answer["step"])
            metrics = payload["metrics_by_step"].get(step, {})
            video_id = str(answer.get("video_id") or "").strip()
            edit_count, delete_count = graph_edits.get((payload["username"], payload["survey_id"], video_id), (0, 0))
            graph_total = float(metrics.get("graph_total_interaction_count") or 0.0)
            rows.append(
                TaskObservation(
                    username=payload["username"],
                    survey_id=payload["survey_id"],
                    cohort=payload["cohort"],
                    cohort_code=payload["cohort_code"],
                    last_survey_date=payload["last_survey_date"],
                    variant=payload["variant"],
                    graph=GRAPH_BY_VARIANT[payload["variant"]],
                    ai_error_exposed=int(payload["cohort"] == "late" and payload["variant"] in {"A", "B"}),
                    step=step,
                    task_order=step - 1,
                    video_id=video_id,
                    trust_before=int(trust_before),
                    trust_after=int(trust_after),
                    trust_delta=int(trust_after) - int(trust_before),
                    delegation_raw=DELEGATION_ORDER[delegation_key],
                    delegation_label=DELEGATION_LABELS[DELEGATION_ORDER[delegation_key]],
                    watch_time_s=float(metrics.get("video_watch_time_s") or 0.0),
                    replay_count=float(metrics.get("video_replay_after_completion_count") or 0.0),
                    graph_click_count=float(metrics.get("graph_click_count") or 0.0),
                    graph_zoom_count=float(metrics.get("graph_zoom_count") or 0.0),
                    graph_drag_count=float(metrics.get("graph_drag_count") or 0.0),
                    graph_total_interaction_count=graph_total,
                    graph_edit_count=float(edit_count),
                    graph_delete_count=float(delete_count),
                    graph_interactions_augmented=graph_total + float(edit_count) + float(delete_count),
                )
            )
    meta = {
        "cohort_metadata": cohort_meta,
        "first_survey_per_user": first_survey_per_user,
        "skipped_rows": skipped,
        "n_users": len({row.username for row in rows}),
        "n_survey_runs": len({(row.username, row.survey_id) for row in rows}),
        "n_rows": len(rows),
    }
    return rows, meta


def finite_difference_hessian(func, x: np.ndarray, step: float = 1e-4) -> np.ndarray:
    n = len(x)
    hessian = np.zeros((n, n), dtype=float)
    fx = float(func(x))
    for i in range(n):
        ei = np.zeros(n, dtype=float)
        ei[i] = step
        hessian[i, i] = (float(func(x + ei)) - 2.0 * fx + float(func(x - ei))) / (step ** 2)
        for j in range(i + 1, n):
            ej = np.zeros(n, dtype=float)
            ej[j] = step
            value = (
                float(func(x + ei + ej))
                - float(func(x + ei - ej))
                - float(func(x - ei + ej))
                + float(func(x - ei - ej))
            ) / (4.0 * step ** 2)
            hessian[i, j] = value
            hessian[j, i] = value
    return hessian


def zscore(values: list[float]) -> list[float]:
    arr = np.array(values, dtype=float)
    if len(arr) == 0:
        return []
    std = float(arr.std(ddof=0))
    if std == 0.0:
        return [0.0] * len(values)
    mean = float(arr.mean())
    return [float((value - mean) / std) for value in values]


def trust_zone(trust_after: int) -> int:
    if trust_after <= 1:
        return 0
    if trust_after == 2:
        return 1
    return 2


def mismatch_zone(trust_after: int, delegation_raw: int) -> int:
    return int(trust_zone(trust_after) != delegation_raw)


def trust_high(trust_after: int) -> int:
    return int(trust_after >= 3)


def delegation_high(delegation_raw: int) -> int:
    return int(delegation_raw >= 2)


def four_zone_label(trust_after: int, delegation_raw: int) -> str:
    mapping = {
        (0, 0): "low_trust_low_delegation",
        (0, 1): "low_trust_high_delegation",
        (1, 0): "high_trust_low_delegation",
        (1, 1): "high_trust_high_delegation",
    }
    return mapping[(trust_high(trust_after), delegation_high(delegation_raw))]


def four_zone_mismatch(trust_after: int, delegation_raw: int) -> int:
    return int(trust_high(trust_after) != delegation_high(delegation_raw))


def normalize_answer(value: str | None) -> str:
    text = unicodedata.normalize("NFKC", str(value or ""))
    text = text.strip().lower()
    return re.sub(r"\s+", " ", text)


def load_dataset_index(path: Path) -> dict[str, dict]:
    if not path.exists():
        raise FileNotFoundError(
            f"Required displayed-answer dataset not found at {path}. "
            "Use the committed derived task-level CSVs for H3/H7/H8, or provide the original answer-key files."
        )
    data = json.loads(path.read_text(encoding="utf-8"))
    return {
        str(row.get("id") or "").strip(): row
        for row in data
        if isinstance(row, dict) and str(row.get("id") or "").strip()
    }


def shown_answer_for_variant(variant: str, video_id: str, generated_index: dict[str, dict], ground_truth_index: dict[str, dict]) -> str:
    if variant in {"A", "B"}:
        row = generated_index.get(video_id, {})
    elif variant in {"C", "D"}:
        row = ground_truth_index.get(video_id, {})
    else:
        row = {}
    return str(row.get("answer") or "").strip()


def build_agreement_observations() -> tuple[list[AgreementObservation], dict]:
    payloads, cohort_meta = load_payloads(first_survey_per_user=True)
    graph_edits = load_graph_edit_counts()
    generated_index = load_dataset_index(GENERATED_DATASET)
    ground_truth_index = load_dataset_index(GROUND_TRUTH_DATASET)

    rows: list[AgreementObservation] = []
    skipped: list[str] = []
    for payload in payloads:
        agreement_eligible = int(payload["last_survey_date"] < AGREEMENT_ELIGIBLE_CUTOFF)
        for answer in payload["answers"]:
            delegation_key = str(answer.get("delegate_summary") or "").strip()
            trust_before = answer.get("answer_trustmeter_before")
            trust_after = answer.get("answer_trustmeter_after")
            user_guess = str(answer.get("answer_user_before") or "").strip()
            video_id = str(answer.get("video_id") or "").strip()
            shown_answer = shown_answer_for_variant(payload["variant"], video_id, generated_index, ground_truth_index)
            if delegation_key not in DELEGATION_ORDER or trust_before is None or trust_after is None or not user_guess or not shown_answer:
                skipped.append(f"{payload['username']}:{payload['survey_id']}:{answer.get('step')}")
                continue
            step = int(answer["step"])
            metrics = payload["metrics_by_step"].get(step, {})
            edit_count, delete_count = graph_edits.get((payload["username"], payload["survey_id"], video_id), (0, 0))
            graph_total = float(metrics.get("graph_total_interaction_count") or 0.0)
            agreement_text = int(normalize_answer(user_guess) == normalize_answer(shown_answer))
            rows.append(
                AgreementObservation(
                    username=payload["username"],
                    survey_id=payload["survey_id"],
                    cohort=payload["cohort"],
                    cohort_code=payload["cohort_code"],
                    last_survey_date=payload["last_survey_date"],
                    agreement_eligible=agreement_eligible,
                    variant=payload["variant"],
                    graph=GRAPH_BY_VARIANT[payload["variant"]],
                    ai_error_exposed=int(payload["cohort"] == "late" and payload["variant"] in {"A", "B"}),
                    step=step,
                    task_order=step - 1,
                    video_id=video_id,
                    trust_before=int(trust_before),
                    trust_after=int(trust_after),
                    trust_delta=int(trust_after) - int(trust_before),
                    delegation_raw=DELEGATION_ORDER[delegation_key],
                    delegation_label=DELEGATION_LABELS[DELEGATION_ORDER[delegation_key]],
                    agreement_text=agreement_text,
                    disagreement_text=1 - agreement_text,
                    answer_user_before=user_guess,
                    shown_answer=shown_answer,
                    watch_time_s=float(metrics.get("video_watch_time_s") or 0.0),
                    replay_count=float(metrics.get("video_replay_after_completion_count") or 0.0),
                    graph_interactions_augmented=graph_total + float(edit_count) + float(delete_count),
                )
            )
    meta = {
        "cohort_metadata": cohort_meta,
        "agreement_eligible_cutoff": AGREEMENT_ELIGIBLE_CUTOFF,
        "skipped_rows": skipped,
        "n_users": len({row.username for row in rows}),
        "n_survey_runs": len({(row.username, row.survey_id) for row in rows}),
        "n_rows": len(rows),
        "n_agreement_eligible_users": len({row.username for row in rows if row.agreement_eligible == 1}),
    }
    return rows, meta


def ordered_thresholds(raw_params: np.ndarray) -> np.ndarray:
    values = [float(raw_params[0])]
    for delta in raw_params[1:]:
        values.append(values[-1] + math.exp(float(delta)))
    return np.array(values, dtype=float)


def initial_threshold_params(y: np.ndarray, n_thresholds: int) -> np.ndarray:
    probs = []
    n = len(y)
    for cutoff in range(n_thresholds):
        p = float(np.sum(y <= cutoff)) / float(n)
        p = min(max(p, 1e-4), 1.0 - 1e-4)
        probs.append(math.log(p / (1.0 - p)))
    raw = [probs[0]]
    for idx in range(1, len(probs)):
        raw.append(math.log(max(probs[idx] - probs[idx - 1], 1e-3)))
    return np.array(raw, dtype=float)


def category_log_prob_matrix(y: np.ndarray, eta: np.ndarray, thresholds: np.ndarray, n_categories: int) -> np.ndarray:
    probs = np.zeros_like(eta, dtype=float)
    cdfs = [expit(threshold - eta) for threshold in thresholds]
    for category in range(n_categories):
        mask = y == category
        if not np.any(mask):
            continue
        if category == 0:
            probs[:, mask] = cdfs[0][:, mask]
        elif category == n_categories - 1:
            probs[:, mask] = (1.0 - cdfs[-1])[:, mask]
        else:
            probs[:, mask] = (cdfs[category] - cdfs[category - 1])[:, mask]
    return np.log(np.clip(probs, 1e-12, 1.0))


class OrdinalRandomInterceptModel:
    def __init__(
        self,
        y: np.ndarray,
        x: np.ndarray,
        users: list[str],
        coefficient_names: list[str],
        quadrature_points: int = 9,
        maxiter: int = 300,
    ) -> None:
        self.y = y.astype(int)
        self.x = x.astype(float)
        self.users = users
        self.coefficient_names = coefficient_names
        self.maxiter = maxiter
        self.n_categories = int(np.max(self.y)) + 1
        self.n_thresholds = self.n_categories - 1
        self.unique_users = list(dict.fromkeys(users))
        user_arr = np.array(users)
        self.group_slices = [np.where(user_arr == user)[0] for user in self.unique_users]
        nodes, weights = np.polynomial.hermite.hermgauss(quadrature_points)
        self.node = math.sqrt(2.0) * nodes
        self.node_log_weights = np.log(weights) - 0.5 * math.log(math.pi)

    def unpack_params(self, params: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
        thresholds = ordered_thresholds(params[: self.n_thresholds])
        beta = params[self.n_thresholds : self.n_thresholds + self.x.shape[1]]
        sd_user = math.exp(float(params[-1]))
        return thresholds, beta, sd_user

    def initial_params(self) -> np.ndarray:
        return np.concatenate(
            [
                initial_threshold_params(self.y, self.n_thresholds),
                np.zeros(self.x.shape[1], dtype=float),
                np.array([math.log(0.8)], dtype=float),
            ]
        )

    def log_likelihood(self, params: np.ndarray) -> float:
        thresholds, beta, sd_user = self.unpack_params(params)
        total = 0.0
        for idx in self.group_slices:
            eta_fixed = self.x[idx] @ beta
            eta = eta_fixed[None, :] + (sd_user * self.node)[:, None]
            logp = category_log_prob_matrix(self.y[idx], eta, thresholds, self.n_categories)
            total += float(logsumexp(self.node_log_weights + np.sum(logp, axis=1)))
        return total

    def fit(self, active_predictor_idx: list[int] | None = None) -> tuple[object, np.ndarray]:
        if active_predictor_idx is None:
            objective = lambda p: -self.log_likelihood(p)
            opt = minimize(objective, self.initial_params(), method="L-BFGS-B", options={"maxiter": self.maxiter})
            return opt, opt.x

        init = self.initial_params()
        active_predictor_idx = list(active_predictor_idx)
        reduced_init = np.concatenate(
            [
                init[: self.n_thresholds],
                init[self.n_thresholds + np.array(active_predictor_idx)],
                init[-1:],
            ]
        )

        def objective(reduced: np.ndarray) -> float:
            full = np.concatenate(
                [
                    reduced[: self.n_thresholds],
                    np.zeros(self.x.shape[1], dtype=float),
                    reduced[-1:],
                ]
            )
            beta_values = reduced[self.n_thresholds : -1]
            for pos, active_idx in enumerate(active_predictor_idx):
                full[self.n_thresholds + active_idx] = beta_values[pos]
            return -self.log_likelihood(full)

        opt = minimize(objective, reduced_init, method="L-BFGS-B", options={"maxiter": self.maxiter})
        full = np.concatenate(
            [
                opt.x[: self.n_thresholds],
                np.zeros(self.x.shape[1], dtype=float),
                opt.x[-1:],
            ]
        )
        beta_values = opt.x[self.n_thresholds : -1]
        for pos, active_idx in enumerate(active_predictor_idx):
            full[self.n_thresholds + active_idx] = beta_values[pos]
        return opt, full

    def summarize_fit(self, opt: object, params: np.ndarray) -> dict:
        ll = self.log_likelihood(params)
        thresholds, beta, sd_user = self.unpack_params(params)
        objective = lambda p: -self.log_likelihood(p)
        beta_summary = {}
        try:
            hessian = finite_difference_hessian(objective, params)
            cov = np.linalg.inv(hessian)
            for idx, name in enumerate(self.coefficient_names):
                param_idx = self.n_thresholds + idx
                estimate = float(beta[idx])
                variance = float(cov[param_idx, param_idx])
                se = math.sqrt(variance) if variance > 0.0 else None
                z_value = estimate / se if se and se > 0.0 else None
                p_value = 2.0 * norm.sf(abs(z_value)) if z_value is not None else None
                beta_summary[name] = {
                    "beta": estimate,
                    "se": se,
                    "z": z_value,
                    "p_value": p_value,
                    "odds_ratio": math.exp(estimate),
                    "ci95_low": math.exp(estimate - 1.96 * se) if se else None,
                    "ci95_high": math.exp(estimate + 1.96 * se) if se else None,
                }
        except Exception:
            for idx, name in enumerate(self.coefficient_names):
                estimate = float(beta[idx])
                beta_summary[name] = {
                    "beta": estimate,
                    "se": None,
                    "z": None,
                    "p_value": None,
                    "odds_ratio": math.exp(estimate),
                    "ci95_low": None,
                    "ci95_high": None,
                }
        n = len(self.y)
        k = len(params)
        return {
            "success": bool(getattr(opt, "success", True)),
            "message": str(getattr(opt, "message", "")),
            "log_likelihood": float(ll),
            "aic": float((2.0 * k) - (2.0 * ll)),
            "bic": float((math.log(n) * k) - (2.0 * ll)),
            "n_parameters": int(k),
            "n_users": int(len(self.unique_users)),
            "n_observations": int(n),
            "thresholds_raw": [float(value) for value in thresholds],
            "sd_user_intercept": float(sd_user),
            "coefficients": beta_summary,
            "optimizer_iterations": getattr(opt, "nit", None),
            "optimizer_function_evals": getattr(opt, "nfev", None),
        }


class BinaryRandomInterceptLogitModel:
    def __init__(
        self,
        y: np.ndarray,
        x: np.ndarray,
        users: list[str],
        coefficient_names: list[str],
        quadrature_points: int = 25,
        maxiter: int = 300,
    ) -> None:
        self.y = y.astype(float)
        x_array = x.astype(float)
        intercept = np.ones((x_array.shape[0], 1), dtype=float)
        self.x = np.hstack([intercept, x_array])
        self.users = users
        self.coefficient_names = ["intercept", *coefficient_names]
        self.quadrature_points = quadrature_points
        self.maxiter = maxiter
        self.unique_users = list(dict.fromkeys(users))
        user_arr = np.array(users)
        self.group_slices = [np.where(user_arr == user)[0] for user in self.unique_users]
        nodes, weights = np.polynomial.hermite.hermgauss(quadrature_points)
        self.node = math.sqrt(2.0) * nodes
        self.node_log_weights = np.log(weights) - 0.5 * math.log(math.pi)

    def unpack_params(self, params: np.ndarray) -> tuple[np.ndarray, float]:
        beta = params[: self.x.shape[1]]
        sd_user = math.exp(float(params[-1]))
        return beta, sd_user

    def initial_params(self) -> np.ndarray:
        return np.concatenate([np.zeros(self.x.shape[1], dtype=float), np.array([math.log(0.8)], dtype=float)])

    def log_likelihood(self, params: np.ndarray) -> float:
        beta, sd_user = self.unpack_params(params)
        total = 0.0
        for idx in self.group_slices:
            eta_fixed = self.x[idx] @ beta
            eta = eta_fixed[None, :] + (sd_user * self.node)[:, None]
            logp = self.y[idx][None, :] * eta - np.logaddexp(0.0, eta)
            total += float(logsumexp(self.node_log_weights + np.sum(logp, axis=1)))
        return total

    def fit(self, active_predictor_idx: list[int] | None = None) -> tuple[object, np.ndarray]:
        if active_predictor_idx is None:
            objective = lambda p: -self.log_likelihood(p)
            opt = minimize(objective, self.initial_params(), method="L-BFGS-B", options={"maxiter": self.maxiter})
            opt.effective_n_parameters = len(opt.x)
            return opt, opt.x

        init = self.initial_params()
        active_predictor_idx = list(active_predictor_idx)
        active_beta_idx = [0, *[idx + 1 for idx in active_predictor_idx]]
        reduced_init = np.concatenate([init[np.array(active_beta_idx)], init[-1:]])

        def objective(reduced: np.ndarray) -> float:
            full = np.concatenate([np.zeros(self.x.shape[1], dtype=float), reduced[-1:]])
            beta_values = reduced[:-1]
            for pos, active_idx in enumerate(active_beta_idx):
                full[active_idx] = beta_values[pos]
            return -self.log_likelihood(full)

        opt = minimize(objective, reduced_init, method="L-BFGS-B", options={"maxiter": self.maxiter})
        full = np.concatenate([np.zeros(self.x.shape[1], dtype=float), opt.x[-1:]])
        for pos, active_idx in enumerate(active_beta_idx):
            full[active_idx] = opt.x[pos]
        opt.effective_n_parameters = len(reduced_init)
        return opt, full

    def summarize_fit(self, opt: object, params: np.ndarray) -> dict:
        ll = self.log_likelihood(params)
        beta, sd_user = self.unpack_params(params)
        objective = lambda p: -self.log_likelihood(p)
        coef_summary = {}
        try:
            hessian = finite_difference_hessian(objective, params)
            cov = np.linalg.inv(hessian)
            for idx, name in enumerate(self.coefficient_names):
                estimate = float(beta[idx])
                variance = float(cov[idx, idx])
                se = math.sqrt(variance) if variance > 0.0 else None
                z_value = estimate / se if se and se > 0.0 else None
                p_value = 2.0 * norm.sf(abs(z_value)) if z_value is not None else None
                coef_summary[name] = {
                    "beta": estimate,
                    "se": se,
                    "z": z_value,
                    "p_value": p_value,
                    "odds_ratio": math.exp(estimate),
                    "ci95_low": math.exp(estimate - 1.96 * se) if se else None,
                    "ci95_high": math.exp(estimate + 1.96 * se) if se else None,
                }
        except Exception:
            for idx, name in enumerate(self.coefficient_names):
                estimate = float(beta[idx])
                coef_summary[name] = {
                    "beta": estimate,
                    "se": None,
                    "z": None,
                    "p_value": None,
                    "odds_ratio": math.exp(estimate),
                    "ci95_low": None,
                    "ci95_high": None,
                }
        n = len(self.y)
        k = int(getattr(opt, "effective_n_parameters", len(params)))
        return {
            "success": bool(getattr(opt, "success", True)),
            "message": str(getattr(opt, "message", "")),
            "log_likelihood": float(ll),
            "aic": float((2.0 * k) - (2.0 * ll)),
            "bic": float((math.log(n) * k) - (2.0 * ll)),
            "n_parameters": int(k),
            "n_users": int(len(self.unique_users)),
            "n_observations": int(n),
            "sd_user_intercept": float(sd_user),
            "quadrature_points": int(self.quadrature_points),
            "coefficients": coef_summary,
            "optimizer_iterations": getattr(opt, "nit", None),
            "optimizer_function_evals": getattr(opt, "nfev", None),
        }


class LinearRandomInterceptModel:
    def __init__(
        self,
        y: np.ndarray,
        x: np.ndarray,
        users: list[str],
        coefficient_names: list[str],
        maxiter: int = 500,
    ) -> None:
        self.y = y.astype(float)
        self.x = x.astype(float)
        self.users = users
        self.coefficient_names = coefficient_names
        self.maxiter = maxiter
        self.unique_users = list(dict.fromkeys(users))
        user_arr = np.array(users)
        self.group_slices = [np.where(user_arr == user)[0] for user in self.unique_users]

    def _beta_and_loglik(self, log_vars: np.ndarray, x_use: np.ndarray | None = None) -> tuple[np.ndarray, float, float, float]:
        x_matrix = self.x if x_use is None else x_use
        sigma_eps = math.exp(float(log_vars[0]))
        sigma_user = math.exp(float(log_vars[1]))
        sigma_eps2 = sigma_eps * sigma_eps
        sigma_user2 = sigma_user * sigma_user
        xt_vinv_x = np.zeros((x_matrix.shape[1], x_matrix.shape[1]), dtype=float)
        xt_vinv_y = np.zeros(x_matrix.shape[1], dtype=float)
        log_det = 0.0
        for idx in self.group_slices:
            xg = x_matrix[idx, :]
            yg = self.y[idx]
            ng = len(idx)
            ones = np.ones(ng, dtype=float)
            coeff = sigma_user2 / (sigma_eps2 * (sigma_eps2 + ng * sigma_user2))
            vinv = (np.eye(ng) / sigma_eps2) - coeff * np.outer(ones, ones)
            xt_vinv_x += xg.T @ vinv @ xg
            xt_vinv_y += xg.T @ vinv @ yg
            log_det += (ng - 1) * math.log(sigma_eps2) + math.log(sigma_eps2 + ng * sigma_user2)
        beta = np.linalg.solve(xt_vinv_x, xt_vinv_y)
        quad = 0.0
        for idx in self.group_slices:
            xg = x_matrix[idx, :]
            yg = self.y[idx]
            ng = len(idx)
            resid = yg - xg @ beta
            ones = np.ones(ng, dtype=float)
            coeff = sigma_user2 / (sigma_eps2 * (sigma_eps2 + ng * sigma_user2))
            vinv = (np.eye(ng) / sigma_eps2) - coeff * np.outer(ones, ones)
            quad += float(resid.T @ vinv @ resid)
        n = len(self.y)
        loglik = -0.5 * (n * math.log(2.0 * math.pi) + log_det + quad)
        return beta, float(loglik), float(sigma_user), float(sigma_eps)

    def fit(self, active_predictor_idx: list[int] | None = None) -> tuple[object, dict]:
        if active_predictor_idx is None:
            x_use = self.x
            kept_names = self.coefficient_names
        else:
            x_use = self.x[:, active_predictor_idx]
            kept_names = [self.coefficient_names[idx] for idx in active_predictor_idx]

        def objective(log_vars: np.ndarray) -> float:
            try:
                _, ll, _, _ = self._beta_and_loglik(log_vars, x_use=x_use)
            except np.linalg.LinAlgError:
                return 1e18
            return -ll

        opt = minimize(objective, np.zeros(2, dtype=float), method="L-BFGS-B", options={"maxiter": self.maxiter})
        beta, ll, sigma_user, sigma_eps = self._beta_and_loglik(opt.x, x_use=x_use)

        sigma_eps2 = sigma_eps * sigma_eps
        sigma_user2 = sigma_user * sigma_user
        xt_vinv_x = np.zeros((x_use.shape[1], x_use.shape[1]), dtype=float)
        for idx in self.group_slices:
            xg = x_use[idx, :]
            ng = len(idx)
            ones = np.ones(ng, dtype=float)
            coeff = sigma_user2 / (sigma_eps2 * (sigma_eps2 + ng * sigma_user2))
            vinv = (np.eye(ng) / sigma_eps2) - coeff * np.outer(ones, ones)
            xt_vinv_x += xg.T @ vinv @ xg
        cov_beta = np.linalg.inv(xt_vinv_x)
        se = np.sqrt(np.diag(cov_beta))
        z_values = beta / se
        p_values = 2.0 * norm.sf(np.abs(z_values))

        coefficients = {
            name: {
                "beta": float(value),
                "se": float(se_value),
                "z": float(z_value),
                "p_value": float(p_value),
                "ci95_low": float(value - 1.96 * se_value),
                "ci95_high": float(value + 1.96 * se_value),
            }
            for name, value, se_value, z_value, p_value in zip(kept_names, beta, se, z_values, p_values)
        }
        full_coefficients = {
            name: coefficients.get(name, {"beta": 0.0, "se": None, "z": None, "p_value": None, "ci95_low": None, "ci95_high": None})
            for name in self.coefficient_names
        }
        k = len(beta) + 2
        result = {
            "success": bool(opt.success),
            "message": str(opt.message),
            "log_likelihood": float(ll),
            "aic": float((2.0 * k) - (2.0 * ll)),
            "bic": float((math.log(len(self.y)) * k) - (2.0 * ll)),
            "n_parameters": int(k),
            "n_users": int(len(self.unique_users)),
            "n_observations": int(len(self.y)),
            "coefficients": full_coefficients,
            "sd_user_intercept": float(sigma_user),
            "sd_residual": float(sigma_eps),
            "optimizer_iterations": getattr(opt, "nit", None),
        }
        return opt, result


def write_csv(path: Path, header: list[str], rows: list[list[object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        writer.writerows(rows)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def build_lr_test(full_log_likelihood: float, restricted_log_likelihood: float, df: int, hypothesis_label: str) -> dict:
    gap = float(full_log_likelihood - restricted_log_likelihood)
    warning = None
    if gap < -1e-6:
        warning = (
            f"Full model log-likelihood was below the restricted model for {hypothesis_label}; "
            "nested-model ordering failed numerically and the LR statistic was truncated at zero."
        )
    lr_stat = max(0.0, 2.0 * gap)
    return {
        "hypothesis_label": hypothesis_label,
        "full_log_likelihood": float(full_log_likelihood),
        "restricted_log_likelihood": float(restricted_log_likelihood),
        "log_likelihood_gap": gap,
        "lr_statistic": float(lr_stat),
        "df": int(df),
        "p_value": float(chi2.sf(lr_stat, df=df)),
        "warning": warning,
    }


def task_rows_to_csv_rows(rows: list[TaskObservation]) -> list[list[object]]:
    output = []
    for row in rows:
        output.append(
            [
                row.username,
                row.survey_id,
                row.cohort,
                row.cohort_code,
                row.last_survey_date,
                row.variant,
                row.graph,
                row.ai_error_exposed,
                row.step,
                row.task_order,
                row.video_id,
                row.trust_before,
                row.trust_after,
                row.trust_delta,
                row.delegation_raw,
                row.delegation_label,
                row.watch_time_s,
                row.replay_count,
                row.graph_click_count,
                row.graph_zoom_count,
                row.graph_drag_count,
                row.graph_total_interaction_count,
                row.graph_edit_count,
                row.graph_delete_count,
                row.graph_interactions_augmented,
            ]
        )
    return output
