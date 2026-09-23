from poverty_targeting.compare import BASELINE, choose, leaderboard, paired_gain


def result(aucs):
    return {"per_fold": {f: {"roc_auc": a} for f, a in zip([1, 2, 3, 4], aucs, strict=True)}}


def test_paired_gain_is_per_fold_difference():
    gain = paired_gain(result([0.82, 0.80, 0.85, 0.79]), result([0.80, 0.81, 0.84, 0.78]))
    assert gain["per_fold"] == {1: 0.02, 2: -0.01, 3: 0.01, 4: 0.01}
    assert gain["wins"] == 3


def test_small_or_inconsistent_gains_keep_the_baseline():
    base = result([0.80, 0.80, 0.80, 0.80])
    tiny = result([0.802, 0.801, 0.803, 0.799])  # 3 wins but mean gain < 0.005
    lucky = result([0.90, 0.79, 0.79, 0.79])  # big mean gain from one fold only
    assert choose({BASELINE: base, "tiny": tiny})[0] == BASELINE
    assert choose({BASELINE: base, "lucky": lucky})[0] == BASELINE


def test_consistent_gain_selects_the_challenger():
    base = result([0.80, 0.80, 0.80, 0.80])
    better = result([0.81, 0.81, 0.80, 0.82])
    chosen, gains = choose({BASELINE: base, "better": better})
    assert chosen == "better"
    assert gains["better"]["wins"] == 3


def test_leaderboard_marks_the_choice():
    rows = [
        {"tier": "A", "model": "logit", "roc_auc": 0.826, "pr_auc": 0.6, "ece": 0.015,
         "exclusion": 0.436, "gain": None, "chosen": True},
        {"tier": "A", "model": "lightgbm", "roc_auc": 0.825, "pr_auc": 0.59, "ece": 0.02,
         "exclusion": 0.429, "gain": {"mean": -0.001, "wins": 2}, "chosen": False},
    ]  # fmt: skip
    table = leaderboard(rows)
    assert "| A | logit | 0.826 |" in table
    assert "-0.0010 (2/4)" in table
    assert table.count("yes") == 1
