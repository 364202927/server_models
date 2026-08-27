from ai.benchmark.checks import evaluate_case


def test_generic_checks() -> None:
    score, _ = evaluate_case('{"signal": "buy"}', {"checks": [{"type": "json"}]})
    assert score == 100
