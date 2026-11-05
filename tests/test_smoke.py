def test_ci_pipeline_runs_tests() -> None:
    """Protects the CI pipeline itself.

    pytest exits with an error when it collects zero tests, so an empty tests
    folder would turn CI red. This placeholder keeps CI meaningful until the
    first real test arrives, then we delete it.
    """
    assert True
