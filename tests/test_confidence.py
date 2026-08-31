"""Tests for the confidence scoring system."""

from typer.testing import CliRunner

from polymind.client.cli.app import app

runner = CliRunner()


class TestDomainCommands:
    """Tests for domain management commands."""

    def test_domain_list(self, tmp_polymind, env_override):
        """domain list should show predefined domains."""
        result = runner.invoke(app, ["domain", "list"])
        assert result.exit_code == 0
        assert "Predefined Domains" in result.output
        assert "mathematics" in result.output
        assert "coding" in result.output
        assert "reasoning" in result.output

    def test_domain_show_predefined(self, tmp_polymind, env_override):
        """domain show should display domain details."""
        result = runner.invoke(app, ["domain", "show", "mathematics"])
        assert result.exit_code == 0
        assert "Mathematics" in result.output
        assert "Predefined" in result.output

    def test_domain_show_not_found(self, tmp_polymind, env_override):
        """domain show should fail for non-existent domain."""
        result = runner.invoke(app, ["domain", "show", "nonexistent"])
        assert result.exit_code != 0
        assert "not found" in result.output.lower()

    def test_domain_create_and_delete(self, tmp_polymind, env_override):
        """domain create and delete should work for custom domains."""
        result = runner.invoke(
            app,
            [
                "domain",
                "create",
                "test_domain",
                "--name",
                "Test Domain",
                "--description",
                "A test domain",
            ],
        )
        assert result.exit_code == 0
        assert "Created domain" in result.output

        # Verify it appears in list
        result = runner.invoke(app, ["domain", "list"])
        assert "test_domain" in result.output
        assert "Custom Domains" in result.output

        # Delete it
        result = runner.invoke(app, ["domain", "delete", "test_domain", "--force"])
        assert result.exit_code == 0
        assert "Deleted" in result.output

    def test_domain_cannot_delete_predefined(self, tmp_polymind, env_override):
        """domain delete should not work on predefined domains."""
        result = runner.invoke(app, ["domain", "delete", "mathematics", "--force"])
        assert result.exit_code != 0
        assert "cannot delete predefined" in result.output.lower()


class TestSuiteCommands:
    """Tests for suite management commands."""

    def test_suite_list_all(self, tmp_polymind, env_override):
        """suite list should show suites from all domains."""
        result = runner.invoke(app, ["suite", "list"])
        assert result.exit_code == 0
        assert "math_arithmetic" in result.output

    def test_suite_list_by_domain(self, tmp_polymind, env_override):
        """suite list <domain> should filter by domain."""
        result = runner.invoke(app, ["suite", "list", "mathematics"])
        assert result.exit_code == 0
        assert "math_arithmetic" in result.output

    def test_suite_show(self, tmp_polymind, env_override):
        """suite show should display question details."""
        result = runner.invoke(app, ["suite", "show", "math_arithmetic"])
        assert result.exit_code == 0
        assert "Basic Arithmetic" in result.output
        assert "Questions:  10" in result.output

    def test_suite_show_not_found(self, tmp_polymind, env_override):
        """suite show should fail for non-existent suite."""
        result = runner.invoke(app, ["suite", "show", "nonexistent_suite"])
        assert result.exit_code != 0
        assert "not found" in result.output.lower()


class TestConfidenceScorer:
    """Tests for the confidence scoring engine."""

    def test_exact_match(self):
        """Exact match should return 1.0 for matching answers."""
        from polymind.core.confidence.scorer import _exact_match
        from polymind.core.confidence.types import TestQuestion

        q = TestQuestion(id="t1", prompt="test", expected="42")
        result = _exact_match("42", q)
        assert result.score == 1.0

    def test_exact_match_partial(self):
        """Exact match should return 0.95 if expected is in response."""
        from polymind.core.confidence.scorer import _exact_match
        from polymind.core.confidence.types import TestQuestion

        q = TestQuestion(id="t1", prompt="test", expected="42")
        result = _exact_match("The answer is 42.", q)
        assert result.score == 0.95

    def test_exact_match_no_match(self):
        """Exact match should return 0.0 for non-matching answers."""
        from polymind.core.confidence.scorer import _exact_match
        from polymind.core.confidence.types import TestQuestion

        q = TestQuestion(id="t1", prompt="test", expected="42")
        result = _exact_match("100", q)
        assert result.score == 0.0

    def test_keyword_match_full(self):
        """Keyword match should return 1.0 when all keywords found."""
        from polymind.core.confidence.scorer import _keyword_match
        from polymind.core.confidence.types import TestQuestion

        q = TestQuestion(id="t1", prompt="test", expected="", keywords=["python", "code"])
        result = _keyword_match("I like python code", q)
        assert result.score == 1.0

    def test_keyword_match_partial(self):
        """Keyword match should return partial score."""
        from polymind.core.confidence.scorer import _keyword_match
        from polymind.core.confidence.types import TestQuestion

        q = TestQuestion(id="t1", prompt="test", expected="", keywords=["python", "code", "fun"])
        result = _keyword_match("I like python", q)
        assert 0.0 < result.score < 1.0

    def test_keyword_match_none(self):
        """Keyword match should return 0.0 when no keywords found."""
        from polymind.core.confidence.scorer import _keyword_match
        from polymind.core.confidence.types import TestQuestion

        q = TestQuestion(id="t1", prompt="test", expected="", keywords=["xyz"])
        result = _keyword_match("hello world", q)
        assert result.score == 0.0

    def test_hybrid_picks_better_score(self):
        """Hybrid should use all methods and pick the best score."""
        from polymind.core.confidence.scorer import _hybrid
        from polymind.core.confidence.types import TestQuestion

        q = TestQuestion(
            id="t1",
            prompt="test",
            expected="42",
            keywords=["fourty-two", "answer"],
        )
        # Exact match wins
        result = _hybrid("42", q)
        assert result.score == 1.0

    def test_hybrid_uses_all_methods(self):
        """Hybrid should run all methods and combine results."""
        from polymind.core.confidence.scorer import _hybrid
        from polymind.core.confidence.types import TestQuestion

        q = TestQuestion(
            id="t1",
            prompt="What is 2+2?",
            expected="4",
            keywords=["4", "four"],
            code="assert 2 + 2 == 4",
        )
        result = _hybrid("4", q)
        # Code passes → score should be 1.0
        assert result.score == 1.0
        assert "hybrid-code" in result.method_used

    def test_hybrid_code_pass_overrides(self):
        """Hybrid with passing code should always score 1.0."""
        from polymind.core.confidence.scorer import _hybrid
        from polymind.core.confidence.types import TestQuestion

        q = TestQuestion(
            id="t1",
            prompt="test",
            expected="something else",
            keywords=["wrong"],
            code="assert True  # always passes",
        )
        result = _hybrid("anything", q)
        assert result.score == 1.0

    def test_hybrid_consensus_bonus(self):
        """Hybrid should boost score when multiple methods agree."""
        from polymind.core.confidence.scorer import _hybrid
        from polymind.core.confidence.types import TestQuestion

        q = TestQuestion(
            id="t1",
            prompt="test",
            expected="Paris",
            keywords=["Paris", "capital", "France"],
        )
        # All methods should agree this is correct
        result = _hybrid("The capital of France is Paris.", q)
        assert result.score >= 0.85

    def test_code_execution_pass(self):
        """Code execution should pass for correct code."""
        from polymind.core.confidence.scorer import _code_execution
        from polymind.core.confidence.types import TestQuestion

        q = TestQuestion(
            id="t1",
            prompt="test",
            expected="120",
            code="assert 5 * 4 * 3 * 2 * 1 == 120",
        )
        result = _code_execution("120", q)
        assert result.score == 1.0

    def test_code_execution_fail(self):
        """Code execution should fail for incorrect code."""
        from polymind.core.confidence.scorer import _code_execution
        from polymind.core.confidence.types import TestQuestion

        q = TestQuestion(
            id="t1",
            prompt="test",
            expected="120",
            code="assert 5 * 4 * 3 * 2 * 1 == 999",
        )
        result = _code_execution("999", q)
        assert result.score == 0.0

    def test_score_suite(self):
        """score_suite should aggregate question scores."""
        from polymind.core.confidence.scorer import score_suite
        from polymind.core.confidence.types import TestQuestion, TestSuite

        suite = TestSuite(
            id="test_suite",
            name="Test",
            description="Test suite",
            difficulty="easy",
            questions=[
                TestQuestion(id="q1", prompt="1+1?", expected="2", keywords=["2"]),
                TestQuestion(id="q2", prompt="2+2?", expected="4", keywords=["4"]),
            ],
        )

        responses = {"q1": "2", "q2": "4"}
        result = score_suite(responses, suite)
        assert result.score == 100.0
        assert result.passed == 2
        assert result.total == 2


class TestConfidenceArtifact:
    """Tests for confidence artifact persistence."""

    def test_save_and_load(self, tmp_polymind, env_override):
        """Save and load should round-trip confidence data."""
        from polymind.core.confidence.artifact import load_confidence, save_confidence
        from polymind.core.confidence.types import (
            DomainScore,
            ModelConfidence,
            SuiteScore,
        )

        scores = {
            "1": ModelConfidence(
                model_id="1",
                domains={
                    "math": DomainScore(
                        domain_id="math",
                        overall=85.5,
                        suites={
                            "basic": SuiteScore(
                                suite_id="basic",
                                score=90.0,
                                passed=9,
                                total=10,
                            )
                        },
                    )
                },
                best_domain="math",
                overall_score=85.5,
            )
        }

        save_confidence(scores)
        loaded = load_confidence()

        assert "1" in loaded
        assert loaded["1"].overall_score == 85.5
        assert loaded["1"].best_domain == "math"
        assert "math" in loaded["1"].domains
        assert loaded["1"].domains["math"].overall == 85.5
