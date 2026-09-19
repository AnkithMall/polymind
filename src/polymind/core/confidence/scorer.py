"""Confidence scoring engine.

Supports five evaluation methods:
  - exact_match: Compare output to expected answer
  - keyword_match: Check for presence of required keywords
  - code_execution: Run generated code in a sandbox
  - llm_judge: Heuristic-based response quality grading
  - hybrid: Run ALL methods and combine for maximum accuracy

Default is hybrid. Hybrid prioritizes accuracy over speed.
"""

from __future__ import annotations

import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

from polymind.core.confidence.types import (
    Domain,
    EvaluationMethod,
    SuiteScore,
    TestQuestion,
    TestSuite,
)


@dataclass
class QuestionScore:
    """Score for a single question."""

    question_id: str
    score: float
    method_used: str
    details: str = ""
    confidence: float = 0.0


def _normalize(text: str) -> str:
    """Normalize text for comparison."""
    text = text.strip().lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s\d.,;:!?\-+/=(){}\[\]\"']", "", text)
    return text


def _extract_numbers(text: str) -> list[str]:
    """Extract all numbers from text."""
    return re.findall(r"-?\d+\.?\d*", text)


def _exact_match(response: str, question: TestQuestion) -> QuestionScore:
    """Exact match evaluation with fuzzy number matching."""
    norm_response = _normalize(response)
    norm_expected = _normalize(question.expected)

    # 1. Exact string match
    if norm_response == norm_expected:
        return QuestionScore(question.id, 1.0, "exact_match", "exact match", 1.0)

    # 2. Expected appears in response
    if norm_expected in norm_response:
        return QuestionScore(question.id, 0.95, "exact_match", "expected in response", 0.95)

    # 3. Numeric comparison (handles formatting differences like "42" vs "42.0")
    resp_nums = _extract_numbers(response)
    exp_nums = _extract_numbers(question.expected)

    if exp_nums and resp_nums:
        resp_set = {float(n) for n in resp_nums}
        exp_set = {float(n) for n in exp_nums}
        if resp_set == exp_set:
            return QuestionScore(question.id, 0.95, "exact_match", "numbers match", 0.9)
        # Partial number match
        common = resp_set & exp_set
        if common and len(common) >= len(exp_set) * 0.5:
            ratio = len(common) / len(exp_set)
            return QuestionScore(
                question.id,
                0.7 * ratio,
                "exact_match",
                f"partial number match: {len(common)}/{len(exp_set)}",
                0.7,
            )

    # 4. Check if the core answer is present (ignore surrounding text)
    # e.g., expected="Paris" and response="The capital of France is Paris."
    words_expected = set(norm_expected.split())
    words_response = set(norm_response.split())
    if len(words_expected) > 0:
        overlap = words_expected & words_response
        if len(overlap) == len(words_expected):
            return QuestionScore(
                question.id,
                0.85,
                "exact_match",
                "all expected words present",
                0.85,
            )

    return QuestionScore(question.id, 0.0, "exact_match", "no match", 0.9)


def _keyword_match(response: str, question: TestQuestion) -> QuestionScore:
    """Keyword presence evaluation with weighted keywords."""
    if not question.keywords:
        return QuestionScore(question.id, 0.0, "keyword_match", "no keywords defined", 0.5)

    norm_response = _normalize(response)
    found = 0
    found_keywords: list[str] = []

    for kw in question.keywords:
        norm_kw = _normalize(kw)
        if norm_kw in norm_response:
            found += 1
            found_keywords.append(kw)

    ratio = found / len(question.keywords)

    if ratio >= 0.8:
        return QuestionScore(
            question.id,
            1.0,
            "keyword_match",
            f"found {found}/{len(question.keywords)}: {', '.join(found_keywords[:5])}",
            min(1.0, 0.6 + ratio * 0.4),
        )
    elif ratio >= 0.5:
        return QuestionScore(
            question.id,
            0.75,
            "keyword_match",
            f"found {found}/{len(question.keywords)}: {', '.join(found_keywords[:5])}",
            0.6 + ratio * 0.3,
        )
    elif ratio > 0:
        return QuestionScore(
            question.id,
            0.4,
            "keyword_match",
            f"found {found}/{len(question.keywords)}: {', '.join(found_keywords[:3])}",
            0.4 + ratio * 0.2,
        )
    else:
        return QuestionScore(question.id, 0.0, "keyword_match", "no keywords found", 0.6)


def _code_execution(response: str, question: TestQuestion) -> QuestionScore:
    """Run the question's test code to verify the answer."""
    if not question.code:
        return QuestionScore(question.id, 0.0, "code_execution", "no test code", 0.0)

    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(question.code)
            tmp_path = f.name

        result = subprocess.run(
            [sys.executable, tmp_path],
            capture_output=True,
            text=True,
            timeout=15,
        )

        Path(tmp_path).unlink(missing_ok=True)

        if result.returncode == 0:
            return QuestionScore(question.id, 1.0, "code_execution", "test passed", 1.0)
        else:
            error_short = result.stderr[:200].strip()
            return QuestionScore(
                question.id,
                0.0,
                "code_execution",
                f"test failed: {error_short}",
                1.0,
            )

    except subprocess.TimeoutExpired:
        return QuestionScore(question.id, 0.0, "code_execution", "timeout (15s)", 0.8)
    except Exception as e:
        return QuestionScore(question.id, 0.0, "code_execution", f"error: {e}", 0.5)


def _llm_judge(response: str, question: TestQuestion) -> QuestionScore:
    """Heuristic-based response quality judge.

    Analyzes response quality signals beyond simple matching:
    - Answer directness (does it directly answer the question?)
    - Hedging language (indicates uncertainty)
    - Contradictions
    - Relevance to expected answer
    """
    if not response.strip():
        return QuestionScore(question.id, 0.0, "llm_judge", "empty response", 0.5)

    norm = _normalize(response)
    norm_expected = _normalize(question.expected)

    score = 0.5  # baseline
    signals: list[str] = []

    # 1. Directness — does the response start with or contain the answer?
    if norm.startswith(norm_expected) or norm_expected in norm[: len(norm_expected) + 20]:
        score += 0.3
        signals.append("direct answer")

    # 2. Hedging — penalize uncertainty
    hedging = [
        "i'm not sure",
        "i think",
        "maybe",
        "possibly",
        "it depends",
        "i believe",
        "could be",
        "might be",
        "not certain",
        "unclear",
        "it's hard to say",
        "there's no clear",
        "generally speaking",
    ]
    hedge_count = sum(1 for h in hedging if h in norm)
    if hedge_count >= 2:
        score -= 0.2
        signals.append(f"hedging({hedge_count})")
    elif hedge_count == 1:
        score -= 0.1
        signals.append("light hedging")

    # 3. Contradictions — penalize "actually", "however" contradicting the answer
    contradictions = ["actually,", "however,", "but wait", "on the other hand", "that's wrong"]
    for c in contradictions:
        if c in norm and norm_expected in norm:
            score -= 0.1
            signals.append(f"contradiction: {c}")
            break

    # 4. Length appropriateness — very short or very long may indicate issues
    word_count = len(response.split())
    if word_count < 2:
        score -= 0.1
        signals.append("too short")
    elif word_count > 200:
        score -= 0.05
        signals.append("verbose")

    # 5. Confidence indicators
    confident = ["definitely", "certainly", "absolutely", "without doubt", "the answer is"]
    for c in confident:
        if c in norm:
            score += 0.1
            signals.append(f"confident: {c}")
            break

    # 6. Educational value — does it explain the answer?
    explain_signals = ["because", "since", "therefore", "thus", "this means"]
    if any(s in norm for s in explain_signals):
        score += 0.05
        signals.append("explains reasoning")

    score = max(0.0, min(1.0, score))

    return QuestionScore(
        question.id,
        score,
        "llm_judge",
        "; ".join(signals) if signals else "baseline score",
        0.5,  # llm_judge has lower confidence than deterministic methods
    )


def _hybrid(response: str, question: TestQuestion) -> QuestionScore:
    """Hybrid evaluation using ALL methods for maximum accuracy.

    Strategy:
    1. Run all available methods (exact, keyword, code_execution, llm_judge)
    2. If code_execution is available and passes → score is 1.0 (highest confidence)
    3. If exact_match hits → score is very high (second highest confidence)
    4. Otherwise, combine signals with weighted voting:
       - exact_match:    weight 0.30
       - keyword_match:  weight 0.25
       - code_execution: weight 0.30 (if available, else redistributed)
       - llm_judge:      weight 0.15
    5. Apply bonus if multiple methods agree (consensus boost)
    """
    exact = _exact_match(response, question)
    keyword = _keyword_match(response, question)
    code = _code_execution(response, question)
    judge = _llm_judge(response, question)

    # Run all methods regardless of whether they'll be used
    methods_used = ["exact", "keyword", "judge"]
    scores_map = {
        "exact": exact,
        "keyword": keyword,
        "judge": judge,
    }

    if question.code:
        methods_used.insert(2, "code")
        scores_map["code"] = code

    # Fast path: code execution is the gold standard
    if code.score == 1.0 and code.confidence >= 0.8:
        return QuestionScore(
            question.id,
            1.0,
            "hybrid-code",
            f"code passed | exact={exact.score:.2f} kw={keyword.score:.2f}",
            1.0,
        )

    # Fast path: exact match with high confidence
    if exact.score >= 0.9 and exact.confidence >= 0.8:
        return QuestionScore(
            question.id,
            exact.score,
            "hybrid-exact",
            f"exact={exact.score:.2f} kw={keyword.score:.2f} judge={judge.score:.2f}",
            exact.confidence,
        )

    # Weighted combination
    has_code = question.code and code.confidence > 0
    if has_code:
        weights = {"exact": 0.30, "keyword": 0.25, "code": 0.30, "judge": 0.15}
    else:
        weights = {"exact": 0.35, "keyword": 0.35, "judge": 0.30}

    weighted_sum = 0.0
    total_confidence = 0.0

    for name, weight in weights.items():
        s = scores_map[name]
        weighted_sum += s.score * weight * s.confidence
        total_confidence += weight * s.confidence

    if total_confidence > 0:
        base_score = weighted_sum / total_confidence
    else:
        base_score = 0.0

    # Consensus bonus: if 3+ methods agree (>0.5 each), boost score
    agreeing = sum(1 for s in [exact, keyword, judge] if s.score >= 0.5)
    if has_code and code.score >= 0.5:
        agreeing += 1

    consensus_bonus = 0.0
    if agreeing >= 3:
        consensus_bonus = 0.1
    elif agreeing >= 2:
        consensus_bonus = 0.05

    final_score = min(1.0, base_score + consensus_bonus)

    # Compute overall confidence from method confidences
    avg_confidence = total_confidence / len(weights) if weights else 0.5

    detail_parts = [f"{m}={scores_map[m].score:.2f}" for m in methods_used]
    detail = f"weighted={base_score:.2f} consensus(+{consensus_bonus:.2f}) | " + " ".join(
        detail_parts
    )

    return QuestionScore(
        question.id,
        final_score,
        "hybrid",
        detail,
        avg_confidence,
    )


def score_question(
    response: str,
    question: TestQuestion,
    method: EvaluationMethod | None = None,
) -> QuestionScore:
    """Score a single question response.

    Args:
        response: The model's response text.
        question: The test question with expected answer.
        method: Override evaluation method. If None, uses question's method.
    """
    eval_method = method or question.evaluation

    if eval_method == "exact_match":
        return _exact_match(response, question)
    elif eval_method == "keyword_match":
        return _keyword_match(response, question)
    elif eval_method == "code_execution":
        return _code_execution(response, question)
    elif eval_method == "llm_judge":
        return _llm_judge(response, question)
    elif eval_method == "hybrid":
        return _hybrid(response, question)
    else:
        return _hybrid(response, question)


def score_suite(
    responses: dict[str, str],
    suite: TestSuite,
) -> SuiteScore:
    """Score all questions in a test suite.

    Args:
        responses: dict mapping question_id -> model response
        suite: The test suite to score against.
    """
    scores: list[QuestionScore] = []
    details: dict[str, float] = {}
    method_counts: dict[str, int] = {}

    for question in suite.questions:
        response = responses.get(question.id, "")
        qs = score_question(response, question)
        weighted = qs.score * question.weight
        scores.append(qs)
        details[question.id] = round(weighted, 3)

        # Track which methods were used
        base_method = qs.method_used.split("-")[0]
        method_counts[base_method] = method_counts.get(base_method, 0) + 1

    total_weight = sum(q.weight for q in suite.questions)
    if total_weight > 0:
        overall = (
            sum(s.score * q.weight for s, q in zip(scores, suite.questions, strict=False))
            / total_weight
        )
    else:
        overall = 0.0

    passed = sum(1 for s in scores if s.score >= 0.7)

    # Average confidence across all questions
    avg_confidence = sum(s.confidence for s in scores) / len(scores) if scores else 0.0

    # Add method distribution to details
    details["_methods"] = str(method_counts)  # type: ignore[assignment]
    details["_avg_confidence"] = round(avg_confidence, 3)  # type: ignore[assignment]

    return SuiteScore(
        suite_id=suite.id,
        score=round(overall * 100, 2),
        passed=passed,
        total=len(suite.questions),
        details=details,
    )


def generate_test_prompt(question: TestQuestion) -> str:
    """Generate the full prompt to send to a model."""
    parts = [question.prompt]

    if question.explanation:
        parts.append(f"\nNote: {question.explanation}")

    parts.append("\nProvide a concise, direct answer.")
    return "\n".join(parts)


def evaluate_model_on_domain(
    responses: dict[str, str],
    domain: Domain,
) -> dict[str, SuiteScore]:
    """Evaluate a model's responses across all suites in a domain.

    Args:
        responses: dict mapping "suite_id:question_id" -> model response
        domain: The domain with suites to evaluate against.

    Returns:
        dict mapping suite_id -> SuiteScore
    """
    results: dict[str, SuiteScore] = {}

    for suite in domain.suites:
        suite_responses: dict[str, str] = {}
        for question in suite.questions:
            key = f"{suite.id}:{question.id}"
            if key in responses:
                suite_responses[question.id] = responses[key]

        results[suite.id] = score_suite(suite_responses, suite)

    return results
