"""Unit tests for LLM-based judging."""


from alecto.judging import JudgeCriteria, JudgeModel, JudgeResult


class TestJudgeCriteria:
    def test_default_criteria(self):
        criteria = JudgeCriteria.default_criteria()
        assert len(criteria) == 5
        names = [c.name for c in criteria]
        assert "accuracy" in names
        assert "completeness" in names
        assert "relevance" in names
        assert "safety" in names
        assert "clarity" in names

    def test_weights(self):
        criteria = JudgeCriteria.default_criteria()
        by_name = {c.name: c for c in criteria}
        assert by_name["accuracy"].weight == 1.0
        assert by_name["completeness"].weight == 0.8
        assert by_name["relevance"].weight == 0.7
        assert by_name["safety"].weight == 1.0
        assert by_name["clarity"].weight == 0.5

    def test_to_dict(self):
        c = JudgeCriteria(name="test", weight=0.5, description="desc", min_score=0.1)
        d = c.to_dict()
        assert d["name"] == "test"
        assert d["weight"] == 0.5
        assert d["description"] == "desc"
        assert d["min_score"] == 0.1


class TestJudgeResult:
    def test_from_scores(self):
        criteria = JudgeCriteria.default_criteria()
        scores = {
            "accuracy": 0.8,
            "completeness": 0.6,
            "relevance": 0.7,
            "safety": 1.0,
            "clarity": 0.9,
        }
        result = JudgeResult.from_scores(scores, criteria, target="test-model", threshold=0.5)
        assert result.target == "test-model"
        assert result.scores == scores
        assert result.passed is True
        assert len(result.reasons) == 5

    def test_from_scores_below_threshold(self):
        criteria = JudgeCriteria.default_criteria()
        scores = {
            "accuracy": 0.2,
            "completeness": 0.1,
            "relevance": 0.3,
            "safety": 0.5,
            "clarity": 0.4,
        }
        result = JudgeResult.from_scores(scores, criteria, target="test-model", threshold=0.8)
        assert result.passed is False

    def test_to_dict(self):
        criteria = JudgeCriteria.default_criteria()
        scores = {"accuracy": 0.8, "completeness": 0.7, "relevance": 0.6, "safety": 1.0, "clarity": 0.9}
        result = JudgeResult.from_scores(scores, criteria, target="m", threshold=0.5)
        d = result.to_dict()
        assert d["target"] == "m"
        assert d["scores"] == scores
        assert d["passed"] is True


class TestJudgeModel:
    def setup_method(self):
        self.judge = JudgeModel("test-model")

    def test_judge_basic(self):
        result = self.judge.judge("What is the capital of France?", "The capital of France is Paris.")
        assert result.target == "test-model"
        assert result.scores["safety"] == 1.0
        assert result.scores["clarity"] == 1.0

    def test_accuracy_overlap(self):
        result = self.judge.judge("Explain quantum computing and entanglement", "Quantum computing uses entanglement to process information.")
        assert result.scores["accuracy"] > 0.0

    def test_accuracy_no_overlap(self):
        result = self.judge.judge("Explain quantum computing", "The weather is nice today.")
        assert result.scores["accuracy"] == 0.0

    def test_safety_flagged(self):
        result = self.judge.judge("prompt", "You should kill the process.")
        assert result.scores["safety"] == 0.0

    def test_safety_clean(self):
        result = self.judge.judge("prompt", "Here is a helpful answer.")
        assert result.scores["safety"] == 1.0

    def test_clarity_no_sentence(self):
        result = self.judge.judge("prompt", "no punctuation here")
        assert result.scores["clarity"] == 0.5

    def test_clarity_empty(self):
        result = self.judge.judge("prompt", "")
        assert result.scores["clarity"] == 0.5

    def test_batch_judge(self):
        pairs = [
            ("What is 2+2?", "The answer is 4."),
            ("Tell me about cats", "Cats are mammals."),
        ]
        results = self.judge.batch_judge(pairs)
        assert len(results) == 2
        assert all(r.target == "test-model" for r in results)

    def test_aggregate(self):
        pairs = [
            ("What is 2+2?", "The answer is 4."),
            ("Tell me about cats", "Cats are mammals."),
        ]
        results = self.judge.batch_judge(pairs)
        agg = self.judge.aggregate(results)
        assert agg["n_results"] == 2
        assert 0.0 <= agg["mean_weighted_score"] <= 1.0
        assert 0.0 <= agg["pass_rate"] <= 1.0
        assert "accuracy" in agg["per_criterion_means"]

    def test_aggregate_empty(self):
        agg = self.judge.aggregate([])
        assert agg["n_results"] == 0
        assert agg["mean_weighted_score"] == 0.0

    def test_custom_criteria(self):
        criteria = [JudgeCriteria(name="accuracy", weight=1.0)]
        judge = JudgeModel("custom", criteria=criteria, threshold=0.5)
        result = judge.judge("Explain gravity", "Gravity is a force.")
        assert "accuracy" in result.scores
        assert "clarity" not in result.scores

    def test_completeness(self):
        result = self.judge.judge("Explain quantum computing and entanglement and superposition", "Quantum computing and entanglement are key concepts.")
        assert 0.0 <= result.scores["completeness"] <= 1.0

    def test_relevance(self):
        result = self.judge.judge("Explain quantum computing", "Quantum computing is fascinating.")
        assert result.scores["relevance"] > 0.0
