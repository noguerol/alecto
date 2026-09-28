"""Unit tests for alecto.context — ContextGenerator and RetrievalTask."""


from alecto.context import (
    ContextConfig,
    ContextGenerator,
    Difficulty,
    GeneratedContext,
    Passage,
    RetrievalTask,
    RetrievalTaskGenerator,
    RetrievalTaskType,
)


class TestContextGenerator:
    def test_default_config(self):
        gen = ContextGenerator()
        assert gen.config.target_length == 4096
        assert gen.config.num_needles == 1
        assert gen.config.difficulty == Difficulty.EASY

    def test_generate_returns_generated_context(self):
        gen = ContextGenerator(ContextConfig(seed=42))
        result = gen.generate()
        assert isinstance(result, GeneratedContext)
        assert result.haystack is not None
        assert len(result.needles) == 1
        assert result.question is not None
        assert result.expected_answer is not None

    def test_deterministic_with_seed(self):
        gen1 = ContextGenerator(ContextConfig(seed=123))
        gen2 = ContextGenerator(ContextConfig(seed=123))
        r1 = gen1.generate()
        r2 = gen2.generate()
        assert r1.haystack == r2.haystack
        assert r1.needles == r2.needles

    def test_different_seeds_produce_different_output(self):
        gen1 = ContextGenerator(ContextConfig(seed=1))
        gen2 = ContextGenerator(ContextConfig(seed=2))
        r1 = gen1.generate()
        r2 = gen2.generate()
        assert r1.needles[0]["value"] != r2.needles[0]["value"]

    def test_configurable_length(self):
        gen = ContextGenerator(ContextConfig(target_length=2048, seed=10))
        result = gen.generate()
        assert result.actual_length > 0

    def test_configurable_topics(self):
        gen = ContextGenerator(ContextConfig(topics=["physics", "biology"], seed=5))
        result = gen.generate()
        assert result.needles[0]["topic"] in ["physics", "biology"]

    def test_multiple_needles(self):
        gen = ContextGenerator(ContextConfig(num_needles=3, seed=7))
        result = gen.generate()
        assert len(result.needles) == 3
        # All keys should be unique
        keys = [n["key"] for n in result.needles]
        assert len(set(keys)) == 3
        # All values should be unique
        values = [n["value"] for n in result.needles]
        assert len(set(values)) == 3

    def test_difficulty_levels(self):
        for difficulty in Difficulty:
            gen = ContextGenerator(ContextConfig(difficulty=difficulty, seed=1))
            result = gen.generate()
            assert result.config.difficulty == difficulty

    def test_nominal_positions(self):
        gen = ContextGenerator(ContextConfig(num_needles=5, seed=99))
        result = gen.generate()
        # 5 needles should use nominal positions 5%, 25%, 50%, 75%, 95%
        assert result.positions == [0.05, 0.25, 0.50, 0.75, 0.95]

    def test_single_needle_position(self):
        gen = ContextGenerator(ContextConfig(num_needles=1, seed=1))
        result = gen.generate()
        assert result.positions == [0.05]

    def test_question_mentions_key(self):
        gen = ContextGenerator(ContextConfig(num_needles=1, seed=42))
        result = gen.generate()
        assert result.needles[0]["key"] in result.question

    def test_expected_answer_is_value(self):
        gen = ContextGenerator(ContextConfig(num_needles=1, seed=42))
        result = gen.generate()
        assert result.expected_answer == result.needles[0]["value"]

    def test_haystack_contains_needle_record(self):
        gen = ContextGenerator(ContextConfig(num_needles=1, seed=42))
        result = gen.generate()
        needle = result.needles[0]
        assert needle["key"] in result.haystack
        assert needle["value"] in result.haystack


class TestRetrievalTask:
    def test_single_kv_task(self):
        gen = RetrievalTaskGenerator()
        task = gen.generate_single_kv(seed=1)
        assert isinstance(task, RetrievalTask)
        assert task.task_type == RetrievalTaskType.SINGLE_KV
        assert len(task.passages) == 1
        assert task.question is not None
        assert task.expected_answer is not None

    def test_multi_needle_task(self):
        gen = RetrievalTaskGenerator()
        task = gen.generate_multi_needle(num_needles=3, seed=2)
        assert task.task_type == RetrievalTaskType.MULTI_NEEDLE
        assert task.metadata["num_needles"] == 3

    def test_absent_key_task(self):
        gen = RetrievalTaskGenerator()
        task = gen.generate_absent_key(seed=3)
        assert task.task_type == RetrievalTaskType.ABSENT_KEY
        assert task.expected_answer == "ABSTAIN"
        assert "nonexistent_key_9999" in task.question

    def test_two_evidence_task(self):
        gen = RetrievalTaskGenerator()
        task = gen.generate_two_evidence(seed=4)
        assert task.task_type == RetrievalTaskType.TWO_EVIDENCE
        assert "+" in task.expected_answer

    def test_task_to_dict(self):
        gen = RetrievalTaskGenerator()
        task = gen.generate_single_kv(seed=5)
        d = task.to_dict()
        assert d["task_type"] == "single_kv"
        assert "passages" in d
        assert "question" in d
        assert "expected_answer" in d

    def test_task_to_json(self):
        import json
        gen = RetrievalTaskGenerator()
        task = gen.generate_single_kv(seed=6)
        j = task.to_json()
        parsed = json.loads(j)
        assert parsed["task_id"] == task.task_id

    def test_passage_fields(self):
        gen = RetrievalTaskGenerator()
        task = gen.generate_single_kv(seed=7)
        passage = task.passages[0]
        assert isinstance(passage, Passage)
        assert passage.id == "ctx"
        assert passage.text is not None
        assert passage.topic is not None

    def test_deterministic_tasks(self):
        gen = RetrievalTaskGenerator()
        t1 = gen.generate_single_kv(seed=99)
        t2 = gen.generate_single_kv(seed=99)
        assert t1.question == t2.question
        assert t1.expected_answer == t2.expected_answer
