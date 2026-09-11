import tempfile
import unittest
from pathlib import Path

from fyp.data import ROOT, normalize_question
from fyp.prepare import build


class PreparationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.clean = build(ROOT / "data/raw", ROOT / "prompts", "question_disjoint_v1")
        cls.historical = build(ROOT / "data/raw", ROOT / "prompts", "historical_candidates")

    def test_leaked_real_questions_are_excluded_before_token_filter(self):
        train, evaluation, manifest, dispositions = self.clean
        ids = {row["question_id"] for row in train}
        self.assertFalse(ids & {row["question_id"] for row in evaluation})
        self.assertEqual(manifest["training_by_category"], {"arithmetic_only": 309, "cot_and_code": 127})
        self.assertEqual(manifest["training_dispositions"]["evaluation_question_overlap"], 114)
        self.assertFalse(manifest["token_filter_applied"])

    def test_known_bad_reference_rows_quarantined_without_changing_raw(self):
        evaluation, manifest = self.clean[1:3]
        math_rows = {r["source_row"] for r in evaluation if r["category"] == "cot_and_code"}
        self.assertNotIn(15, math_rows)
        self.assertNotIn(43, math_rows)
        self.assertEqual(manifest["evaluation_by_category"], {"arithmetic_only": 100, "cot_and_code": 87})
        self.assertTrue(all("generated_solution" not in r and "messages" not in r for r in evaluation))

    def test_historical_candidates_remain_explicitly_contaminated(self):
        train, evaluation, manifest, _ = self.historical
        self.assertEqual(len(train), 560)
        self.assertEqual([r["source_row"] for r in evaluation], list(range(1, 21)))
        self.assertEqual(manifest["train_eval_normalized_question_overlap"], 20)

    def test_marker_case_and_whitespace_are_normalized(self):
        self.assertEqual(normalize_question(" $Calculate#  the\nSUM "), normalize_question("Calculate the sum"))


if __name__ == "__main__":
    unittest.main()
