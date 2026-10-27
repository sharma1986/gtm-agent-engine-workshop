"""Unit tests for prospect scoring without live model calls."""

import os
import unittest
from unittest.mock import patch

with patch.dict(os.environ, {
    "OPENAI_API_KEY": "test-key",
    "LANGSMITH_TRACING": "false",
}):
    from gtm_agent import gtm_agent as agent


class ScoreProspectTests(unittest.TestCase):
    def setUp(self):
        self.enterContext(patch.dict(os.environ, {"LANGSMITH_TRACING": "false"}))
        self.scoring_llm = self.enterContext(patch.object(agent, "_scoring_llm"))
        self.inputs = {
            "prospect_profile": agent.build_prospect_profile.invoke({
                "prospect_id": "LEAD-71001",
            })["prospect_profile"],
            "offering": agent.lookup_offering.invoke({
                "offering_id": "OFFER-10005",
            })["offering"],
        }

    def test_overrides_conflicting_model_totals(self):
        scores = []
        for model_score in (85, 90, 12):
            with self.subTest(model_score=model_score):
                self.scoring_llm.invoke.return_value = agent.ProspectScore(
                    score=model_score,
                    justification="Revenue and segment fit; one required technology is missing.",
                    rubric_breakdown=agent.RubricBreakdown(
                        revenue_fit=100, tech_stack_match=75, segment_fit=100,
                    ),
                )
                result = agent.score_prospect.invoke(self.inputs)
                self.assertEqual(result["score"], 90.0)
                self.assertEqual(result["rubric_breakdown"]["tech_stack_match"], 75)
                scores.append(result["score"])
        self.assertEqual(scores, [90.0, 90.0, 90.0])

    def test_rounds_weighted_score_to_one_decimal(self):
        self.scoring_llm.invoke.return_value = agent.ProspectScore(
            justification="Partial fit across all components.",
            rubric_breakdown=agent.RubricBreakdown(
                revenue_fit=83.33, tech_stack_match=66.67, segment_fit=50.33,
            ),
        )
        result = agent.score_prospect.invoke(self.inputs)
        self.assertEqual(result["score"], 70.1)
        self.assertEqual(result["max_score"], 100)

    def test_full_rubric_returns_maximum_score(self):
        self.scoring_llm.invoke.return_value = agent.ProspectScore(
            score=1,
            justification="Full fit across all components.",
            rubric_breakdown=agent.RubricBreakdown(
                revenue_fit=100, tech_stack_match=100, segment_fit=100,
            ),
        )
        self.assertEqual(agent.score_prospect.invoke(self.inputs)["score"], 100.0)

    def test_invalid_offering_preserves_error(self):
        self.inputs["offering"] = None
        self.assertEqual(agent.score_prospect.invoke(self.inputs), {
            "score": None, "error": "Cannot score without a valid offering.",
        })


if __name__ == "__main__":
    unittest.main()
