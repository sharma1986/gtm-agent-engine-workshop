import importlib
import json
import unittest
from unittest.mock import patch

from gtm_agent import data_service


class UpdateProspectInfoTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            cls.agent = importlib.import_module("gtm_agent.gtm_agent")

    def setUp(self):
        self.prospect_id = "TEST-PROSPECT"
        self.record = {
            "tech_stack": ["AWS"],
            "engagement_history": [],
            "account_details": [],
            "annual_revenue": 1000000,
        }
        self.enterContext(patch.dict(data_service.PROSPECTS, {self.prospect_id: self.record}, clear=True))
        self.enterContext(patch.dict(data_service._PROFILES, {}, clear=True))
        self.enterContext(patch.dict("os.environ", {"LANGSMITH_TRACING": "false"}))

    def test_update_persists_technology(self):
        result = data_service.update_prospect_info(self.prospect_id, "Kubernetes")

        self.assertEqual(result, {
            "updated": True,
            "found": True,
            "tech_stack": ["AWS", "Kubernetes"],
        })
        self.assertEqual(data_service.fetch_tech_stack(self.prospect_id), ["AWS", "Kubernetes"])

    def test_update_rebuilds_cached_profile(self):
        original = self.agent.build_prospect_profile.invoke({"prospect_id": self.prospect_id})
        self.assertEqual(original["prospect_profile"]["tech_stack"], ["AWS"])

        data_service.update_prospect_info(self.prospect_id, "Kubernetes")
        rebuilt = self.agent.build_prospect_profile.invoke({"prospect_id": self.prospect_id})

        self.assertTrue(rebuilt["found"])
        self.assertEqual(rebuilt["prospect_profile"]["tech_stack"], ["AWS", "Kubernetes"])

    def test_scoring_observes_update_even_with_old_profile(self):
        profile = self.agent.build_prospect_profile.invoke({"prospect_id": self.prospect_id})["prospect_profile"]
        offering = {
            "description": "Cloud platform",
            "min_annual_revenue": 1000000,
            "required_tech_stack": ["AWS", "Kubernetes"],
        }

        def score_saved_stack(messages):
            saved_profile = json.loads(messages[1]["content"].split("Prospect profile:\n", 1)[1])
            self.assertEqual(saved_profile["tech_stack"], ["AWS", "Kubernetes"])
            return self.agent.ProspectScore(
                score=100,
                justification="AWS and Kubernetes are present; no required technologies are missing.",
                rubric_breakdown=self.agent.RubricBreakdown(
                    revenue_fit=100, tech_stack_match=100, segment_fit=100,
                ),
            )

        self.agent.update_prospect_info.invoke({
            "prospect_id": self.prospect_id, "technology": "Kubernetes",
        })
        with patch.object(self.agent, "_scoring_llm") as scoring_llm:
            scoring_llm.invoke.side_effect = score_saved_stack
            result = self.agent.score_prospect.invoke({"prospect_profile": profile, "offering": offering})

        self.assertEqual(result["rubric_breakdown"]["tech_stack_match"], 100)
        self.assertIn("AWS and Kubernetes are present", result["justification"])

    def test_update_existing_technology_does_not_duplicate_it(self):
        data_service.update_prospect_info(self.prospect_id, "AWS")

        self.assertEqual(data_service.fetch_tech_stack(self.prospect_id), ["AWS"])

    def test_missing_prospect_is_not_updated(self):
        self.assertEqual(data_service.update_prospect_info("MISSING", "Kubernetes"), {
            "updated": False, "found": False,
        })
        self.assertIsNone(data_service.get_prospect_record("MISSING"))


if __name__ == "__main__":
    unittest.main()
