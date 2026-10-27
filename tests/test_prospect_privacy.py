import importlib
import json
import os
import unittest
from copy import deepcopy
from unittest.mock import patch

with patch.dict(os.environ, {"OPENAI_API_KEY": "test", "LANGSMITH_TRACING": "false"}):
    agent = importlib.import_module("gtm_agent.gtm_agent")


class ProspectPrivacyTests(unittest.TestCase):
    def setUp(self):
        self.prospect_id = "LEAD-TEST"
        self.record = {
            "name": "Test Prospect",
            "email": "prospect@example.com",
            "disqualified": False,
            "annual_revenue": 70000000,
            "engagement_history": [{"channel": "Webinar"}],
            "account_details": [{"industry": "Software"}],
            "tech_stack": ["AWS"],
            "enrichment_source": "Test source",
            "billing_qualification": {
                "tax_id": "private-tax-id",
                "card_on_file": "private-card",
                "date_of_birth": "private-birth-date",
                "credit_check_ref": "private-credit-ref",
            },
        }
        self.offering = {
            "required_tech_stack": ["AWS"],
            "min_annual_revenue": 1000000,
            "description": "Cloud platform",
        }
        self.score = agent.ProspectScore(
            score=80,
            justification="Revenue and technology fit",
            rubric_breakdown=agent.RubricBreakdown(
                revenue_fit=90, tech_stack_match=100, segment_fit=50,
            ),
        )
        self.enterContext(patch.dict(os.environ, {"LANGSMITH_TRACING": "false"}))
        self.enterContext(patch.dict(agent.data_service.PROSPECTS, {self.prospect_id: self.record}))
        self.enterContext(patch.dict(agent.data_service._PROFILES, {}, clear=True))
        self.model = self.enterContext(patch.object(agent, "_scoring_llm"))
        self.model.invoke.return_value = self.score

    def assert_private_data_absent(self, value):
        serialized = json.dumps(value)
        for private in ("billing_qualification", "private-tax-id", "private-card",
                        "private-birth-date", "private-credit-ref"):
            self.assertNotIn(private, serialized)

    def scoring_profile(self):
        return {"prospect_id": self.prospect_id, **{
            k: v for k, v in self.record.items() if k != "billing_qualification"
        }}

    def test_contact_returns_only_approved_fields(self):
        result = agent.get_prospect.invoke({"prospect_id": self.prospect_id})
        self.assertEqual(result, {"found": True, "prospect": {
            "prospect_id": self.prospect_id,
            **{k: self.record[k] for k in ("name", "email", "disqualified", "annual_revenue")},
        }})

    def test_built_profile_retains_scoring_fields_without_sensitive_data(self):
        result = agent.build_prospect_profile.invoke({"prospect_id": self.prospect_id})
        self.assertEqual(result, {"found": True, "prospect_profile": self.scoring_profile()})
        self.assertEqual(agent.data_service._PROFILES[self.prospect_id], self.scoring_profile())

    def test_cached_profile_is_filtered_at_traced_read_and_tool_output(self):
        agent.data_service._PROFILES[self.prospect_id] = {"prospect_id": self.prospect_id, **self.record}
        self.assertEqual(agent.data_service.get_profile_from_db(self.prospect_id), {
            "prospect_profile": self.scoring_profile(),
        })
        result = agent.build_prospect_profile.invoke({"prospect_id": self.prospect_id})
        self.assertEqual(result, {"found": True, "prospect_profile": self.scoring_profile()})

    def test_scoring_filters_sensitive_input_without_mutating_it(self):
        profile = {"prospect_id": self.prospect_id, **self.record}
        original = deepcopy(profile)
        result = agent.score_prospect.invoke({"prospect_profile": profile, "offering": self.offering})
        self.assertEqual(result, self.score.model_dump())
        messages = self.model.invoke.call_args.args[0]
        self.assert_private_data_absent(messages)
        serialized_profile = messages[1]["content"].split("Prospect profile:\n")[1]
        self.assertEqual(json.loads(serialized_profile), self.scoring_profile())
        self.assertEqual(profile, original)

    def test_scoring_preserves_clean_profile_and_saved_tech_stack(self):
        profile = {**self.scoring_profile(), "tech_stack": ["Unknown"]}
        result = agent.score_prospect.invoke({"prospect_profile": profile, "offering": self.offering})
        self.assertEqual(result, self.score.model_dump())
        messages = self.model.invoke.call_args.args[0]
        self.assertEqual(json.loads(messages[1]["content"].split("Prospect profile:\n")[1]), self.scoring_profile())
        self.assertEqual(agent.score_prospect.invoke({"prospect_profile": profile}), {
            "score": None, "error": "Cannot score without a valid offering.",
        })

    def test_lookup_to_scoring_flow_keeps_tool_context_and_arguments_clean(self):
        for cached in (False, True):
            with self.subTest(cached=cached):
                agent.data_service._PROFILES.clear()
                if cached:
                    agent.data_service._PROFILES[self.prospect_id] = {"prospect_id": self.prospect_id, **self.record}
                contact = agent.get_prospect.invoke({"prospect_id": self.prospect_id})
                profile = agent.build_prospect_profile.invoke({"prospect_id": self.prospect_id})
                arguments = {"prospect_profile": profile["prospect_profile"], "offering": self.offering}
                result = agent.score_prospect.invoke(arguments)
                self.assert_private_data_absent([contact, profile, arguments, result])
                self.assert_private_data_absent(self.model.invoke.call_args.args[0])
