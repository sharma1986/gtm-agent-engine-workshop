import importlib
import json
import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key", "LANGSMITH_TRACING": "false"}):
    agent = importlib.import_module("gtm_agent.gtm_agent")


class RecordResolutionTests(unittest.TestCase):
    def setUp(self):
        self.runtime = SimpleNamespace(config={"metadata": {"user_id": "rep-id"}})
        self.rep = {"name": "Sales Rep", "email": "rep@example.com"}

    def test_email_resolves_contact_and_sender(self):
        with patch.object(agent.data_service, "get_prospect_record", return_value={"name": "Prospect", "email": "lead@example.com"}), patch.object(agent.data_service, "get_rep", return_value=self.rep):
            result = agent.send_prospect_email.func("lead-id", "Subject", "Body", self.runtime)
        self.assertEqual(result["status"], "sent")
        self.assertEqual(result["to"], "lead@example.com")
        self.assertEqual(result["from"], self.rep["email"])
        self.assertEqual((result["subject"], result["body"]), ("Subject", "Body"))

    def test_email_override_and_failures(self):
        override = {"email": "override@example.com"}
        with patch.object(agent.data_service, "get_prospect_record", return_value={"email": "lead@example.com"}):
            result = agent.send_prospect_email.func("lead-id", "Subject", "Body", self.runtime, override)
        self.assertEqual(result["from"], override["email"])
        for record, error in [(None, "Prospect record not found."), ({}, "Prospect record has no email address.")]:
            with self.subTest(record=record), patch.object(agent.data_service, "get_prospect_record", return_value=record), patch.object(agent.data_service, "get_rep", return_value=self.rep):
                result = agent.send_prospect_email.func("missing", "Subject", "Body", self.runtime)
            self.assertEqual(result, {"status": "failed", "error": error})
        with patch.object(agent.data_service, "get_prospect_record", side_effect=RuntimeError("Unavailable")):
            result = agent.send_prospect_email.func("lead-id", "Subject", "Body", self.runtime)
        self.assertEqual(result["status"], "failed")
        self.assertIn("Unavailable", result["error"])

    def test_scoring_resolves_ids_and_refreshes_cached_stack(self):
        prospect_id = next(iter(agent.data_service.PROSPECTS))
        offering_id = next(iter(agent.data_service.OFFERINGS))
        score = {"score": 75, "justification": "Fit"}
        with patch.dict(agent.data_service._PROFILES, {}, clear=True), patch.object(agent, "_scoring_llm") as model:
            model.invoke.return_value.model_dump.return_value = score
            self.assertEqual(agent.score_prospect.invoke({"prospect_id": prospect_id, "offering_id": offering_id}), score)
            prompt = model.invoke.call_args.args[0][1]["content"]
            self.assertIn(json.dumps(agent.data_service.OFFERINGS[offering_id], indent=2), prompt)
            self.assertEqual(json.loads(prompt.split("Prospect profile:\n")[1])["account_details"], agent.data_service.PROSPECTS[prospect_id]["account_details"])
            agent.data_service._PROFILES[prospect_id]["tech_stack"] = ["Stale"]
            with patch.object(agent.data_service, "fetch_tech_stack", return_value=["Current"]):
                self.assertEqual(agent.score_prospect.invoke({"prospect_id": prospect_id, "offering_id": offering_id}), score)
            prompt = model.invoke.call_args.args[0][1]["content"]
            self.assertIn('"Current"', prompt)
            self.assertNotIn('"Stale"', prompt)

    def test_scoring_resolution_errors_do_not_score(self):
        offering_id = next(iter(agent.data_service.OFFERINGS))
        with patch.object(agent, "_scoring_llm") as model:
            for offering in (None, {}, {"description": "Incomplete"}):
                with self.subTest(offering=offering), patch.object(agent.data_service, "get_offering", return_value=offering):
                    self.assertEqual(agent.score_prospect.invoke({"prospect_id": "missing", "offering_id": "invalid"}), {"score": None, "error": "Cannot score without a valid offering."})
            result = agent.score_prospect.invoke({"prospect_id": "missing", "offering_id": offering_id})
            self.assertEqual(result, {"score": None, "error": "Prospect record not found."})
            with patch.object(agent.data_service, "get_offering", side_effect=RuntimeError("Unavailable")):
                result = agent.score_prospect.invoke({"prospect_id": "missing", "offering_id": offering_id})
            self.assertIsNone(result["score"])
            self.assertIn("Unavailable", result["error"])
            model.invoke.assert_not_called()

    def test_email_graph_sends_directly_from_ids(self):
        from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
        from langchain_core.messages import AIMessage

        class ToolModel(FakeMessagesListChatModel):
            def bind_tools(self, tools, **kwargs):
                return self

        model = ToolModel(responses=[
            AIMessage(content="", tool_calls=[{"name": "send_prospect_email", "args": {"prospect_id": "lead-id", "subject": "Subject", "body": "Body"}, "id": "send-1"}]),
            AIMessage(content="Sent"),
        ])
        graph = agent.create_deep_agent(model=model, tools=[agent.send_prospect_email, agent.get_current_rep], system_prompt=agent.SYSTEM_PROMPT, context_schema=dict, middleware=[agent.rep_prompt])
        with patch.object(agent, "gtm_agent", graph), patch.object(agent.data_service, "get_rep", return_value=self.rep), patch.object(agent.data_service, "get_prospect_record", return_value={"email": "lead@example.com"}):
            result = agent.run_agent("Email lead-id", user_id="rep-id")
        self.assertEqual(result["reply"], "Sent")
        sent = [message for message in result["messages"] if message.type == "tool"]
        self.assertEqual(len(sent), 1)
        self.assertEqual(json.loads(sent[0].content)["to"], "lead@example.com")
        self.assertEqual(json.loads(sent[0].content)["from"], self.rep["email"])

    def test_rep_context_and_lookup_remain_available(self):
        with patch.object(agent.data_service, "get_rep", return_value=self.rep), patch.object(agent, "gtm_agent") as graph:
            graph.invoke.return_value = {"messages": [SimpleNamespace(content="Sent")]}
            self.assertEqual(agent.run_agent("Email lead-id", user_id="rep-id")["reply"], "Sent")
            self.assertEqual(graph.invoke.call_args.kwargs["context"]["signed_in_rep"], self.rep)
            self.assertEqual(agent.get_current_rep.func(self.runtime), {"rep": self.rep, "found": True})
        self.assertEqual(set(agent.score_prospect.args), {"prospect_id", "offering_id"})
        self.assertIn("prospect_id", agent.send_prospect_email.args)
        self.assertNotIn("prospect", agent.send_prospect_email.args)


if __name__ == "__main__":
    unittest.main()
