import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from langchain_core.messages import ToolMessage

with patch.dict(os.environ, {"OPENAI_API_KEY": os.getenv("OPENAI_API_KEY", "test"), "LANGSMITH_TRACING": "false"}):
    from gtm_agent import gtm_agent as agent


class SendProspectEmailTests(unittest.TestCase):
    def setUp(self):
        self.runtime = SimpleNamespace(config={"metadata": {"user_id": "rep_amills"}})
        self.prospect = {
            "prospect_id": "LEAD-50001",
            "name": "Test Prospect",
            "email": "prospect@example.com",
            "disqualified": False,
        }

    def send(self):
        return agent.send_prospect_email.func(
            self.prospect, "Demo", "Please book a demo.", self.runtime
        )

    def test_disqualified_source_blocks_send_despite_contact_flag(self):
        with patch.object(agent.uuid, "uuid4") as message_id:
            result = self.send()
        self.assertEqual(result, {
            "status": "blocked",
            "error": "Prospect is disqualified; email not sent.",
            "prospect_id": "LEAD-50001",
        })
        message_id.assert_not_called()

    def test_missing_source_blocks_send(self):
        for prospect_id in ("missing", None):
            with self.subTest(prospect_id=prospect_id):
                self.prospect["prospect_id"] = prospect_id
                with patch.object(agent.uuid, "uuid4") as message_id:
                    result = self.send()
                self.assertEqual(result, {
                    "status": "blocked",
                    "error": "Prospect is disqualified; email not sent.",
                    "prospect_id": prospect_id,
                })
                message_id.assert_not_called()

    def test_qualified_source_retains_successful_send(self):
        self.prospect["prospect_id"] = "LEAD-12853"
        result = self.send()
        self.assertEqual(result["status"], "sent")
        self.assertTrue(result["message_id"].startswith("msg-"))
        self.assertEqual(result["to"], self.prospect["email"])
        self.assertEqual(result["from"], "ava.mills@northpoint.com")
        self.assertEqual(result["subject"], "Demo")
        self.assertEqual(result["body"], "Please book a demo.")

    @unittest.skipUnless(os.getenv("RUN_LIVE_AGENT_TESTS"), "Requires model credentials and live model access")
    def test_agent_reports_block_without_retry(self):
        with patch.dict(os.environ, {"LANGSMITH_TRACING": "false"}):
            result = agent.run_agent(
                "Email LEAD-50001 to invite them to book a demo.", user_id="rep_amills"
            )
        sends = [
            message for message in result["messages"]
            if isinstance(message, ToolMessage) and message.name == "send_prospect_email"
        ]
        self.assertEqual(len(sends), 1)
        self.assertIn('"blocked"', sends[0].content)
        self.assertIn("disqualified", result["reply"].lower())
        self.assertRegex(result["reply"].lower(), r"not sent|wasn't sent|cannot send|can't send|blocked")


if __name__ == "__main__":
    unittest.main()
