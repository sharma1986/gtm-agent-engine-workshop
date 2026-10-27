import importlib
import json
import os
import unittest
from unittest.mock import patch

from langchain_core.messages import AIMessage, SystemMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_openai import ChatOpenAI


class GTMAgentTests(unittest.TestCase):
    def test_lookup_preserves_tools_runtime_and_final_reply(self):
        with patch.dict(os.environ, {
            "OPENAI_API_KEY": "test-key",
            "LANGSMITH_TRACING": "false",
        }):
            agent = importlib.import_module("gtm_agent.gtm_agent")
            rep = agent.REP_IDS[0]
            expected_tools = {
                "lookup_offering", "build_prospect_profile", "get_prospect",
                "send_prospect_email", "score_prospect", "update_prospect_info",
                "get_current_rep",
            }
            calls = []

            def generate(model, messages, **kwargs):
                calls.append(messages)
                systems = [m.content for m in messages if isinstance(m, SystemMessage)]
                self.assertEqual(systems, [agent.SYSTEM_PROMPT])
                self.assertEqual(
                    {t["function"]["name"] for t in kwargs["tools"]},
                    expected_tools,
                )
                if len(calls) == 1:
                    reply = AIMessage(content="", tool_calls=[{
                        "name": "get_current_rep", "args": {}, "id": "rep",
                    }])
                elif len(calls) == 2:
                    self.assertIsInstance(messages[-1], ToolMessage)
                    self.assertEqual(json.loads(messages[-1].content)["rep"], rep)
                    reply = AIMessage(content="", tool_calls=[{
                        "name": "lookup_offering",
                        "args": {"offering_id": "OFFER-10001"}, "id": "offering",
                    }])
                else:
                    offering = json.loads(messages[-1].content)
                    self.assertTrue(offering["found"])
                    reply = AIMessage(content=offering["offering"]["name"])
                return ChatResult(generations=[ChatGeneration(message=reply)])

            with patch.object(ChatOpenAI, "_generate", autospec=True, side_effect=generate):
                result = agent.run_agent(
                    "Look up OFFER-10001", user_id=rep["rep_id"],
                )
            self.assertEqual(len(calls), 3)
            self.assertEqual(result["reply"], "Core Platform Suite")
            self.assertEqual(result["reply"], result["messages"][-1].content)


if __name__ == "__main__":
    unittest.main()
