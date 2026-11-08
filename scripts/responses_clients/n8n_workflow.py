"""n8n check for Cascadia's /v1/responses: proves n8n's AI Agent node, driven by the OpenAI Chat Model node with
"Use Responses API" on, runs a tool loop against Cascadia's mock engine (two model runs, one get_weather tool run,
agent output "mock: got ..."). Prints N8N_RESPONSES_AGENT_TOOL_LOOP: PASS or FAIL.

Usage:
  python3 n8n_workflow.py gen BASE_URL CREDS_JSON WORKFLOW_JSON   write the credential and workflow to import
  python3 n8n_workflow.py verify EXECUTE_OUTPUT                     judge `n8n execute --rawOutput` output
The credential's API key is the placeholder "not-needed"; Cascadia does not check it.
"""
import json
import sys

WORKFLOW_ID = "v3n8nagent"


def gen(base, creds_path, wf_path):
    creds = [{"id": "cascadia1", "name": "cascadia", "type": "openAiApi", "data": {"apiKey": "not-needed", "url": base}}]
    wf = {"id": WORKFLOW_ID, "name": "v3 n8n agent", "active": False, "settings": {}, "connections": {
        "Start": {"main": [[{"node": "AI Agent", "type": "main", "index": 0}]]},
        "OpenAI Chat Model": {"ai_languageModel": [[{"node": "AI Agent", "type": "ai_languageModel", "index": 0}]]},
        "get_weather": {"ai_tool": [[{"node": "AI Agent", "type": "ai_tool", "index": 0}]]}},
      "nodes": [
        {"id": "n1", "name": "Start", "type": "n8n-nodes-base.manualTrigger", "typeVersion": 1, "position": [0, 0], "parameters": {}},
        {"id": "n2", "name": "AI Agent", "type": "@n8n/n8n-nodes-langchain.agent", "typeVersion": 3.1, "position": [200, 0],
         "parameters": {"promptType": "define", "text": "weather __mock_tool_call__", "options": {}}},
        {"id": "n3", "name": "OpenAI Chat Model", "type": "@n8n/n8n-nodes-langchain.lmChatOpenAi", "typeVersion": 1.3, "position": [200, 200],
         "parameters": {"model": {"__rl": True, "mode": "id", "value": "mock-tools-model"}, "responsesApiEnabled": True, "options": {}},
         "credentials": {"openAiApi": {"id": "cascadia1", "name": "cascadia"}}},
        {"id": "n4", "name": "get_weather", "type": "@n8n/n8n-nodes-langchain.toolCode", "typeVersion": 1.3, "position": [400, 200],
         "parameters": {"name": "get_weather", "description": "Current weather for a city", "language": "javaScript",
                        "jsCode": "return 'sunny';", "specifyInputSchema": True, "schemaType": "manual",
                        "inputSchema": json.dumps({"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]})}}]}
    with open(creds_path, "w") as f:
        json.dump(creds, f)
    with open(wf_path, "w") as f:
        json.dump([wf], f)


def verify(out_path):
    ok = False
    try:
        s = open(out_path).read()
        d = json.loads(s[s.index("{"):])
        rd = d["data"]["resultData"]["runData"]
        model_runs = len(rd.get("OpenAI Chat Model", []))
        tool_runs = len(rd.get("get_weather", []))
        out = rd["AI Agent"][0]["data"]["main"][0][0]["json"]["output"]
        print("status:", d["status"], "model runs:", model_runs, "tool runs:", tool_runs, "agent output:", repr(out))
        ok = d["status"] == "success" and model_runs == 2 and tool_runs == 1 and out.startswith("mock: got")
    except (OSError, ValueError, KeyError, IndexError, TypeError) as e:
        print("could not read the n8n execution result:", repr(e))
    print("N8N_RESPONSES_AGENT_TOOL_LOOP:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    if len(sys.argv) == 5 and sys.argv[1] == "gen":
        gen(*sys.argv[2:])
    elif len(sys.argv) == 3 and sys.argv[1] == "verify":
        sys.exit(verify(sys.argv[2]))
    else:
        sys.exit(__doc__)
