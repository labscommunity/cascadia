"""Proves the OpenAI Agents SDK (openai-agents, default Responses model) runs a tool-calling agent against
Cascadia's mock engine: the mock calls get_weather once, the SDK runs it and sends the output back, and the
final answer is the mock's "mock: got sunny". Prints AGENTS_SDK_TOOL_LOOP: PASS or FAIL.
Usage: python agents_sdk_check.py BASE_URL [MODEL]   (e.g. http://127.0.0.1:18080/v1)
"""
import asyncio, os, sys
os.environ["OPENAI_AGENTS_DISABLE_TRACING"] = "1"
from openai import AsyncOpenAI
from agents import Agent, Runner, function_tool, set_default_openai_client, set_tracing_disabled

base = sys.argv[1]  # e.g. http://127.0.0.1:18080/v1
model = sys.argv[2] if len(sys.argv) > 2 else "mock-tools-model"
calls = []

@function_tool
def get_weather(city: str) -> str:
    """Current weather for a city."""
    calls.append(city)
    return "sunny"

async def main():
    set_tracing_disabled(True)
    set_default_openai_client(AsyncOpenAI(base_url=base, api_key="not-needed"), use_for_tracing=False)
    agent = Agent(name="weather", instructions="Use the tool.", tools=[get_weather], model=model)
    result = await Runner.run(agent, "weather __mock_tool_call__", max_turns=4)
    print("tool calls:", calls)
    print("final_output:", repr(result.final_output))
    ok = calls == ["mock"] and "mock: got sunny" in str(result.final_output)
    print("AGENTS_SDK_TOOL_LOOP:", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)

asyncio.run(main())
