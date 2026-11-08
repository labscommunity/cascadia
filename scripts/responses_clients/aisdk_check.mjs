// Proves @ai-sdk/openai (Responses, store left at its default) streams text and a tool call in a multi-step
// loop against Cascadia's mock engine, and that a later turn which sends item_reference inputs (the SDK's
// default replay of stored items) resolves from Cascadia's item cache. Prints
// AISDK_TOOL_LOOP_AND_ITEM_REFERENCE: PASS or FAIL.
// Usage: node aisdk_check.mjs BASE_URL   (run from a dir whose node_modules has ai, @ai-sdk/openai, zod)
import { createOpenAI } from '@ai-sdk/openai';
import { streamText, tool, stepCountIs } from 'ai';
import { z } from 'zod';

const baseURL = process.argv[2];               // e.g. http://127.0.0.1:18080/v1
const bodies = [];
const fetchLog = async (url, init) => { if (init?.body) bodies.push(JSON.parse(init.body)); return fetch(url, init); };
const openai = createOpenAI({ baseURL, apiKey: 'not-needed', fetch: fetchLog });
const model = openai('mock-tools-model');
const calls = [];
const tools = { get_weather: tool({ description: 'Current weather', inputSchema: z.object({ city: z.string() }),
  execute: async ({ city }) => { calls.push(city); return 'sunny'; } }) };

const r1 = streamText({ model, tools, stopWhen: stepCountIs(3), prompt: 'weather __mock_tool_call__' });
let text1 = ''; const kinds = new Set();
for await (const part of r1.fullStream) { kinds.add(part.type); if (part.type === 'text-delta') text1 += part.text ?? part.delta ?? ''; }
const msgs = (await r1.response).messages;
console.log('step parts:', [...kinds].sort().join(','));
console.log('tool calls:', calls, 'text:', JSON.stringify(text1));

const r2 = streamText({ model, tools, messages: [{ role: 'user', content: 'weather __mock_tool_call__' }, ...msgs,
  { role: 'user', content: 'thanks' }] });
let text2 = ''; for await (const t of r2.textStream) text2 += t;
const lastInput = bodies.at(-1).input;
const refs = lastInput.filter(i => i.type === 'item_reference');
console.log('turn 2 text:', JSON.stringify(text2), 'item_references sent:', refs.length, 'store in body:', bodies.at(-1).store);
const ok = calls.length === 1 && kinds.has('tool-call') && text1.includes('mock: got sunny') && refs.length > 0 && text2.includes('mock: ok');
console.log('AISDK_TOOL_LOOP_AND_ITEM_REFERENCE:', ok ? 'PASS' : 'FAIL');
process.exit(ok ? 0 : 1);
