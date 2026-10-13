// OpenAI → Gemini / Cursor / CommandCode request translation.
import { describe, it, expect } from "vitest";
import "./registerAll.js";
import { translateRequest } from "../../open-sse/translator/index.js";
import { FORMATS } from "../../open-sse/translator/formats.js";

const O2G = (body) => translateRequest(FORMATS.OPENAI, FORMATS.GEMINI, "m", body, true, null, "gemini");
const O2C = (body) => translateRequest(FORMATS.OPENAI, FORMATS.CURSOR, "m", body, true, null, "cursor");
const O2CC = (body) => translateRequest(FORMATS.OPENAI, FORMATS.COMMANDCODE, "m", body, true, null, "commandcode");

describe("OpenAI → Gemini", () => {
  it("preserves and combines system and developer instructions", () => {
    const out = O2G({
      messages: [
        { role: "system", content: "RULE_ONE" },
        { role: "developer", content: "RULE_TWO" },
        { role: "user", content: "hi" },
      ],
    });
    expect(JSON.stringify(out.systemInstruction)).toContain("RULE_ONE");
    expect(JSON.stringify(out.systemInstruction)).toContain("RULE_TWO");
  });
});

describe("OpenAI → Gemini image parts", () => {
  it("uses the Vertex/Gemini REST mimeType field for inline image data", () => {
    const out = O2G({
      messages: [{ role: "user", content: [
        { type: "image_url", image_url: { url: "data:image/png;base64,AAAA" } },
      ] }],
    });
    expect(out.contents[0].parts[0]).toEqual({
      inlineData: { mimeType: "image/png", data: "AAAA" },
    });
  });

  it("moves images in tool output into image parts instead of function response text", () => {
    const out = O2G({
      messages: [
        { role: "assistant", content: null, tool_calls: [
          { id: "c1", type: "function", function: { name: "read_file", arguments: "{}" } },
        ] },
        { role: "tool", tool_call_id: "c1", content: JSON.stringify({
          image: "data:image/png;base64,AAAA",
        }) },
      ],
    });
    const parts = out.contents.flatMap((turn) => turn.parts);
    expect(parts).toContainEqual({ inlineData: { mimeType: "image/png", data: "AAAA" } });
    const functionResponse = parts.find((part) => part.functionResponse);
    expect(JSON.stringify(functionResponse)).not.toContain("data:image/png;base64");
    expect(JSON.stringify(functionResponse)).toContain("[image attached separately]");
  });

  it("extracts raw base64 image fields from tool output", () => {
    const out = O2G({
      messages: [
        { role: "assistant", content: null, tool_calls: [
          { id: "c2", type: "function", function: { name: "read_file", arguments: "{}" } },
        ] },
        { role: "tool", tool_call_id: "c2", content: JSON.stringify({
          image: "iVBORw0KGgoAAAANSUhEUgAAAAEAAAAB",
        }) },
      ],
    });
    const parts = out.contents.flatMap((turn) => turn.parts);
    expect(parts).toContainEqual({
      inlineData: { mimeType: "image/png", data: "iVBORw0KGgoAAAANSUhEUgAAAAEAAAAB" },
    });
  });
});

describe("OpenAI → Cursor", () => {
  // openai-to-cursor.js:12-24 — image content fully dropped (text only)
  // KNOWN BUG
  it.fails("image content is preserved", () => {
    const out = O2C({
      messages: [{ role: "user", content: [
        { type: "text", text: "look" },
        { type: "image_url", image_url: { url: "data:image/png;base64,AAAA" } },
      ] }],
    });
    expect(JSON.stringify(out), "image dropped").toContain("AAAA");
  });

  // openai-to-cursor.js:179 — max_tokens hardcoded to 32000
  // KNOWN BUG
  it.fails("respects client max_tokens", () => {
    const out = O2C({ max_tokens: 200, messages: [{ role: "user", content: "hi" }] });
    expect(out.max_tokens).toBe(200);
  });
});

describe("OpenAI → CommandCode", () => {
  // openai-to-commandcode.js:53-57 — safeParseJson returns {} on bad JSON (args silently lost)
  // KNOWN BUG
  it.fails("malformed tool arguments are not silently emptied", () => {
    const out = O2CC({
      messages: [
        { role: "user", content: "go" },
        { role: "assistant", content: "", tool_calls: [
          { id: "c1", type: "function", function: { name: "f", arguments: "{bad" } },
        ] },
        { role: "tool", tool_call_id: "c1", content: "r" },
      ],
    });
    const asst = out.params.messages.find((m) => m.role === "assistant");
    const call = asst.content.find((b) => b.type === "tool-call");
    expect(Object.keys(call.input).length, "arguments silently dropped to {}").toBeGreaterThan(0);
  });

  it("image content is preserved as native CommandCode image blocks", () => {
    const out = O2CC({
      messages: [{ role: "user", content: [
        { type: "text", text: "look" },
        { type: "image_url", image_url: { url: "data:image/png;base64,BBBB" } },
      ] }],
    });
    expect(JSON.stringify(out)).toContain("BBBB");
    expect(JSON.stringify(out)).not.toContain("[image omitted]");
    expect(out.params.messages[0].content).toEqual([
      { type: "text", text: "look" },
      { type: "image", image: "data:image/png;base64,BBBB", mimeType: "image/png" },
    ]);
  });
});
