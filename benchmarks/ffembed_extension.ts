/**
 * pi extension: adds a `semantic_search` tool backed by ffembed.
 *
 * Load with: pi -e /path/to/ffembed_extension.ts
 * Requires the `ffembed` CLI on PATH and an indexed target.
 */
import { execFileSync } from "child_process";

interface SemanticSearchArgs {
  query: string;
}

const tool = {
  name: "semantic_search",
  label: "Semantic search",
  description:
    "Search the notes in this directory by meaning using local text embeddings. " +
    "Pass a natural-language description of what you are looking for. " +
    "Returns ranked filenames with a snippet of matching text.",
  promptSnippet:
    "- semantic_search(query): find notes by meaning (local embeddings, ranked snippets)",
  promptGuidelines: [
    "Prefer semantic_search over reading many files when looking for notes by topic or meaning.",
  ],
  parameters: {
    type: "object",
    properties: {
      query: {
        type: "string",
        description: "Natural-language description of the note you want to find.",
      },
    },
    required: ["query"],
  },

  async execute(
    _id: string,
    params: SemanticSearchArgs,
    _signal: unknown,
    _onUpdate: unknown,
    _ctx: unknown,
  ) {
    let out: string;
    try {
      out = execFileSync("ffembed", ["search", params.query, "-k", "5"], {
        encoding: "utf8",
        timeout: 60_000,
      });
    } catch (err: any) {
      out = String(err.stdout ?? err.message ?? err);
    }
    return {
      content: [{ type: "text", text: out.trim() || "(no matches)" }],
    };
  },
};

export default function (pi: any) {
  pi.registerTool(tool);
}
