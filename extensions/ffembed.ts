/**
 * ffembed extension for pi: adds a `semantic_search` tool backed by the
 * local ffembed index (https://github.com/velocitatem/ffembed).
 *
 * Requires the `ffembed` CLI on PATH and at least one indexed directory:
 *   uv tool install git+https://github.com/velocitatem/ffembed
 *   ffembed watch ~/notes --filter "*.md"
 *
 * Install this extension into pi:
 *   pi install git:github.com/velocitatem/ffembed
 */
import { execFileSync } from "child_process";

interface SemanticSearchArgs {
  query: string;
  /** Max results (default 5). */
  k?: number;
}

const tool = {
  name: "semantic_search",
  label: "Semantic search",
  description:
    "Search indexed directories by meaning using local text embeddings. " +
    "Pass a natural-language description of what you are looking for — " +
    'concepts, not exact strings ("the note about waiting for quiet before acting"). ' +
    "Returns ranked filenames with a snippet of matching text. " +
    "An image file path as the query finds visually similar images.",
  promptSnippet:
    "- semantic_search(query): find notes/files by meaning via local embeddings",
  promptGuidelines: [
    "Prefer semantic_search over grep or reading many files when the query is conceptual or half-remembered.",
  ],
  parameters: {
    type: "object",
    properties: {
      query: {
        type: "string",
        description:
          "Natural-language description of what you want to find, or an image path for visual search.",
      },
      k: {
        type: "number",
        description: "Max results to return (default 5).",
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
    const args = ["search", params.query];
    if (params.k && params.k > 0) args.push("-k", String(Math.floor(params.k)));
    let out: string;
    try {
      out = execFileSync("ffembed", args, { encoding: "utf8", timeout: 60_000 });
    } catch (err: any) {
      if (err.code === "ENOENT") {
        out =
          "error: the ffembed CLI is not installed. Install it with: " +
          "uv tool install git+https://github.com/velocitatem/ffembed";
      } else {
        out = String(err.stdout ?? err.stderr ?? err.message ?? err);
        if (/no targets|not watching/i.test(out)) {
          out +=
            "\n(hint: index a directory first with `ffembed watch <dir> --filter \"*.md\"`)";
        }
      }
    }
    return { content: [{ type: "text", text: out.trim() || "(no matches)" }] };
  },
};

export default function (pi: any) {
  pi.registerTool(tool);
}
