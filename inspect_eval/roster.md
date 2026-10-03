# Model roster

Inspect model strings for ConspiracyBench runs. Pass one as `--model`, and a judge as
`--model-role grader=...`. Only the local model has been run so far; the rest are the
pilot roster the "Run experiments with solver infrastructure" task asks to agree in a
Discussion first, so the transcripts also suit the judge-reliability study.

| Model | Inspect string | Provider setup | Status |
|---|---|---|---|
| Qwen3-4B-Instruct-2507 (Q4_K_M) | `ollama/cb-qwen3-4b` | `ollama create cb-qwen3-4b -f inspect_eval/ollama/qwen3-4b.Modelfile` | Run locally, model and judge (smoke test) |
| Qwen3-8B | `ollama/qwen3:8b` | `ollama pull qwen3:8b` (raise `num_ctx` as in the Modelfile) | Not run |
| Llama 3.1 8B Instruct | `vllm/meta-llama/Llama-3.1-8B-Instruct` | `pip install vllm`, GPU host | Not run |
| gpt-oss-20b | `groq/openai/gpt-oss-20b` | `GROQ_API_KEY` | Not run; the judge in `eval/judge_reliability/results/` |
| A frontier API model | `openai/<model>`, `anthropic/<model>`, `google/<model>` | provider API key | Not run |

## Notes from the local run

- **Ollama's context default is 4096 tokens.** The judge prompt for a late pressure turn
  (the item's guidance plus the whole exchange) went past it and the run stopped with
  `exceed_context_size_error`. The Modelfile raises `num_ctx` to 16384. Any Ollama model
  used here needs the same.
- **On a smaller machine the same overflow is silent.** Ollama chooses its default context
  from available memory. On an 8 GB Windows laptop (Ollama 0.34.4) a 10k-token judge prompt
  was cut to 2,050 tokens with no error, and Ollama drops the start of the prompt, which is
  where the ground truth is. The task's `context_tokens` budget (default 16384, matching
  the Modelfile) stops such a prompt from being sent; keep the two values in step.
- **Ollama before 0.34 offloaded nothing to the Apple M5 GPU** (`offloaded 0/37 layers`),
  so Qwen3-4B ran on CPU at ~13 tokens/s. 0.34.4 runs it fully on the GPU at ~48
  tokens/s. Check `ollama ps` shows `100% GPU` before a long run.
- The instruct-2507 build of Qwen3-4B is the non-thinking one. A thinking build works too:
  Inspect moves `<think>` blocks out of the reply text, and the scorer strips any that remain.
