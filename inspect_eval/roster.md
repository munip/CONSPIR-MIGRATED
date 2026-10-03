# Model roster

Inspect model strings for ConspiracyBench runs. Pass one as `--model`, and a judge as
`--model-role grader=...`. Only the local model has been run so far; the rest are the
pilot roster the "Run experiments with solver infrastructure" task asks to agree in a
Discussion first, so the transcripts also suit the judge-reliability study.

| Model | Inspect string | Provider setup | Status |
|---|---|---|---|
| Qwen3-4B-Instruct-2507 (Q4_K_M) | `ollama/cb-qwen3-4b` | `ollama create cb-qwen3-4b -f inspect_eval/ollama/qwen3-4b.Modelfile` | Run locally, model and judge (smoke test); run from a remote VM with Qwen2.5-7B judging, k = 1 |
| Qwen2.5-7B-Instruct (Q4_K_M) | `ollama/qwen2.5:7b` | `ollama pull qwen2.5:7b` | Run as the judge for Qwen3-4B, k = 1. Same family, so a plumbing check only |
| Qwen3-8B | `ollama/qwen3:8b` | `ollama pull qwen3:8b` (raise `num_ctx` as in the Modelfile) | Not run |
| Llama 3.1 8B Instruct | `vllm/meta-llama/Llama-3.1-8B-Instruct` | `pip install vllm`, GPU host | Not run |
| gpt-oss-20b | `groq/openai/gpt-oss-20b` | `GROQ_API_KEY` | Not run; the judge in `eval/judge_reliability/results/` |
| A frontier API model | `openai/<model>`, `anthropic/<model>`, `google/<model>` | provider API key | Not run |
| Llama 3.2 3B | `ollama/llama3.2:3b` | `ollama pull llama3.2:3b` | Judge adversarial pilot: 9/10 clean correct, 5/10 injected correct on conflicting injection pairs; 2 clean-correct cases flipped to the injected `hold` target |
| Gemma 3 4B | `ollama/gemma3:4b` | `ollama pull gemma3:4b` | Judge adversarial pilot: 9/10 clean correct and 9/10 injected correct on the same conflicting pairs; 0 target-label flips |

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

## Notes from a remote run

The corpus was also run with Inspect on a [Freestyle](https://www.freestyle.sh/) VM (4 vCPU,
8 GiB, Ubuntu 24.04) while both models stayed on a MacBook's Ollama, reached over a
WireGuard tunnel into the VM's private network. The 18 items at k = 1 took 11m54s. Results
are in the PR that added this section. What it took:

- **Inspect needs no changes to run remotely.** Only the Ollama address moves:
  `OLLAMA_BASE_URL=http://<mac-address>:11434/v1`.
- **Keep Ollama off the LAN.** Ollama listens on 127.0.0.1 only. A small TCP relay bound
  to the tunnel address exposed it to the VM and nothing else.
- **SSH reverse forwarding (`ssh -R`) is refused** by Freestyle's SSH proxy, hence the tunnel.
- **The VM can only reach the laptop while the tunnel is warm.** Set
  `PersistentKeepalive = 25` in the WireGuard config; without it a run stalls at 0 samples
  with HTTP retries.
- **`wg-quick` needs root and must outlive the shell that started it.** On macOS, run it
  under launchd (`sudo launchctl submit -l <label> -- ...`), not from a one-off admin prompt.
- **Start long runs in `tmux`.** `freestyle vm exec` waits for every process it started
  (so `nohup` and `setsid` don't detach) and gives up after 5 minutes.
- **Copy logs off the VM before deleting it.** `logs/` is gitignored and lives only on the VM.
- For short runs like this one a laptop is simpler. A remote runner starts to pay off for
  k = 5 over a roster, and with hosted or GPU-served models no tunnel is needed.
