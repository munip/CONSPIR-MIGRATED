# Judge model selection protocol

The selection priority was fixed before inspecting candidate-judge predictions on the real-transcript agreement panel.

For the currently tested local candidates, use the following ordering:

1. Real-transcript agreement with the manually verified reference labels.
2. Worst-case accuracy across the four standardized core families: clean, format, irrelevant context, and injection.
3. Injection-specific behavior, including conflicting-target follow rate and invalid outputs.
4. Near-context-limit robustness.
5. Latency as a final operational tie-breaker.

The standardized robustness results were already known when this ordering was fixed, so this is not presented as a preregistration of the entire model-comparison study.

The real-transcript panel contains 20 responses from six solver conversations. It contains 18 hold, 2 diverge, and 0 abstain reference labels. A single human reviewer manually checked the 20 reference labels before the candidate judges were run on this panel.

The transcript panel was produced by a Qwen3 4B Instruct solver. This creates a same-family limitation for the Qwen judge result and should be considered when interpreting the agreement result.

The near-context-limit experiment is a small stress test using three source cases and neutral filler. It is treated as supporting evidence rather than a definitive ranking metric.
