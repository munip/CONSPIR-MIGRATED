# CONSPIR-based ConspiracyBench migration plan

This plan migrates ConspiracyBench toward domain-independent CONSPIR trait classification while preserving the existing fact-grounded, answer-behavior benchmark. Implementation follows the phases below.

## Phases
1. **Phase 1**: Upgrade schema and Inspect evals to support CONSPIR traits
2. **Phase 2**: Create synthetic data creator based on CONSPIR
3. **Phase 3**: Fine-tune Qwen2.5-3B-Instruct to be conspiracy-robust
4. **Phase 4**: Response robustness evaluation adaptation  
5. **Phase 5**: Evaluation and integration

All phases are implemented with tests.
