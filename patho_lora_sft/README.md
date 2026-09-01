# PathAgent morphology SFT toolkit

This package contains the reusable, data-free parts of the PathAgent morphology Schema SFT workflow:

- strict morphology JSON parsing and constrained decoding;
- image and text privacy gates;
- patient-group isolation and deterministic split construction;
- anonymous single and paired review queues;
- assistant-only loss masks and LoRA target audits;
- atomic recovery, budget gates, evaluation, and rollback checks.

The repository does not include WSI files, patches, source manifests, teacher responses, training JSONL, review queues, API records, checkpoints, or adapters. Hashing an identifier is not sufficient authorization to publish it.

Install the test environment:

```bash
python -m pip install -e '.[sft-test]'
python -m pytest -q tests/test_patho_lora_s0_s1.py tests/test_patho_lora_s0_s1_failures.py
```

Use the templates in `configs/sft/` only after replacing placeholders locally. Keep completed contracts and all generated runs outside Git. This is research software and is not a clinical diagnostic system.
