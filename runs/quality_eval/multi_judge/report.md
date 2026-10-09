# Three-judge anonymous comparison

Three separately produced score sheets were unblinded only after all 60 rows were complete. Each prompt contributes one judgment per model pair from each judge. Majority consensus requires at least two matching votes; one first-win, one tie and one second-win is recorded as no consensus.

## Majority consensus

| comparison | first wins | ties | first losses | no consensus | decisive 95% Wilson CI |
|---|---:|---:|---:|---:|---|
| base_vs_v1 | 20 | 1 | 36 | 3 | [0.245, 0.488] |
| v1_vs_v2 | 10 | 30 | 14 | 6 | [0.245, 0.612] |
| base_vs_v2 | 19 | 1 | 35 | 5 | [0.238, 0.485] |

## Individual judges

### chatgpt_work

| comparison | first wins | ties | first losses |
|---|---:|---:|---:|
| base_vs_v1 | 23 | 2 | 35 |
| v1_vs_v2 | 10 | 27 | 23 |
| base_vs_v2 | 24 | 1 | 35 |

### deepseek

| comparison | first wins | ties | first losses |
|---|---:|---:|---:|
| base_vs_v1 | 21 | 2 | 37 |
| v1_vs_v2 | 19 | 23 | 18 |
| base_vs_v2 | 23 | 3 | 34 |

### qwen

| comparison | first wins | ties | first losses |
|---|---:|---:|---:|
| base_vs_v1 | 17 | 7 | 36 |
| v1_vs_v2 | 8 | 41 | 11 |
| base_vs_v2 | 18 | 7 | 35 |

## Agreement

Across 180 prompt-pair items, all three judges agreed on 92 (51.1%); 14 had one vote in each category. Fleiss κ over first/tie/second was 0.438.

- chatgpt_work_vs_deepseek: 118/180 agreement (65.6%).
- chatgpt_work_vs_qwen: 107/180 agreement (59.4%).
- deepseek_vs_qwen: 125/180 agreement (69.4%).

## Limits

The judges are language models rather than independent human raters and can share training-data and stylistic biases. The Qwen judge also belongs to the broader model family being evaluated, which can introduce family-specific bias. Randomized A/B/C order reduces fixed position bias but cannot remove judge bias. The comparison uses 60 uniformly sampled prompts, deterministic generations and a 512-token cap; some outputs may still be truncated. Majority voting reduces single-judge idiosyncrasy but does not establish factual correctness. Results remain exploratory and apply to this internal UltraFeedback-derived test sample.

## Input hashes

- sample_ids: `bd2cf806988dead9a5655c86ac06a4fd7e63bff2fdf4b0f0666f09a76d46bd2d`
- blind_key: `d912da955a12b01df5db30488ef71679541dee0c8188620f667a20922b033048`
- chatgpt_work: `c40670b189a7e36c14b7e16a26abce883718ab8a713df50fe4de9c4fb58e5764`
- deepseek: `e2be6fc3ec68876ff36384865007b226253d948207a3b60000f7676789f17421`
- qwen: `a09e02b68dfe4dbce1b0bf8738990a458e738e7c0f054c18b834cd2aa283affa`
