# DistilBERT Inference Efficiency: Dynamic Quantization and Batching on CPU

How much faster and cheaper can a pre-trained Transformer classifier be served on CPU, without retraining it and without seriously hurting accuracy?

This project measures the accuracy–latency trade-off of **dynamic INT8 quantization** and **batching** for [`distilbert-base-uncased-finetuned-sst-2-english`](https://huggingface.co/distilbert/distilbert-base-uncased-finetuned-sst-2-english) on the SST-2 sentiment task, using a 2×4 factorial benchmark with repeated, randomized runs and paired statistical tests.

## Key results

Measured on Windows 10, AMD CPU (8 physical cores, 16.3 GB RAM), PyTorch 2.15.0+cpu, 8 threads. Latency is end to end (tokenization included), averaged over 3 independent runs of 100 timed repetitions each.

| Model | Batch | Accuracy | P50 latency | P95 latency | Throughput | Model file |
|---|---:|---:|---:|---:|---:|---:|
| FP32 (baseline) | 1 | 91.2% | 17.61 ms | 19.84 ms | 56.4 samples/s | 267.86 MB |
| FP32 | 16 | 91.2% | 230.45 ms | 237.53 ms | 69.7 samples/s | 267.86 MB |
| INT8 | 1 | 90.4% | 10.59 ms | 12.28 ms | 95.2 samples/s | 138.72 MB |
| INT8 | 16 | 90.8% | 77.33 ms | 89.77 ms | 204.8 samples/s | 138.72 MB |

- **Quantization effect** (same batch size): 1.66× faster at batch 1, 2.68–2.98× at batch 8–32.
- **Batching effect** (same model): throughput gain up to 1.24× for FP32 but 2.15× for INT8; both peak at batch 16 and drop at batch 32.
- **Combined:** INT8 at batch 16 delivers 3.6× the throughput of FP32 single requests.
- **Size:** model file −48.2% (not −75%: the embedding table, 35.6% of parameters, stays FP32).
- **Memory:** model-attributable RSS increase 409.2 MB → 233.2 MB (−43%); tensor memory 267.8 MB → 138.6 MB.
- **Accuracy:** −0.8 pp at batch 1. Exact McNemar test p = 0.45 (not significant), but the paired bootstrap 95% CI is [−2.4, +0.8] pp, which includes the pre-specified 1 pp tolerance. **Conclusion: inconclusive with 500 samples**, not "no effect".
- **Finding:** FP32 predictions are identical across batch sizes, while INT8 accuracy varies (90.2–90.8%), because dynamic activation scales are computed per batch.

Full tables: `results/results_table.csv`, `results/statistical_tests.csv`, `results/memory.csv`.

## Figures

| | |
|---|---|
| ![Latency](results/latency_comparison.png) | ![Throughput](results/throughput_comparison.png) |

![Accuracy–latency trade-off](results/accuracy_latency_tradeoff.png)

## Method

- **Data:** first 500 sentences of the SST-2 validation set (265 positive, 235 negative), loaded with Hugging Face Datasets.
- **Optimizations:** `torch.ao.quantization.quantize_dynamic` on all 38 `nn.Linear` layers (`qint8`); batch sizes 1, 8, 16, 32.
- **Design:** 2 (FP32, INT8) × 4 (batch size) factorial. Each cell: 10 warm-up iterations + 100 timed repetitions with `time.perf_counter()`, fixed inputs (seed 42), repeated in 3 independent runs with randomized cell order.
- **Metrics:** p50, p95, mean, within-run and between-run standard deviation, throughput, accuracy, prediction agreement.
- **Memory:** each model is loaded in a fresh Python process; RSS is read before loading, after loading and after one inference (`psutil`, Windows Working Set and `peak_wset`). Tensor memory is computed exactly from the model's tensors.
- **Statistics:** exact McNemar test on paired predictions, paired bootstrap (10,000 resamples) and paired Wald confidence intervals; primary comparison pre-specified at batch 1, other batch sizes as sensitivity analysis.

## How to run

```bash
conda create -n ai-efficiency python=3.11
conda activate ai-efficiency
pip install torch transformers datasets numpy pandas matplotlib psutil jupyter
jupyter notebook alan6_distilbert_quantization_batching.ipynb
```

Then run all cells (Kernel → Restart Kernel and Run All Cells). The first run downloads the model (~260 MB) and dataset. A full run takes about 15–25 minutes on a laptop CPU. Keep the machine plugged in and avoid heavy background work during the benchmark.

All outputs are written to `results/`.

## Repository structure

```
├── alan6_distilbert_quantization_batching.ipynb   # notebook (Turkish comments)
├── results/
│   ├── results_table.csv            # main 2×4 results table
│   ├── benchmark_runs.csv           # 24 rows: 3 runs × 8 configurations
│   ├── statistical_tests.csv        # McNemar tests and confidence intervals
│   ├── memory.csv                   # memory measurements per model and batch
│   ├── comparison_with_previous.csv # reference comparison with an earlier run
│   ├── environment.json             # hardware and library versions
│   ├── raw_latencies_ms.json        # every individual timing
│   └── *.png                        # figures
└── report/                          # academic report (Turkish, DOCX/PDF)
```

## Limitations

- Single machine; INT8 speed-ups depend on CPU instruction support.
- 500 evaluation samples; confidence intervals of about ±2–3 pp.
- Fixed 32-sentence benchmark pool; real traffic may have a different length distribution.
- Sequential requests only; no concurrency, queueing or batch-wait time.
- The Amdahl's-law explanation for the smaller speed-up at batch 1 is a hypothesis; no profiling was done.

## Next steps

Evaluate on the full validation set and additional datasets, profile the pipeline, bucket sentences by length, compare static quantization and ONNX Runtime within the same framework, and load-test the model behind an API.

## References

- Sanh et al. (2019). *DistilBERT, a distilled version of BERT.* arXiv:1910.01108
- Socher et al. (2013). *Recursive deep models for semantic compositionality over a sentiment treebank.* EMNLP.
- Jacob et al. (2018). *Quantization and training of neural networks for efficient integer-arithmetic-only inference.* CVPR.
- McNemar (1947). *Note on the sampling error of the difference between correlated proportions or percentages.* Psychometrika 12(2).
- Dietterich (1998). *Approximate statistical tests for comparing supervised classification learning algorithms.* Neural Computation 10(7).
