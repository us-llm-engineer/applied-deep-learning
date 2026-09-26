# Robust speech command study

This repository provides deterministic data, audio-shift, cache, calibration, and paired-evaluation foundations for a compact Speech Commands robustness study. The current notebooks use local synthetic fallbacks to make the workflow reviewable without downloading audio or fitting a model; their metrics are demonstrations, not research results.

## Run locally

From this directory, install the project dependencies and run the contract suite:

```bash
python3 -m pip install -e .
python3 -m pytest tests -q
```

Run the lightweight foundations notebook:

```bash
python3 -c "import nbformat; from nbclient import NotebookClient; p='notebooks/01_research_foundations.ipynb'; n=nbformat.read(p, as_version=4); NotebookClient(n, timeout=300, kernel_name='python3').execute(); nbformat.write(n, p)"
```

Notebook 02 combines local synthetic checks and figures with cells tagged `offloaded`. Run only untagged code cells on a local CPU. The tagged cells describe actual archive preparation, feature caching, model training, and held-out bootstrap work; execute them on the designated accelerator and bring back the same code, outputs, artifact hashes, and resource records. Do not run those cells as part of a local notebook execution.

## Selected references

- Warden, *Speech Commands: A Dataset for Limited-Vocabulary Speech Recognition* (2018), [arXiv:1804.03209](https://arxiv.org/abs/1804.03209). The dataset distribution is identified as CC BY 4.0 in its [TensorFlow Datasets catalog](https://www.tensorflow.org/datasets/catalog/speech_commands); verify the downloaded release and attribution requirements before redistribution.
- Angelopoulos et al., *Conformal Risk Control* (ICLR 2024), [paper page](https://proceedings.iclr.cc/paper_files/paper/2024/hash/f3549ef9b5ff520a7e41ff3cc306ab2b-Abstract-Conference.html).
- Liang, Peng, and Sun, *Selective Classification Under Distribution Shifts* (TMLR 2024), [paper page](https://mlanthology.org/tmlr/2024/liang2024tmlr-selective/).
- Erichson et al., *NoisyMix: Boosting Model Robustness to Common Corruptions* (AISTATS 2024), [paper page](https://proceedings.mlr.press/v238/erichson24a.html).

## Limitations

The current local notebooks do not download the public dataset, fit a CNN or pretrained encoder, or establish a robustness improvement. The planned 70/15/15 split is sample-level and can place the same speaker in multiple partitions. Acoustic augmentation motivated by non-audio corruption work remains an empirical hypothesis. Calibration thresholds measured on shifted test data are descriptive and do not provide a shift-valid guarantee. Heavy cells require the separately budgeted accelerator run; absent pretrained weights or a stopped run must be reported as unavailable rather than replaced silently.
