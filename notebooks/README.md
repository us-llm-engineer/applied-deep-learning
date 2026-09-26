# Research notebooks

`01_research_foundations.ipynb` is the first cumulative study notebook. It uses seeded toy probabilities and a synthetic waveform to demonstrate metric definitions, calibration sensitivity to binning, confidence-ranked abstention, and a known-severity waveform perturbation. It does not download data, use a GPU, or reproduce the methods of the cited papers.

Run it from the project directory with:

```bash
python3 -c "import nbformat; from nbclient import NotebookClient; p='notebooks/01_research_foundations.ipynb'; n=nbformat.read(p, as_version=4); NotebookClient(n, timeout=300, kernel_name='python3').execute(); nbformat.write(n, p)"
```

The notebook requires NumPy, Matplotlib, and the Python package dependencies available to `src.metrics`, `src.calibration`, and `src.plotting`.
