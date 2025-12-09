# Master Doge 🐕 -> 📈

**Master Doge** is a Deep Learning project designed for time-series prediction, specifically targeting financial data (OHLCV). It leverages hybrid **CNN-LSTM** architectures to capture both spatial features (via CNNs) and temporal dependencies (via LSTMs) in the data.

## 📂 Project Structure

The project is organized as follows:

```
master_doge/
├── data/               # Dataset storage
├── models/             # PyTorch model definitions and source code
│   ├── model_class.py  # Core model architectures (CNN-LSTM variants)
│   ├── load_seq.py     # Sequence loading utilities
│   ├── metrics.py      # Custom evaluation metrics
│   └── ...
├── src/                # Experimentation and training
│   ├── Optimised.ipynb      # Main optimization experiments
│   ├── tuned.ipynb          # Hyperparameter tuning notebooks
│   └── models arch.md       # Architecture diagrams (Mermaid)
└── DLRepport.pdf       # Detailed project report
```

## 🧠 Model Architectures

The project explores several variations of CNN-LSTM models, defined in `models/model_class.py`:

| Model Name | Description | Input Dim | Key Features |
| :--- | :--- | :--- | :--- |
| **Basic CNN-LSTM** | Baseline Hybrid Model | 7 | 2 Conv1D layers (32/64) + 1 LSTM (64) |
| **Optimised CNN-LSTM** | Stability Focused | 7 | Adds **Batch Normalization** after Conv layers |
| **Optimised Large CNN** | High Capacity | 7 | Larger filters (64/128) & LSTM hidden size (128) |
| **Tuned CNN-LSTM** | Flexible | 7 | Variable LSTM hidden size for hyperparameter tuning |
| **Baseline LSTM** | Simple Baseline | 5 | Pure LSTM model for performance comparison |

### Architecture Visualization
You can find visual representations of these architectures in `src/models arch.md`.

## 🚀 Getting Started

### Prerequisites
*   Python 3.x
*   PyTorch
*   Jupyter Notebook
*   Pandas, NumPy, Matplotlib

### Usage
1.  **Data Preparation**: Ensure your data is placed in the `data/` directory.
2.  **Training**: Open the notebooks in `src/` (e.g., `Optimised.ipynb` or `tuned.ipynb`) to run training experiments.
3.  **Model Definitions**: If you need to modify the architectures, edit `models/model_class.py`.

