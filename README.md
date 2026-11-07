# LoRA Fine-Tuning: A Comparative Study of Accuracy and Computational Efficiency

An experimental comparison of **Full Fine-Tuning** and **Low-Rank Adaptation (LoRA)** using DistilBERT on the IMDb movie review sentiment classification dataset.

This project investigates how LoRA reduces trainable parameters while maintaining competitive classification performance, and examines how different adapter ranks affect accuracy, training time, and GPU memory consumption.

## Research Objectives

- Understand and implement Low-Rank Adaptation (LoRA).
- Compare LoRA with conventional full fine-tuning.
- Investigate the effect of different LoRA ranks (4, 8, and 16) on classification performance.
- Measure trainable parameter reduction.
- Compare training time and peak GPU memory usage.
- Analyse the trade-off between model accuracy and computational efficiency.

## Research Questions

1. How much does LoRA reduce the number of trainable parameters compared with full fine-tuning?
2. How does increasing LoRA rank affect sentiment classification accuracy?
3. Does increasing LoRA rank improve model performance at the cost of additional computation?
4. Which LoRA rank provides the best balance between accuracy and computational efficiency?
5. How do training time and GPU memory requirements differ between full fine-tuning and LoRA?

## Experimental Setup

| Component | Configuration |
|---|---|
| Base model | `distilbert/distilbert-base-uncased` |
| Dataset | IMDb Movie Reviews |
| Task | Binary Sentiment Classification |
| Framework | PyTorch |
| Model library | Hugging Face Transformers |
| LoRA implementation | Hugging Face PEFT |
| Training examples | 5,000 |
| Validation examples | 1,000 |
| Test examples | 2,000 |
| Training epochs | 3 |
| Batch size | 16 |
| Maximum sequence length | 128 |
| LoRA ranks | 4, 8, 16 |
| LoRA target modules | `q_lin`, `v_lin` |
| LoRA alpha | 2 × rank |
| LoRA dropout | 0.1 |
| Random seeds | 0, 1, 2 |

### Experimental Methodology

The experiment follows these steps:

1. Load and preprocess the IMDb dataset.
2. Split the original training data into training and validation subsets, keeping the official test split separate.
3. Load the pretrained DistilBERT model.
4. Perform a learning-rate search using validation accuracy.
5. Train a full fine-tuning baseline.
6. Train LoRA models with ranks 4, 8, and 16.
7. Repeat the final experiments using three random seeds.
8. Evaluate final models on the held-out test set.
9. Compare accuracy, weighted F1-score, trainable parameters, training time, and peak GPU memory.

For LoRA experiments, the pretrained transformer weights remain frozen while the adapter parameters and classification head are trainable.

## Evaluation Metrics

The following metrics are used:

- **Accuracy:** Percentage of correctly classified movie reviews.
- **Weighted F1-score:** Classification performance accounting for the number of examples in each class.
- **Trainable Parameters:** Number of parameters updated during training.
- **Training Time:** Time required to complete the training loop.
- **Peak GPU Memory:** Maximum GPU memory allocated during training.
- **Mean and Standard Deviation:** Used to analyse results across three random seeds.

## Visualizations

The notebook generates four visualizations:

1. LoRA rank vs. test accuracy.
2. LoRA rank vs. adapter parameters.
3. Training time comparison.
4. Peak GPU memory comparison.

These visualizations help illustrate the relationship between adapter rank, classification performance, and computational efficiency.

## How to Run

### Option 1: Google Colab (Recommended)

1. Open the notebook in Google Colab.
2. Select **Runtime → Change runtime type**.
3. Select **T4 GPU** if available.
4. Run all cells from top to bottom.
5. Wait for the learning-rate search and final experiments to complete.
6. Review the generated tables, plots, and summary.

### Option 2: Local Environment

Install the required dependencies:

```bash
pip install torch transformers datasets peft accelerate scikit-learn pandas numpy matplotlib
```

Open the notebook using Jupyter:

```bash
jupyter notebook
```

Run all cells in the notebook.

## Experiment Modes

The notebook supports two execution modes:

- **Full experiment (`QUICK_MODE = False`):** Uses 5,000 training examples, 1,000 validation examples, 2,000 test examples, three epochs, and three random seeds. Use this mode for reporting research results.
- **Quick experiment (`QUICK_MODE = True`):** Uses smaller subsets, one epoch, and one seed to check whether the implementation runs correctly. Do not use quick-mode results as final research findings.

## Results and Output Files

After execution, the notebook creates a `results/` directory containing:

| File | Description |
|---|---|
| `lr_sweep.csv` | Learning-rate search results |
| `final_runs.csv` | Individual experimental results across seeds |
| `summary.csv` | Aggregated mean and standard deviation |
| `summary.txt` | Automatically generated numerical summary |
| `rank_vs_accuracy.png` | Accuracy comparison |
| `rank_vs_adapter_params.png` | Adapter parameter comparison |
| `training_time.png` | Training time comparison |
| `peak_memory.png` | GPU memory comparison |

## Important Notes and Limitations

- Experimental results depend on GPU hardware, software versions, and random seeds.
- The classification head is also trainable during LoRA fine-tuning; therefore, adapter parameters and total trainable parameters are reported separately.
- The learning rate is selected using validation accuracy, not test accuracy.
- The test set is reserved for final model evaluation.
- Only one pretrained model architecture and one dataset are investigated.
- Only three LoRA ranks are compared.
- Three random seeds provide a preliminary measure of variability but are not sufficient for strong statistical claims.
- Training time and memory measurements are hardware-dependent.

The experiment demonstrates parameter efficiency and computational trade-offs, but its findings should not be interpreted as establishing a universally optimal LoRA rank.

## Repository Structure

```text
.
├── LoRA_Experiments_v3_1_Annotated.ipynb
├── README.md
├── requirements.txt
└── results/
    ├── lr_sweep.csv
    ├── final_runs.csv
    ├── summary.csv
    ├── summary.txt
    ├── rank_vs_accuracy.png
    ├── rank_vs_adapter_params.png
    ├── training_time.png
    └── peak_memory.png
```

## References

1. Hu et al. (2021). *LoRA: Low-Rank Adaptation of Large Language Models.*  
   https://arxiv.org/abs/2106.09685

2. Hugging Face PEFT Documentation.  
   https://huggingface.co/docs/peft

3. Hugging Face Transformers Documentation.  
   https://huggingface.co/docs/transformers

4. IMDb Dataset on Hugging Face.  
   https://huggingface.co/datasets/stanfordnlp/imdb

---

**Project Focus:** Parameter-Efficient Fine-Tuning, Low-Rank Adaptation, Transfer Learning, and Computational Efficiency.
