# LoRA Fine-Tuning: A Comparative Study of Accuracy and Computational Efficiency

An experimental comparison of **Full Fine-Tuning** and **Low-Rank Adaptation (LoRA)** using DistilBERT on the IMDb movie review sentiment classification dataset.

This project investigates how LoRA reduces the number of trainable parameters while maintaining competitive classification performance. It also examines how different adapter ranks affect model accuracy, training time, and GPU memory consumption.

The study focuses on the trade-off between model performance and computational efficiency in parameter-efficient fine-tuning.

## Research Objectives

- Study and evaluate Low-Rank Adaptation (LoRA) using Hugging Face PEFT.
- Compare LoRA with conventional full fine-tuning.
- Investigate the effect of different LoRA ranks (4, 8, and 16) on classification performance.
- Measure the reduction in trainable parameters achieved through LoRA.
- Compare training time and peak GPU memory consumption.
- Analyse the relationship between adapter rank, model accuracy, and computational efficiency.

## Research Questions

1. How much does LoRA reduce the number of trainable parameters compared with full fine-tuning?
2. How does changing the LoRA rank affect sentiment classification accuracy?
3. Does increasing the LoRA rank improve model performance at the cost of additional computation?
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
| GPU used for reported measurements | NVIDIA Tesla T4 |

## Experimental Methodology

The experiment follows a controlled evaluation procedure:

1. Load and preprocess the IMDb movie review dataset.
2. Split the original training data into training and validation subsets while keeping the official test split separate.
3. Load the pretrained DistilBERT model.
4. Perform a learning-rate search using validation accuracy.
5. Train a full fine-tuning baseline.
6. Configure LoRA using Hugging Face PEFT with ranks 4, 8, and 16.
7. Adapt the query and value projection layers of the transformer.
8. Keep pretrained transformer weights frozen during LoRA training while updating adapter parameters and the classification head.
9. Repeat the final experiments using three random seeds.
10. Evaluate the trained models on the held-out test set.
11. Compare classification performance and computational efficiency.

### LoRA Configuration

The experiment evaluates three adapter ranks:

\[
r \in \{4,8,16\}
\]

The LoRA scaling parameter is configured as:

\[
\alpha = 2r
\]

The LoRA dropout rate is set to 0.1.

LoRA is applied to the following DistilBERT projection modules:

- `q_lin`: Query projection layer.
- `v_lin`: Value projection layer.

During LoRA fine-tuning, the pretrained transformer backbone remains frozen. The LoRA adapter parameters and classification head are trainable.

## Evaluation Metrics

The following metrics are used to compare the experimental configurations:

- **Accuracy:** Percentage of correctly classified movie reviews.
- **Weighted F1-score:** Classification performance accounting for the number of examples in each class.
- **Trainable Parameters:** Number of model parameters updated during training.
- **Training Time:** Time required to complete the training loop.
- **Peak GPU Memory:** Maximum GPU memory allocated during training.
- **Mean and Standard Deviation:** Used to analyse variability in results across three random seeds.

## Experimental Results

The final experiments compare full fine-tuning against LoRA configurations with ranks 4, 8, and 16.

Each final configuration was evaluated across three random seeds.

### 1. Test Accuracy Comparison

| Configuration | Mean Test Accuracy | Standard Deviation |
|---|---:|---:|
| Full Fine-Tuning | 85.42% | 0.24% |
| LoRA (r = 4) | 84.58% | 0.38% |
| LoRA (r = 8) | 84.45% | 0.62% |
| LoRA (r = 16) | 85.07% | 0.59% |

Rank-16 achieved the highest mean validation accuracy among the evaluated LoRA configurations, at **85.47%**.

Its mean test accuracy was **85.07%**, compared with **85.42%** for full fine-tuning.

The observed difference in mean test accuracy was 0.35 percentage points. These results indicate that rank-16 LoRA achieved competitive performance in this experimental setting.

However, three random seeds provide only a preliminary estimate of variability and are insufficient to establish statistical equivalence between the approaches.

### 2. Trainable Parameter Comparison

One of the primary objectives of this experiment is to investigate the parameter efficiency of LoRA.

For the rank-16 configuration, the trainable parameter breakdown is:

| Parameter Category | Number of Parameters |
|---|---:|
| LoRA adapter parameters | 294,912 |
| Classification head parameters | 592,130 |
| Total LoRA trainable parameters | 887,042 |
| Full fine-tuning trainable parameters | 66,955,010 |

The total trainable parameter reduction is approximately **98.67%** compared with full fine-tuning.

This distinction is important because the classification head is also trainable during LoRA fine-tuning. Therefore, adapter parameters and total trainable parameters are reported separately.

### 3. Computational Efficiency

The computational measurements were obtained using an NVIDIA Tesla T4 GPU.

| Metric | LoRA (r = 16) | Full Fine-Tuning |
|---|---:|---:|
| Mean Test Accuracy | 85.07% | 85.42% |
| Total Trainable Parameters | 887,042 | 66,955,010 |
| Peak GPU Memory | ~804 MB | ~1,436 MB |
| Training Time | ~130 seconds | ~190 seconds |

The rank-16 LoRA configuration demonstrated:

- Approximately **98.67% fewer total trainable parameters**.
- Approximately **44% lower peak GPU memory consumption**.
- Approximately **31% shorter training time**.

These measurements are specific to the experimental hardware and configuration. Training time and memory consumption may vary across GPUs, software versions, and execution environments.

### 4. Discussion

The experiments demonstrate that LoRA can substantially reduce the number of trainable parameters and computational requirements while maintaining competitive classification performance.

An increase in LoRA rank did not produce a strictly increasing trend in test accuracy. This highlights that a larger adapter rank does not necessarily guarantee better performance.

Among the evaluated LoRA configurations, rank 16 achieved the highest mean validation and test accuracy.

The results illustrate the importance of evaluating parameter-efficient fine-tuning methods using both model performance and computational efficiency rather than accuracy alone.

## Visualizations

The notebook generates the following visualizations:

1. **LoRA Rank vs. Test Accuracy:** Compares classification performance across different LoRA ranks.
2. **LoRA Rank vs. Adapter Parameters:** Illustrates how the number of adapter parameters changes with rank.
3. **Training Time Comparison:** Compares the training time of LoRA and full fine-tuning.
4. **Peak GPU Memory Comparison:** Analyses GPU memory consumption across fine-tuning configurations.

These visualizations help illustrate the relationship between adapter rank, model performance, and computational efficiency.

## How to Run

### Option 1: Google Colab (Recommended)

1. Open `LoRA_Experiments_v3_1_Annotated.ipynb` in Google Colab.
2. Select **Runtime → Change runtime type**.
3. Select **T4 GPU** if available.
4. Run all cells from top to bottom.
5. Wait for the learning-rate search and final experiments to complete.
6. Review the generated tables, plots, and numerical summary.

### Option 2: Local Environment

Clone the repository:

```bash
git clone https://github.com/Aniket7059/LoRA_CV_Project_Starter.git
cd LoRA_CV_Project_Starter
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

Launch Jupyter Notebook:

```bash
jupyter notebook
```

Open `LoRA_Experiments_v3_1_Annotated.ipynb` and execute the notebook cells.

For GPU-based experiments, ensure that your PyTorch installation supports your available CUDA environment.

## Experiment Modes

The notebook supports two execution modes:

### Full Experiment (`QUICK_MODE = False`)

- 5,000 training examples.
- 1,000 validation examples.
- 2,000 test examples.
- Three training epochs.
- Three random seeds.
- Full evaluation of full fine-tuning and LoRA ranks 4, 8, and 16.

Use this mode when generating experimental results for research analysis.

### Quick Experiment (`QUICK_MODE = True`)

- Uses smaller data subsets.
- Uses one training epoch.
- Uses one random seed.

This mode is intended to check whether the implementation runs correctly. Quick-mode results should not be used as final research findings.

## Results and Output Files

After execution, the notebook generates a `results/` directory containing the experimental outputs.

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

These files are generated by running the notebook and may be added to the repository to make the experimental findings easier to inspect.

## Important Notes and Limitations

- Experimental results depend on GPU hardware, software versions, and random seeds.
- The classification head is also trainable during LoRA fine-tuning; therefore, adapter parameters and total trainable parameters are reported separately.
- Learning rates are selected using validation accuracy rather than test accuracy.
- The test set is reserved for final model evaluation.
- Only one pretrained model architecture (DistilBERT) and one dataset (IMDb) are investigated.
- Only three LoRA ranks are compared.
- Three random seeds provide a preliminary measure of variability but are insufficient for strong statistical claims.
- Training time and GPU memory measurements are hardware-dependent.
- The study uses a maximum input sequence length of 128 tokens and a limited set of learning-rate configurations.
- LoRA is applied only to the query and value projection layers.

The experiment demonstrates parameter efficiency and computational trade-offs in a specific fine-tuning setting. Its findings should not be interpreted as establishing a universally optimal LoRA rank or proving that LoRA and full fine-tuning are statistically equivalent.

## Repository Structure

```text
LoRA_CV_Project_Starter/
│
├── LoRA_Experiments_v3_1_Annotated.ipynb
├── README.md
├── requirements.txt
│
└── results/                 # Generated after notebook execution
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

**Project Focus:** Parameter-Efficient Fine-Tuning, Low-Rank Adaptation, Transfer Learning, Deep Learning Optimization, and Computational Efficiency.
