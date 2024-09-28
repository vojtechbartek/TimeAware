# Framework for benchmarking LLMs on [TimeAware dataset](https://huggingface.co/datasets/hereldav/TimeAware)

## Overview

Who is the US President? The answer changes depending on **when** the question is asked. While large language models (LLMs) are evaluated on various reasoning tasks, they often miss a crucial dimension: **time**. In real-world scenarios, the correctness of answers is frequently tied to **temporal context**.

TimeAware is a novel dataset designed to rigorously test LLMs' ability to handle **time-sensitive facts**. Our benchmark offers a systematic way to measure how well models align their knowledge with the **correct time context**, filling a key gap in current evaluation methods and offering a valuable tool for improving real-world applicability in future models.

---

## Key Features
- **Time-Specific Evaluation**: Events are labeled with the **exact month** and **year**, allowing precise assessment of a model's ability to track information across time.
- **Diverse Domains**: Events span a broad spectrum, from **Politics** to **Science**, ensuring comprehensive coverage of real-world knowledge.
- **Multiple Paraphrases**: Each event is paired with **four paraphrases**, testing the robustness of models to reworded facts and phrasing variations.
- **Global Scope**: Data covers key global events, ensuring that the dataset reflects a wide range of cultural and geographical contexts.
- **Real-World Applicability**: Designed for applications in **virtual assistants**, **fact-checking systems**, and **temporal question answering**, where time-dependent accuracy is paramount.

---

## Quick Start

Install the required packages:

```bash
pip install -r requirements.txt
```

Export Hugging Face API token:

```bash
export HF_TOKEN=<YOUR_HF_TOKEN>
```

Benchmark a Hugging Face model on TimeAware:

```bash
python main.py --model_name <Hugging Face model name>
```

Calculate scores for already tested model:

```bash
python main.py --output_file <path to results file>
```

---

## Results

| Model                                      | Top-1 Acc | Top-3 Acc | Top-5 Acc | Stability |
|--------------------------------------------|-----------|-----------|-----------|-----------|
| microsoft/phi-2                            | 4.7       | 14.78     | 28.26     | 58.8      |
| google/gemma-2-2b-it                       | 5.65      | 14.7      | 23.39     | 54.23     |
| microsoft/Phi-3.5-mini-instruct            | 6.26      | 15.83     | 25.48     | 37.15     |
| microsoft/Phi-3-medium-4k-instruct         | 8.87      | 25.91     | 39.48     | 52.7      |
| google/gemma-2-2b                          | 9.83      | 25.91     | 38.0      | 59.29     |
| google/gemma-2-9b-it                       | 10.7      | 23.04     | 34.17     | 50.81     |
| meta-llama/Meta-Llama-3.1-8B-Instruct      | 14.09     | 30.87     | 42.7      | 57.87     |
| mistralai/Mistral-Nemo-Instruct-2407       | 16.09     | 35.83     | 48.0      | 57.16     |
| meta-llama/Meta-Llama-3.1-8B               | 16.43     | 35.3      | 47.91     | 60.05     |
| google/gemma-2-27b-it                      | 17.57     | 36.35     | 47.83     | 52.35     |
| mistralai/Mistral-Nemo-Base-2407           | 17.83     | 39.48     | 51.83     | 61.71     |
| google/gemma-2-9b                          | 19.22     | 42.0      | 53.3      | 64.82     |
| google/gemma-2-27b                         | 30.96     | 55.74     | 67.91     | 63.13     |
| meta-llama/Meta-Llama-3.1-70B-Instruct     | 34.61     | 58.09     | 70.26     | 65.95     |
| meta-llama/Meta-Llama-3.1-70B              | 39.74     | 66.52     | 75.39     | 65.97     |


## Citation

If you use **TimeAware**, please cite the accompanying research paper:
```
@misc{herel2024timeawarenesslargelanguage,
      title={Time Awareness in Large Language Models: Benchmarking Fact Recall Across Time}, 
      author={David Herel and Vojtech Bartek and Tomas Mikolov},
      year={2024},
      eprint={2409.13338},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2409.13338}, 
}
```
