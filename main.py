import argparse
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
from accelerate import Accelerator
import torch
from tqdm import tqdm
from huggingface_hub import login
from datasets import load_dataset
import csv
import os

# Login to Hugging Face (optional, if using gated models)
login(token=os.environ['HF_TOKEN'])

# Prompt for the task
PROMPT = "It is {month} {year} and {formulation}"

# Dictionary to map month numbers to text form
MONTHS = {
    1: "January",
    2: "February",
    3: "March",
    4: "April",
    5: "May",
    6: "June",
    7: "July",
    8: "August",
    9: "September",
    10: "October",
    11: "November",
    12: "December",
}

# Load model and tokenizer with model parallelism using device_map
def load_model_and_tokenizer(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Use device_map to shard the model across multiple GPUs
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",  # Automatically distribute model across all available GPUs
        offload_folder="./offload",  # Optionally offload to disk if needed
        trust_remote_code=True,
    )
    return tokenizer, model


# Function to calculate log probability of a given sentence
def calculate_logprob(model, tokenizer, text):
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    # Remove 'token_type_ids' from inputs if it exists (for models like Mistral that don't use it)
    if "token_type_ids" in inputs:
        del inputs["token_type_ids"]

    # Move all inputs to the same device as the model
    inputs = {key: value.to(model.device) for key, value in inputs.items()}

    with torch.no_grad():
        # Ensure `use_cache=False` for inference, as we're not using past_key_values here
        outputs = model(**inputs, use_cache=False)

    # Get the log probabilities
    logits = outputs.logits
    shift_logits = logits[..., :-1, :].contiguous()  # Shift the logits
    shift_labels = inputs["input_ids"][..., 1:].contiguous()  # Shift the labels

    log_probs = torch.nn.functional.log_softmax(shift_logits, dim=-1)

    # Gather log probabilities of the actual tokens in the text
    log_probs_for_labels = log_probs.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)

    # Total log probability for the sequence
    total_logprob = log_probs_for_labels.sum().item()

    return (
        total_logprob,
        len(inputs["input_ids"][0].tolist()),
        log_probs_for_labels.tolist(),
    )


def test_model_on_dataset(model_name, dataset, output_file=None):
    tokenizer, model = load_model_and_tokenizer(model_name)

    if output_file is None:
        output_file = f"{model_name.replace('/', '_')}_results.csv"

    # Create CSV file with headers
    with open(output_file, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "idx",
                "year",
                "month",
                "category",
                "model",
                "formulation",
                "tokens",
                "log_probs",
                "sum_log_probs",
                "input_text",
                "correct_year",
                "correct_month",
            ]
        )

        for index, row in tqdm(
            enumerate(dataset), total=len(dataset), desc="Processing events"
        ):
            category = row["category"]  # Category column

            formulations = {
                "original_sentence": row["original_sentence"],
                "paraphrase_1": row["paraphrase_1"],
                "paraphrase_2": row["paraphrase_2"],
                "paraphrase_3": row["paraphrase_3"],
                "paraphrase_4": row["paraphrase_4"],
            }

            # Generate prompts for each month in 2022 and 2023 and for each formulation
            for key, formulation in formulations.items():
                logprobs = []
                for year in [2022, 2023]:
                    for month in range(1, 13):
                        # Generate prompt, avoid extra dot if formulation already ends with a dot
                        if not formulation.endswith("."):
                            formulation = formulation + "."
                        input_text = PROMPT.format(
                            month=MONTHS[month], year=year, formulation=formulation
                        )

                        # Calculate log probabilities
                        sum_log_prob, num_tokens, token_log_probs = calculate_logprob(model, tokenizer, input_text)
                        logprobs.append(
                            {
                                "month": month,
                                "year": year,
                                "sum_log_prob": sum_log_prob,
                                "input_text": input_text,
                                "num_tokens": num_tokens,
                                "log_probs": token_log_probs,
                            }
                        )

                # Write the results to the CSV file as we go so we don't lose progress in case of a crash
                for logprob_entry in logprobs:
                    writer.writerow(
                        [
                            index,
                            logprob_entry["year"],
                            logprob_entry["month"],
                            category,
                            model_name,
                            key,
                            logprob_entry["num_tokens"],
                            logprob_entry["log_probs"],
                            logprob_entry["sum_log_prob"],
                            logprob_entry["input_text"],
                            row["year"],
                            row["month"],
                        ]
                    )

    return output_file


def calculate_accuracy(output_file):
    # Calculate the top-1, top-3, and top-5 accuracy of the model on the original sentences

    # Load the results CSV file
    df = pd.read_csv(output_file)
    num_events = len(df["idx"].unique())

    # Get the top-5 most probable sentences for each idx (event)
    top_5_first_phrases = (
        df[df["formulation"] == "original_sentence"]
        .groupby("idx")
        .apply(lambda x: x.nlargest(5, "sum_log_probs"))
        .reset_index(drop=True)
    )

    # Top-1 Accuracy
    # Get the most probable sentence
    top_1_first_phrases = top_5_first_phrases.groupby("idx").head(1) 
    correct_top_1_phrases = top_1_first_phrases[
        (top_1_first_phrases["year"] == top_1_first_phrases["correct_year"])
        & (top_1_first_phrases["month"] == top_1_first_phrases["correct_month"])
    ]
    top_1_accuracy = len(correct_top_1_phrases) / num_events

    # Top-3 Accuracy
    top_3_first_phrases = top_5_first_phrases.groupby("idx").head(3)
    correct_top_3_phrases = top_3_first_phrases[
        (top_3_first_phrases["year"] == top_3_first_phrases["correct_year"])
        & (top_3_first_phrases["month"] == top_3_first_phrases["correct_month"])
    ]
    top_3_accuracy = len(correct_top_3_phrases) / num_events

    # Top-5 Accuracy
    correct_top_5_phrases = top_5_first_phrases[
        (top_5_first_phrases["year"] == top_5_first_phrases["correct_year"])
        & (top_5_first_phrases["month"] == top_5_first_phrases["correct_month"])
    ]
    top_5_accuracy = len(correct_top_5_phrases) / num_events
    
    return top_1_accuracy, top_3_accuracy, top_5_accuracy


def calculate_stability(output_file):
    # Computes the probability that given the original sentence was correctly
    # classified, the paraphrases are also correctly classified
    df = pd.read_csv(output_file)

    # Find events where the original sentence was correctly classified
    most_probable_original_sentences = df.loc[
        df[df["formulation"] == "original_sentence"]
        .groupby("idx")["sum_log_probs"]
        .idxmax()
    ]
    correct_original_sentences = most_probable_original_sentences[
        (
            most_probable_original_sentences["year"]
            == most_probable_original_sentences["correct_year"]
        )
        & (
            most_probable_original_sentences["month"]
            == most_probable_original_sentences["correct_month"]
        )
    ]

    correct_indices = correct_original_sentences["idx"]
    df = df[
        df["idx"].isin(correct_indices) & (df["formulation"] != "original_sentence")
    ]  # Drop the original sentence from the list of sentences

    # For paraphrase, find the most probable sentence
    most_probable_sentences = df.loc[
        df.groupby(["idx", "formulation"])["sum_log_probs"].idxmax()
    ]
    correct_sentences = most_probable_sentences[
        (most_probable_sentences["year"] == most_probable_sentences["correct_year"])
        & (most_probable_sentences["month"] == most_probable_sentences["correct_month"])
    ]

    # Calculate the stability
    stability = len(correct_sentences) / len(most_probable_sentences)
    return stability


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name", type=str, help="The Hugging Face model name, required if you want to test the model"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="Path to your result CSV file, if provided, the script will not test model again", 
    )
    args = parser.parse_args()

    if not args.model_name and not args.output_file:
        raise ValueError("You must provide either --model_name or __output_file")

    if args.output_file and os.path.exists(args.output_file): 
        output_file = args.output_file
    elif args.model_name:
        dataset = load_dataset("hereldav/TimeAware", split="train")
        output_file = test_model_on_dataset(args.model_name, dataset, args.output_file)
    else:
        raise ValueError("If no result file exists, you must specify --model_name to generate it")

    # calculate accuracy and stability
    top_1_accuracy, top_3_accuracy, top_5_accuracy = calculate_accuracy(output_file)
    print("-" * 40)
    print(f"Top-1 Accuracy: {top_1_accuracy:.2%}")
    print(f"Top-3 Accuracy: {top_3_accuracy:.2%}")
    print(f"Top-5 Accuracy: {top_5_accuracy:.2%}")

    stability = calculate_stability(output_file)
    print(f"Stability: {stability:.2%}")
    print("-" * 40)

if __name__ == "__main__":
    main()
