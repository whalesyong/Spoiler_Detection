from transformers import PreTrainedTokenizerFast
from datasets import load_from_disk

# Paths and constants
TOKENIZER_PATH = "bpe_tokenizer_with_special"
RAW_DATASET_PATH = "preprocessing/BookCorpus_cleaned_hf"
TOKENIZED_DATASET_PATH = "BookCorpus_tokenized_hf"
MAX_SEQ_LENGTH = 512
TOKENIZE_BATCH_SIZE = 256  # Batch size for tokenization

def tokenize_function(examples):
    return tokenizer(
        examples["cleaned_text"],
        truncation=True,
        padding="max_length",
        max_length=MAX_SEQ_LENGTH
    )

if __name__ == "__main__":
    print(f"Loading tokenizer from {TOKENIZER_PATH}...")
    tokenizer = PreTrainedTokenizerFast.from_pretrained(TOKENIZER_PATH)

    print(f"Loading raw dataset from {RAW_DATASET_PATH}...")
    dataset = load_from_disk(RAW_DATASET_PATH)

    print("Tokenizing dataset (this may take a while)...")
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        num_proc=16,
        batch_size=TOKENIZE_BATCH_SIZE,
        remove_columns=["cleaned_text"]
    )

    print(f"Saving tokenized dataset to {TOKENIZED_DATASET_PATH}...")
    tokenized_dataset.save_to_disk(TOKENIZED_DATASET_PATH)

    print("Tokenization complete!")
