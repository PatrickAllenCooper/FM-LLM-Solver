import os
import argparse
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    pipeline,
    logging,
)
from peft import LoraConfig, PeftModel, get_peft_model
from trl import SFTTrainer
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")
logging.set_verbosity_error() # Reduce transformers logging verbosity

# --- Configuration --- #

# Model parameters
# Recommended: Start with smaller models like Mistral-7B or Llama-3-8B
# Hugging Face model name
DEFAULT_MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"

# Data parameters
DEFAULT_DATA_PATH = "paper_population/finetuning_data.jsonl" # Output from create_finetuning_data.py
DEFAULT_DATA_FORMAT = "instruction" # Must match the format used in create_finetuning_data.py ('instruction' or 'prompt_completion')

# QLoRA parameters
LORA_R = 64             # LoRA attention dimension (rank)
LORA_ALPHA = 16         # Alpha parameter for LoRA scaling
LORA_DROPOUT = 0.1      # Dropout probability for LoRA layers
TARGET_MODULES = [      # Target modules for LoRA varies by model arch, check model card or experiment
    "q_proj",           # Common for Llama/Mistral
    "k_proj",
    "v_proj",
    "o_proj",
    # "gate_proj",        # Uncomment based on model architecture
    # "up_proj",
    # "down_proj",
]

# bitsandbytes parameters
USE_4BIT = True                 # Activate 4-bit precision base model loading
BNB_4BIT_COMPUTE_DTYPE = "float16" # Compute dtype for 4-bit base models (e.g., "float16", "bfloat16")
BNB_4BIT_QUANT_TYPE = "nf4"     # Quantization type (fp4 or nf4)
USE_NESTED_QUANT = False        # Activate nested quantization for 4-bit base models (double quantization)

# TrainingArguments parameters
OUTPUT_DIR = "./results_barrier_certs" # Directory to save checkpoints and final adapter
NUM_TRAIN_EPOCHS = 1              # Number of training epochs (start with 1-3)
PER_DEVICE_TRAIN_BATCH_SIZE = 4   # Batch size per GPU for training (adjust based on VRAM)
PER_DEVICE_EVAL_BATCH_SIZE = 4    # Batch size per GPU for evaluation (if using evaluation dataset)
GRADIENT_ACCUMULATION_STEPS = 1   # Number of update steps to accumulate gradients over (increase if low VRAM)
GRADIENT_CHECKPOINTING = True     # Enable gradient checkpointing to save memory
MAX_GRAD_NORM = 0.3               # Max gradient norm (gradient clipping)
LEARNING_RATE = 2e-4              # Initial learning rate (AdamW optimizer)
WEIGHT_DECAY = 0.001              # Weight decay applied to all layers except bias/LayerNorm weights
OPTIM = "paged_adamw_32bit"       # Optimizer to use (paged optimizers save memory)
LR_SCHEDULER_TYPE = "cosine"      # Learning rate schedule type (e.g., "linear", "cosine")
MAX_STEPS = -1                    # Number of training steps (overrides num_train_epochs if > 0)
WARMUP_RATIO = 0.03               # Ratio of steps for linear warmup (from 0 to learning rate)
GROUP_BY_LENGTH = True            # Group sequences into batches with similar lengths - saves memory & speeds up training
SAVE_STEPS = 25                   # Save checkpoint every X update steps
LOGGING_STEPS = 25                # Log metrics every X update steps
PACKING = False                   # Pack multiple short examples into the same input sequence to increase efficiency (requires max_seq_length)
DEVICE_MAP = {"": 0}              # Load the entire model on the same GPU (adjust for multi-GPU)
MAX_SEQ_LENGTH = None             # Maximum sequence length to use (set if using packing, e.g., 1024 or 2048)


# --- Helper Functions ---

def formatting_prompts_func(example):
    """Function to format dataset examples for instruction fine-tuning."""
    output_texts = []
    for i in range(len(example['instruction'])):
        # Apply a chat template if the base model supports it (recommended for Instruct models)
        # This example assumes a simple concatenation, adapt based on model docs/tokenizer
        text = f"<s>[INST] {example['instruction'][i]} \n System: {example['input'][i]} [/INST] {example['output'][i]} </s>"
        output_texts.append(text)
    return output_texts

def formatting_prompt_completion_func(example):
    """Function to format dataset examples for prompt/completion fine-tuning."""
    # Directly use the 'prompt' and 'completion' fields if they exist
    # The SFTTrainer expects a 'text' field by default, or you can specify dataset_text_field
    # This function ensures the data is in the expected "prompt" + "completion" format
    # (Assuming the SFTTrainer handles this implicitly or via dataset_text_field)
    # If using dataset_text_field="text", combine them here:
    output_texts = []
    for i in range(len(example['prompt'])):
         text = example['prompt'][i] + example['completion'][i]
         # Add EOS token if needed by the model/trainer setup
         # text += "</s>" # Example for Llama
         output_texts.append(text)
    return output_texts


# --- Main Fine-tuning Logic ---

def main(model_name, data_path, data_format):

    print(f"--- Starting Fine-tuning Process ---")
    print(f"Model: {model_name}")
    print(f"Data Path: {data_path}")
    print(f"Data Format: {data_format}")

    # 1. Load Dataset
    print(f"Loading dataset from {data_path}...")
    try:
        dataset = load_dataset('json', data_path=data_path, split='train')
        print(f"Dataset loaded: {dataset}")
        # Optional: Shuffle dataset
        dataset = dataset.shuffle(seed=42)
        # Optional: Create train/validation split
        # dataset_split = dataset.train_test_split(test_size=0.1)
        # train_dataset = dataset_split["train"]
        # eval_dataset = dataset_split["test"]
        train_dataset = dataset # Use full dataset for training if no split
        eval_dataset = None     # Set to None if not evaluating

    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    # 2. Configure Quantization (QLoRA)
    compute_dtype = getattr(torch, BNB_4BIT_COMPUTE_DTYPE)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=USE_4BIT,
        bnb_4bit_quant_type=BNB_4BIT_QUANT_TYPE,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=USE_NESTED_QUANT,
    )

    # Check GPU compatibility with bfloat16
    if compute_dtype == torch.float16 and USE_4BIT:
        major, _ = torch.cuda.get_device_capability()
        if major >= 8:
            print("=" * 80)
            print("Your GPU supports bfloat16: accelerate training with bf16=True")
            print("=" * 80)
            # Set bf16=True in TrainingArguments below if desired

    # 3. Load Base Model
    print(f"Loading base model: {model_name}...")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map=DEVICE_MAP, # Automatically distributes across GPUs if available
            # trust_remote_code=True # Needed for some models
        )
        model.config.use_cache = False # Required for gradient checkpointing
        model.config.pretraining_tp = 1 # Set if needed for some model types
    except Exception as e:
        print(f"Error loading base model: {e}")
        return

    # 4. Load Tokenizer
    print(f"Loading tokenizer for {model_name}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        # Set padding token if missing (e.g., for Llama)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            print("Set pad_token to eos_token")
        tokenizer.padding_side = "right" # Fix weird overflow issue with fp16 training
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        return

    # 5. Configure LoRA
    peft_config = LoraConfig(
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        r=LORA_R,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=TARGET_MODULES
    )

    # If not using PEFT, model is loaded directly. If using PEFT:
    # model = get_peft_model(model, peft_config) # Apply LoRA config
    # print("PEFT model configured:")
    # model.print_trainable_parameters()

    # 6. Configure Training Arguments
    training_arguments = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_TRAIN_EPOCHS,
        per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        optim=OPTIM,
        save_steps=SAVE_STEPS,
        logging_steps=LOGGING_STEPS,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        fp16=False, # Set to True if using float16 compute dtype and GPU supports it
        bf16=True if BNB_4BIT_COMPUTE_DTYPE == 'bfloat16' else False, # Set based on compute dtype
        max_grad_norm=MAX_GRAD_NORM,
        max_steps=MAX_STEPS,
        warmup_ratio=WARMUP_RATIO,
        group_by_length=GROUP_BY_LENGTH,
        lr_scheduler_type=LR_SCHEDULER_TYPE,
        report_to="tensorboard", # Or "wandb", "none"
        gradient_checkpointing=GRADIENT_CHECKPOINTING,
        # evaluation_strategy="steps" if eval_dataset else "no", # Enable if eval_dataset exists
        # eval_steps=SAVE_STEPS if eval_dataset else None,       # Evaluate every save_steps
        # per_device_eval_batch_size=PER_DEVICE_EVAL_BATCH_SIZE, # If evaluating
    )

    # 7. Initialize SFT Trainer
    print("Initializing SFT Trainer...")

    # Select formatting function based on data_format
    formatting_func = None
    dataset_text_field = None # Use default 'text' field if formatting_func is None
    if data_format == "instruction":
        # For instruction format, we need to format the data into a single string
        # SFTTrainer can handle this if dataset has 'instruction', 'input', 'output'
        # but often a custom formatting function applying a model-specific template is better.
        # Let's assume a simple text field 'text' will be created by a map function
        # Or set dataset_text_field if your dataset has the combined text already
        # Alternatively, provide the formatting_func directly:
        formatting_func = formatting_prompts_func
        # We need to map the dataset to create the 'text' field expected by SFTTrainer
        # when formatting_func is provided this way.
        # However, TRL's SFTTrainer can sometimes infer this. Let's rely on that for now
        # If errors occur, explicitly map the dataset first:
        # mapped_train_dataset = train_dataset.map(lambda x: {'text': formatting_prompts_func(x)}, batched=True)
        print("Using instruction formatting.")

    elif data_format == "prompt_completion":
        # For prompt/completion, SFTTrainer can often handle this directly if
        # the dataset has 'prompt' and 'completion' columns. We might need to specify:
        # dataset_text_field = "prompt" # Or combine them into a 'text' field
        # Let's assume SFTTrainer handles prompt+completion, or use a formatting func to combine:
        formatting_func = formatting_prompt_completion_func # Combines prompt+completion into 'text'
        print("Using prompt/completion formatting (will combine into single 'text' field).")
        # mapped_train_dataset = train_dataset.map(lambda x: {'text': formatting_prompt_completion_func(x)}, batched=True)


    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset, # Use mapped_train_dataset if you explicitly mapped
        eval_dataset=eval_dataset, # Use mapped_eval_dataset if applicable
        peft_config=peft_config, # Pass LoRA config here
        # dataset_text_field="text", # Specify if your dataset has a pre-formatted text field
        formatting_func=formatting_func, # Pass formatting func if needed
        max_seq_length=MAX_SEQ_LENGTH, # Specify sequence length if packing or needed
        tokenizer=tokenizer,
        args=training_arguments,
        packing=PACKING,
    )

    # 8. Start Training
    print("--- Starting Training ---")
    try:
        trainer.train()
        print("--- Training Finished ---")
    except Exception as e:
        print(f"Error during training: {e}")
        return

    # 9. Save Final Adapter
    final_adapter_path = os.path.join(OUTPUT_DIR, "final_adapter")
    print(f"Saving final adapter model to {final_adapter_path}...")
    try:
        trainer.model.save_pretrained(final_adapter_path)
        # Also save tokenizer
        tokenizer.save_pretrained(final_adapter_path)
        print("Adapter and tokenizer saved successfully.")
    except Exception as e:
        print(f"Error saving final adapter: {e}")

    # --- Optional: Test Inference --- #
    # print("\n--- Testing Inference --- ")
    # try:
    #     prompt = "Given the autonomous system described by the following dynamics, propose a suitable barrier certificate function B(x).\n System: dx/dt = -x^3 - y\ndy/dt = x - y^3" # Example prompt
    #     pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer, max_length=200)
    #     result = pipe(f"<s>[INST] {prompt} [/INST]")
    #     print(result[0]['generated_text'])
    # except Exception as e:
    #     print(f"Error during inference test: {e}")

    print("\n--- Fine-tuning Script Completed ---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune a model for barrier certificate generation using QLoRA.")
    parser.add_argument("--model_name", type=str, default=DEFAULT_MODEL_NAME,
                        help=f"Hugging Face model name (default: {DEFAULT_MODEL_NAME}).")
    parser.add_argument("--data_path", type=str, default=DEFAULT_DATA_PATH,
                        help=f"Path to the fine-tuning data JSONL file (default: {DEFAULT_DATA_PATH}).")
    parser.add_argument("--data_format", type=str, default=DEFAULT_DATA_FORMAT, choices=["instruction", "prompt_completion"],
                        help=f"Format of the data in the JSONL file (default: {DEFAULT_DATA_FORMAT}).")
    # Add more arguments to override defaults if needed (e.g., --num_train_epochs)
    parser.add_argument("--num_train_epochs", type=int, default=NUM_TRAIN_EPOCHS,
                        help=f"Number of training epochs (default: {NUM_TRAIN_EPOCHS}).")
    parser.add_argument("--output_dir", type=str, default=OUTPUT_DIR,
                        help=f"Output directory for checkpoints and final adapter (default: {OUTPUT_DIR}).")

    args = parser.parse_args()

    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)

    main(args.model_name, args.data_path, args.data_format) 