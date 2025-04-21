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
from utils.config_loader import load_config, DEFAULT_CONFIG_PATH # Import config loader
from omegaconf import OmegaConf, ListConfig # Import OmegaConf

# Suppress warnings
warnings.filterwarnings("ignore")
logging.set_verbosity_error() # Reduce transformers logging verbosity

# --- Configuration --- #

# Get base directory of the script (fine_tuning)
BASE_DIR = os.path.dirname(__file__)
# Project root directory (paper_population)
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))

# Model parameters
# Recommended: Start with smaller models like Mistral-7B or Llama-3-8B
DEFAULT_MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"

# Data parameters
# Default data path relative to this script's directory
# Uses the output of combine_datasets.py by default
DEFAULT_DATA_PATH = os.path.join(BASE_DIR, "finetuning_data_combined.jsonl")
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
# Default output dir relative to project root
DEFAULT_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "results_barrier_certs")
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

def main(cfg):
    """Main fine-tuning function, accepts configuration object."""

    print(f"--- Starting Fine-tuning Process ---")
    # Log relevant config sections
    print(f"Fine-tuning Config:
{OmegaConf.to_yaml(cfg.fine_tuning)}")
    print(f"Paths Config:
{OmegaConf.to_yaml(cfg.paths)}")

    # 1. Load Dataset
    data_path = cfg.paths.ft_combined_data_file
    data_format = cfg.fine_tuning.data_format
    print(f"Loading dataset from {data_path} (Format: {data_format})...")
    try:
        dataset = load_dataset('json', data_path=data_path, split='train')
        print(f"Dataset loaded: {dataset}")
        dataset = dataset.shuffle(seed=42)
        # TODO: Add train/test split based on config flag if needed
        train_dataset = dataset
        eval_dataset = None
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    # 2. Configure Quantization (QLoRA)
    compute_dtype = getattr(torch, cfg.fine_tuning.quantization.bnb_4bit_compute_dtype)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=cfg.fine_tuning.quantization.use_4bit,
        bnb_4bit_quant_type=cfg.fine_tuning.quantization.bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=cfg.fine_tuning.quantization.use_nested_quant,
    )

    # Check GPU compatibility with bfloat16
    if compute_dtype == torch.float16 and cfg.fine_tuning.quantization.use_4bit:
        major, _ = torch.cuda.get_device_capability()
        if major >= 8:
            print("=" * 80)
            print("Your GPU supports bfloat16: accelerate training with bf16=True")
            print("=" * 80)

    # 3. Load Base Model
    model_name = cfg.fine_tuning.base_model_name
    output_dir = cfg.paths.ft_output_dir # Needed for TrainingArguments
    print(f"Loading base model: {model_name}...")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map={"" : 0}, # Simple mapping, adjust if multi-GPU needed
            trust_remote_code=True # Often needed
        )
        model.config.use_cache = False
        model.config.pretraining_tp = 1
    except Exception as e:
        print(f"Error loading base model: {e}")
        return

    # 4. Load Tokenizer
    print(f"Loading tokenizer for {model_name}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            print("Set pad_token to eos_token")
        tokenizer.padding_side = "right"
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        return

    # 5. Configure LoRA
    # Convert OmegaConf ListConfig to Python list for target_modules
    lora_target_modules = list(cfg.fine_tuning.lora.target_modules)
    peft_config = LoraConfig(
        lora_alpha=cfg.fine_tuning.lora.alpha,
        lora_dropout=cfg.fine_tuning.lora.dropout,
        r=cfg.fine_tuning.lora.r,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=lora_target_modules
    )
    # Apply PEFT model wrapper
    # model = get_peft_model(model, peft_config) # Apply LoRA config - Use SFTTrainer's peft_config instead
    # print("PEFT model configured via SFTTrainer parameter.")
    # model.print_trainable_parameters()

    # 6. Configure Training Arguments
    # Convert relevant section to dict, handle specific types/defaults
    training_args_dict = OmegaConf.to_container(cfg.fine_tuning.training, resolve=True)

    # Handle device map (simple case for now)
    # device_map_setting = cfg.fine_tuning.training.get("device_map", {"": 0})

    # Handle boolean flags explicitly
    packing_setting = bool(cfg.fine_tuning.training.get("packing", False))
    group_by_length_setting = bool(cfg.fine_tuning.training.get("group_by_length", True))
    gradient_checkpointing_setting = bool(cfg.fine_tuning.training.get("gradient_checkpointing", True))

    # Determine fp16/bf16 flags based on compute dtype
    use_fp16 = False
    use_bf16 = (cfg.fine_tuning.quantization.bnb_4bit_compute_dtype == 'bfloat16')
    # Check GPU capability for bf16 again if needed
    if use_bf16 and torch.cuda.is_available() and not torch.cuda.is_bf16_supported():
        print("Warning: bf16=True but not supported by GPU. Setting bf16=False.")
        use_bf16 = False

    training_arguments = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=training_args_dict['per_device_train_batch_size'],
        gradient_accumulation_steps=training_args_dict['gradient_accumulation_steps'],
        optim=training_args_dict['optim'],
        save_steps=training_args_dict['save_steps'],
        logging_steps=training_args_dict['logging_steps'],
        learning_rate=training_args_dict['learning_rate'],
        num_train_epochs=training_args_dict['num_train_epochs'],
        weight_decay=training_args_dict['weight_decay'],
        fp16=use_fp16,
        bf16=use_bf16,
        max_grad_norm=training_args_dict['max_grad_norm'],
        max_steps=training_args_dict['max_steps'],
        warmup_ratio=training_args_dict['warmup_ratio'],
        group_by_length=group_by_length_setting,
        lr_scheduler_type=training_args_dict['lr_scheduler_type'],
        report_to="tensorboard",
        gradient_checkpointing=gradient_checkpointing_setting,
        # Add eval args if eval_dataset exists and configured
        # evaluation_strategy="steps" if eval_dataset else "no",
        # eval_steps=training_args_dict['save_steps'] if eval_dataset else None,
        # per_device_eval_batch_size=training_args_dict['per_device_eval_batch_size'] if eval_dataset else None,
    )

    # 7. Initialize SFT Trainer
    print("Initializing SFT Trainer...")
    formatting_func = None
    if data_format == "instruction":
        formatting_func = formatting_prompts_func
        print("Using instruction formatting.")
    elif data_format == "prompt_completion":
        formatting_func = formatting_prompt_completion_func
        print("Using prompt/completion formatting (will combine into single 'text' field).")

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=peft_config, # Pass LoRA config here
        formatting_func=formatting_func,
        max_seq_length=cfg.fine_tuning.training.max_seq_length if packing_setting else None,
        tokenizer=tokenizer,
        args=training_arguments,
        packing=packing_setting,
    )

    # 8. Start Training
    print("--- Starting Training ---")
    try:
        trainer.train()
        print("--- Training Finished ---")
    except Exception as e:
        print(f"Error during training: {e}")
        # Consider saving state or model here if possible
        return

    # 9. Save Final Adapter
    final_adapter_path = os.path.join(output_dir, "final_adapter")
    print(f"Saving final adapter model to {final_adapter_path}...")
    try:
        # Use save_pretrained from the trainer's model
        trainer.model.save_pretrained(final_adapter_path)
        tokenizer.save_pretrained(final_adapter_path)
        print("Adapter and tokenizer saved successfully.")
    except Exception as e:
        print(f"Error saving final adapter: {e}")

    print("\n--- Fine-tuning Script Completed ---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune a model for barrier certificate generation using QLoRA.")
    parser.add_argument("--config", type=str, default=DEFAULT_CONFIG_PATH,
                        help="Path to the configuration YAML file.")
    # Add specific CLI overrides if desired, e.g.:
    # parser.add_argument("--num_train_epochs", type=int, help="Override num_train_epochs from config.")
    args = parser.parse_args()

    # Load config
    cfg = load_config(args.config)

    # --- Handle Overrides --- Example:
    # if args.num_train_epochs is not None:
    #     print(f"Overriding num_train_epochs with CLI value: {args.num_train_epochs}")
    #     cfg.fine_tuning.training.num_train_epochs = args.num_train_epochs

    # Ensure output directory exists
    os.makedirs(cfg.paths.ft_output_dir, exist_ok=True)

    main(cfg) 