"""
Create type-specific fine-tuning data for discrete and continuous barrier certificates.

This script filters the combined fine-tuning dataset based on the barrier certificate type
and creates separate datasets for discrete and continuous barrier certificates.
"""

import os
import json
import logging
import argparse
import sys
from typing import List, Dict, Any

# Add project root to Python path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
sys.path.insert(0, PROJECT_ROOT)

from utils.config_loader import load_config, DEFAULT_CONFIG_PATH
from knowledge_base.document_classifier import BarrierCertificateClassifier
from knowledge_base.kb_utils import get_ft_data_path_by_type

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_combined_dataset(file_path: str) -> List[Dict[str, Any]]:
    """
    Load the combined fine-tuning dataset.
    
    Parameters
    ----------
    file_path : str
        Path to the combined dataset file
        
    Returns
    -------
    List[Dict[str, Any]]
        List of training examples
    """
    examples = []
    if not os.path.exists(file_path):
        logging.warning(f"Combined dataset file not found: {file_path}")
        return examples
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if line:
                    try:
                        example = json.loads(line)
                        examples.append(example)
                    except json.JSONDecodeError as e:
                        logging.warning(f"Skipping invalid JSON at line {line_num}: {e}")
        
        logging.info(f"Loaded {len(examples)} examples from {file_path}")
        return examples
        
    except Exception as e:
        logging.error(f"Error loading combined dataset: {e}")
        return []

def classify_training_example(example: Dict[str, Any], classifier: BarrierCertificateClassifier) -> tuple:
    """
    Classify a training example as discrete, continuous, or both.
    
    Parameters
    ----------
    example : Dict[str, Any]
        Training example
    classifier : BarrierCertificateClassifier
        Document classifier instance
        
    Returns
    -------
    tuple
        (classification, confidence, details)
    """
    # Extract text content from different formats
    text_content = ""
    
    if 'instruction' in example and 'input' in example and 'output' in example:
        # Instruction format
        text_content = f"{example['instruction']} {example['input']} {example['output']}"
    elif 'prompt' in example and 'completion' in example:
        # Prompt-completion format
        text_content = f"{example['prompt']} {example['completion']}"
    elif 'text' in example:
        # Direct text format
        text_content = example['text']
    else:
        # Try to concatenate all string values
        text_parts = []
        for key, value in example.items():
            if isinstance(value, str):
                text_parts.append(value)
        text_content = " ".join(text_parts)
    
    if not text_content.strip():
        logging.warning("Empty text content for classification")
        return "both", 0.0, {"reason": "empty_content"}
    
    # Classify the content
    return classifier.classify_document(text_content, "training_example")

def filter_examples_by_type(examples: List[Dict[str, Any]], 
                          target_type: str, 
                          classifier: BarrierCertificateClassifier) -> List[Dict[str, Any]]:
    """
    Filter training examples by barrier certificate type.
    
    Parameters
    ----------
    examples : List[Dict[str, Any]]
        List of training examples
    target_type : str
        Target type ('discrete' or 'continuous')
    classifier : BarrierCertificateClassifier
        Document classifier instance
        
    Returns
    -------
    List[Dict[str, Any]]
        Filtered examples
    """
    filtered_examples = []
    classification_results = []
    
    for i, example in enumerate(examples):
        try:
            classification, confidence, details = classify_training_example(example, classifier)
            
            # Determine if this example should be included
            include_example = False
            if target_type == "discrete":
                include_example = classification in ["discrete", "both"]
            elif target_type == "continuous":
                include_example = classification in ["continuous", "both"]
            
            if include_example:
                # Add classification metadata to the example
                example_copy = example.copy()
                if 'metadata' not in example_copy:
                    example_copy['metadata'] = {}
                example_copy['metadata']['classification'] = classification
                example_copy['metadata']['confidence'] = confidence
                example_copy['metadata']['target_type'] = target_type
                
                filtered_examples.append(example_copy)
            
            classification_results.append({
                "example_index": i,
                "classification": classification,
                "confidence": confidence,
                "details": details,
                "included": include_example,
                "target_type": target_type
            })
            
            if i % 10 == 0:
                logging.info(f"Processed {i+1}/{len(examples)} examples for {target_type} type")
                
        except Exception as e:
            logging.error(f"Error classifying example {i}: {e}")
            continue
    
    logging.info(f"Filtered {len(filtered_examples)} examples for {target_type} type from {len(examples)} total")
    return filtered_examples, classification_results

def save_filtered_dataset(examples: List[Dict[str, Any]], 
                         output_path: str, 
                         target_type: str) -> bool:
    """
    Save filtered dataset to file.
    
    Parameters
    ----------
    examples : List[Dict[str, Any]]
        Filtered examples
    output_path : str
        Output file path
    target_type : str
        Target type for logging
        
    Returns
    -------
    bool
        True if successful, False otherwise
    """
    try:
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for example in examples:
                f.write(json.dumps(example) + '\n')
        
        logging.info(f"Saved {len(examples)} {target_type} examples to {output_path}")
        return True
        
    except Exception as e:
        logging.error(f"Error saving {target_type} dataset: {e}")
        return False

def save_classification_report(results: List[Dict[str, Any]], 
                             output_path: str) -> bool:
    """
    Save classification report to file.
    
    Parameters
    ----------
    results : List[Dict[str, Any]]
        Classification results
    output_path : str
        Output file path
        
    Returns
    -------
    bool
        True if successful, False otherwise
    """
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, default=str)
        
        logging.info(f"Saved classification report to {output_path}")
        return True
        
    except Exception as e:
        logging.error(f"Error saving classification report: {e}")
        return False

def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Create type-specific fine-tuning datasets for discrete and continuous barrier certificates."
    )
    parser.add_argument("--config", type=str, default=DEFAULT_CONFIG_PATH,
                        help="Path to the configuration YAML file.")
    parser.add_argument("--input-file", type=str, default=None,
                        help="Override path to combined fine-tuning data file.")
    parser.add_argument("--types", nargs='+', choices=["discrete", "continuous"], 
                        default=["discrete", "continuous"],
                        help="Types of datasets to create.")
    parser.add_argument("--force", action="store_true",
                        help="Overwrite existing output files.")
    
    args = parser.parse_args()
    
    # Load configuration
    cfg = load_config(args.config)
    
    # Determine input file
    input_file = args.input_file if args.input_file else cfg.paths.ft_combined_data_file
    
    if not os.path.exists(input_file):
        logging.error(f"Input file not found: {input_file}")
        logging.error("Please create combined fine-tuning data first using combine_datasets.py")
        return False
    
    # Initialize classifier
    try:
        classifier = BarrierCertificateClassifier(cfg)
        logging.info("Document classifier initialized successfully")
    except Exception as e:
        logging.error(f"Failed to initialize document classifier: {e}")
        return False
    
    # Load combined dataset
    examples = load_combined_dataset(input_file)
    if not examples:
        logging.error("No examples loaded from input file")
        return False
    
    # Process each requested type
    success = True
    for target_type in args.types:
        logging.info(f"Processing {target_type} barrier certificate examples...")
        
        # Get output path
        output_path = get_ft_data_path_by_type(cfg, target_type)
        
        # Check if output file exists
        if os.path.exists(output_path) and not args.force:
            logging.warning(f"Output file exists: {output_path}")
            logging.warning("Use --force to overwrite or remove the file manually")
            continue
        
        # Filter examples
        filtered_examples, classification_results = filter_examples_by_type(
            examples, target_type, classifier
        )
        
        if not filtered_examples:
            logging.warning(f"No examples found for {target_type} type")
            continue
        
        # Save filtered dataset
        if not save_filtered_dataset(filtered_examples, output_path, target_type):
            success = False
            continue
        
        # Save classification report
        report_path = output_path.replace('.jsonl', '_classification_report.json')
        save_classification_report(classification_results, report_path)
        
        logging.info(f"Successfully created {target_type} dataset with {len(filtered_examples)} examples")
    
    if success:
        logging.info("Type-specific dataset creation completed successfully")
        logging.info("Next steps:")
        logging.info("1. Review the created datasets")
        logging.info("2. Update config to use the appropriate barrier_certificate_type")
        logging.info("3. Run fine-tuning with the type-specific dataset")
    else:
        logging.error("Some datasets could not be created")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)