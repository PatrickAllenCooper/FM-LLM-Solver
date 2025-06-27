"""
Document Classifier for Barrier Certificate Papers

This module provides functionality to automatically classify papers as discrete,
continuous, or both based on their content. It uses keyword matching and
semantic analysis to determine the most appropriate category.
"""

import re
import logging
from typing import Dict, List, Tuple, Optional
from collections import Counter
import json

class BarrierCertificateClassifier:
    """
    Classifier for determining whether a paper focuses on discrete or continuous
    barrier certificates, or both.
    """
    
    def __init__(self, cfg):
        """
        Initialize the classifier with configuration.
        
        Parameters
        ----------
        cfg : omegaconf.dictconfig.DictConfig
            Configuration object containing classification settings
        """
        self.cfg = cfg
        self.discrete_keywords = cfg.knowledge_base.classification.discrete_keywords
        self.continuous_keywords = cfg.knowledge_base.classification.continuous_keywords
        self.stochastic_keywords = cfg.knowledge_base.classification.get('stochastic_keywords', [])
        self.confidence_threshold = cfg.knowledge_base.classification.confidence_threshold
        self.stochastic_confidence_threshold = cfg.knowledge_base.classification.stochastic_filter.get('stochastic_confidence_threshold', 0.4)
        self.min_stochastic_keywords = cfg.knowledge_base.classification.stochastic_filter.get('min_stochastic_keywords', 2)
        
        # Compile regex patterns for better performance
        self.discrete_patterns = [re.compile(r'\b' + keyword + r'\b', re.IGNORECASE) 
                                 for keyword in self.discrete_keywords]
        self.continuous_patterns = [re.compile(r'\b' + keyword + r'\b', re.IGNORECASE) 
                                   for keyword in self.continuous_keywords]
        self.stochastic_patterns = [re.compile(r'\b' + keyword + r'\b', re.IGNORECASE) 
                                   for keyword in self.stochastic_keywords]
        
        logging.info(f"Initialized classifier with {len(self.discrete_keywords)} discrete, "
                    f"{len(self.continuous_keywords)} continuous, and "
                    f"{len(self.stochastic_keywords)} stochastic keywords")
    
    def extract_text_features(self, text: str) -> Dict[str, int]:
        """
        Extract keyword counts from text.
        
        Parameters
        ----------
        text : str
            Input text to analyze
            
        Returns
        -------
        Dict[str, int]
            Dictionary containing discrete, continuous, and stochastic keyword counts
        """
        # Clean and normalize text
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)  # Remove punctuation
        text = re.sub(r'\s+', ' ', text)      # Normalize whitespace
        
        # Count discrete keywords
        discrete_count = 0
        discrete_matches = []
        for pattern in self.discrete_patterns:
            matches = pattern.findall(text)
            discrete_count += len(matches)
            discrete_matches.extend(matches)
        
        # Count continuous keywords
        continuous_count = 0
        continuous_matches = []
        for pattern in self.continuous_patterns:
            matches = pattern.findall(text)
            continuous_count += len(matches)
            continuous_matches.extend(matches)
        
        # Count stochastic keywords
        stochastic_count = 0
        stochastic_matches = []
        for pattern in self.stochastic_patterns:
            matches = pattern.findall(text)
            stochastic_count += len(matches)
            stochastic_matches.extend(matches)
        
        return {
            'discrete_count': discrete_count,
            'continuous_count': continuous_count,
            'stochastic_count': stochastic_count,
            'discrete_matches': discrete_matches,
            'continuous_matches': continuous_matches,
            'stochastic_matches': stochastic_matches,
            'total_words': len(text.split())
        }
    
    def classify_document(self, text: str, source_path: str = None) -> Tuple[str, float, Dict]:
        """
        Classify a document as discrete, continuous, or both.
        
        Parameters
        ----------
        text : str
            Document text content
        source_path : str, optional
            Path to the source document for logging
            
        Returns
        -------
        Tuple[str, float, Dict]
            Classification result (discrete/continuous/both), confidence score, and details
        """
        if not text or len(text.strip()) < 100:
            logging.warning(f"Document too short for reliable classification: {source_path}")
            return "both", 0.0, {"reason": "insufficient_text"}
        
        features = self.extract_text_features(text)
        
        discrete_count = features['discrete_count']
        continuous_count = features['continuous_count']
        total_words = features['total_words']
        
        # Calculate relative frequencies
        discrete_freq = discrete_count / max(total_words, 1)
        continuous_freq = continuous_count / max(total_words, 1)
        
        # Classification logic
        if discrete_count == 0 and continuous_count == 0:
            # No specific keywords found, classify as both to be safe
            classification = "both"
            confidence = 0.0
            reason = "no_keywords_found"
        elif discrete_count > 0 and continuous_count == 0:
            # Only discrete keywords
            classification = "discrete"
            confidence = min(discrete_freq * 100, 1.0)  # Scale frequency to confidence
        elif continuous_count > 0 and discrete_count == 0:
            # Only continuous keywords
            classification = "continuous"
            confidence = min(continuous_freq * 100, 1.0)
        else:
            # Both types of keywords present
            ratio = discrete_count / (discrete_count + continuous_count)
            
            if ratio > 0.7:
                classification = "discrete"
                confidence = ratio
            elif ratio < 0.3:
                classification = "continuous"  
                confidence = 1.0 - ratio
            else:
                # Roughly balanced, classify as both
                classification = "both"
                confidence = 1.0 - abs(0.5 - ratio) * 2  # Higher confidence when more balanced
            
            reason = f"mixed_keywords_ratio_{ratio:.2f}"
        
        # Apply confidence threshold
        if confidence < self.confidence_threshold and classification != "both":
            original_classification = classification
            classification = "both"
            reason = f"low_confidence_{confidence:.2f}_from_{original_classification}"
        
        # Add stochastic information to details
        stochastic_count = features.get('stochastic_count', 0)
        stochastic_freq = stochastic_count / max(total_words, 1)
        
        details = {
            "discrete_count": discrete_count,
            "continuous_count": continuous_count,
            "stochastic_count": stochastic_count,
            "discrete_freq": discrete_freq,
            "continuous_freq": continuous_freq,
            "stochastic_freq": stochastic_freq,
            "total_words": total_words,
            "discrete_matches": features['discrete_matches'][:10],  # Limit for logging
            "continuous_matches": features['continuous_matches'][:10],
            "stochastic_matches": features.get('stochastic_matches', [])[:10],
            "reason": reason if 'reason' in locals() else "standard_classification"
        }
        
        logging.info(f"Classified {source_path or 'document'} as '{classification}' "
                    f"(confidence: {confidence:.3f}, discrete: {discrete_count}, "
                    f"continuous: {continuous_count})")
        
        return classification, confidence, details
    
    def classify_stochastic_content(self, text: str, source_path: str = None) -> Tuple[bool, float, Dict]:
        """
        Classify a document to determine if it contains stochastic barrier certificate content.
        
        Parameters
        ----------
        text : str
            Document text content
        source_path : str, optional
            Path to the source document for logging
            
        Returns
        -------
        Tuple[bool, float, Dict]
            (is_stochastic, confidence, details) where is_stochastic indicates if document 
            contains stochastic content, confidence is the classification confidence, and 
            details contains keyword counts and matches
        """
        if not text or len(text.strip()) < 100:
            logging.warning(f"Document too short for reliable stochastic classification: {source_path}")
            return False, 0.0, {"reason": "insufficient_text"}
        
        features = self.extract_text_features(text)
        
        stochastic_count = features['stochastic_count']
        total_words = features['total_words']
        
        # Calculate relative frequency of stochastic keywords
        stochastic_freq = stochastic_count / max(total_words, 1)
        
        # Classification logic for stochastic content
        if stochastic_count >= self.min_stochastic_keywords:
            is_stochastic = True
            # Higher confidence with more keywords, scaled by frequency
            confidence = min(stochastic_freq * 50 + (stochastic_count / self.min_stochastic_keywords) * 0.5, 1.0)
        else:
            is_stochastic = False
            confidence = 1.0 - (stochastic_count / max(self.min_stochastic_keywords, 1))
        
        # Apply confidence threshold
        if confidence < self.stochastic_confidence_threshold:
            confidence = max(confidence, 0.1)  # Minimum confidence
        
        details = {
            "stochastic_count": stochastic_count,
            "stochastic_freq": stochastic_freq,
            "total_words": total_words,
            "stochastic_matches": features['stochastic_matches'][:10],  # Limit for logging
            "min_keywords_required": self.min_stochastic_keywords,
            "confidence_threshold": self.stochastic_confidence_threshold
        }
        
        logging.info(f"Stochastic classification for {source_path or 'document'}: "
                    f"is_stochastic={is_stochastic} (confidence: {confidence:.3f}, "
                    f"keywords: {stochastic_count}, required: {self.min_stochastic_keywords})")
        
        return is_stochastic, confidence, details
    
    def classify_chunks(self, chunks: List[str], source_path: str = None) -> Tuple[str, float, Dict]:
        """
        Classify a document based on multiple chunks.
        
        Parameters
        ----------
        chunks : List[str]
            List of text chunks from the document
        source_path : str, optional
            Path to the source document for logging
            
        Returns
        -------
        Tuple[str, float, Dict]
            Classification result, confidence score, and aggregated details
        """
        if not chunks:
            return "both", 0.0, {"reason": "no_chunks"}
        
        # Classify each chunk
        chunk_classifications = []
        chunk_details = []
        
        for i, chunk in enumerate(chunks):
            if len(chunk.strip()) < 50:  # Skip very short chunks
                continue
                
            classification, confidence, details = self.classify_document(
                chunk, f"{source_path}_chunk_{i}" if source_path else f"chunk_{i}"
            )
            chunk_classifications.append((classification, confidence))
            chunk_details.append(details)
        
        if not chunk_classifications:
            return "both", 0.0, {"reason": "no_valid_chunks"}
        
        # Aggregate results
        classification_counts = Counter([c[0] for c in chunk_classifications])
        
        # Weight by confidence
        weighted_scores = {"discrete": 0.0, "continuous": 0.0, "both": 0.0}
        total_weight = 0.0
        
        for classification, confidence in chunk_classifications:
            weighted_scores[classification] += confidence
            total_weight += confidence
        
        # Normalize scores
        if total_weight > 0:
            for key in weighted_scores:
                weighted_scores[key] /= total_weight
        
        # Determine final classification
        if weighted_scores["both"] > 0.5:
            final_classification = "both"
            final_confidence = weighted_scores["both"]
        elif weighted_scores["discrete"] > weighted_scores["continuous"]:
            if weighted_scores["discrete"] > 0.6:
                final_classification = "discrete"
                final_confidence = weighted_scores["discrete"]
            else:
                final_classification = "both"
                final_confidence = 1.0 - abs(weighted_scores["discrete"] - weighted_scores["continuous"])
        else:
            if weighted_scores["continuous"] > 0.6:
                final_classification = "continuous"
                final_confidence = weighted_scores["continuous"]
            else:
                final_classification = "both"
                final_confidence = 1.0 - abs(weighted_scores["discrete"] - weighted_scores["continuous"])
        
        # Aggregate details
        aggregated_details = {
            "chunk_count": len(chunk_classifications),
            "classification_counts": dict(classification_counts),
            "weighted_scores": weighted_scores,
            "total_discrete_keywords": sum(d["discrete_count"] for d in chunk_details),
            "total_continuous_keywords": sum(d["continuous_count"] for d in chunk_details),
            "total_words": sum(d["total_words"] for d in chunk_details),
            "chunk_classifications": chunk_classifications[:5]  # Sample for debugging
        }
        
        logging.info(f"Aggregated classification for {source_path or 'document'}: "
                    f"'{final_classification}' (confidence: {final_confidence:.3f}) "
                    f"from {len(chunk_classifications)} chunks")
        
        return final_classification, final_confidence, aggregated_details
    
    def should_include_document(self, text: str, source_path: str = None) -> Tuple[bool, str, Dict]:
        """
        Determine if a document should be included based on stochastic filtering configuration.
        
        Parameters
        ----------
        text : str
            Document text content
        source_path : str, optional
            Path to the source document for logging
            
        Returns
        -------
        Tuple[bool, str, Dict]
            (should_include, reason, stochastic_details) where should_include indicates if 
            document should be included, reason explains the decision, and stochastic_details 
            contains classification details
        """
        # Check if stochastic filtering is enabled
        filter_config = self.cfg.knowledge_base.classification.stochastic_filter
        if not filter_config.get('enable', False):
            return True, "Stochastic filtering disabled", {}
        
        # Classify stochastic content
        is_stochastic, confidence, stochastic_details = self.classify_stochastic_content(text, source_path)
        
        filter_mode = filter_config.get('mode', 'exclude')
        
        if filter_mode == 'exclude':
            # Exclude stochastic papers
            should_include = not is_stochastic
            reason = f"Excluded stochastic content (confidence: {confidence:.3f})" if is_stochastic else f"Included non-stochastic content (confidence: {confidence:.3f})"
        elif filter_mode == 'include':
            # Include only stochastic papers
            should_include = is_stochastic
            reason = f"Included stochastic content (confidence: {confidence:.3f})" if is_stochastic else f"Excluded non-stochastic content (confidence: {confidence:.3f})"
        else:
            logging.warning(f"Unknown stochastic filter mode: {filter_mode}. Defaulting to include all.")
            should_include = True
            reason = f"Unknown filter mode '{filter_mode}', included by default"
        
        logging.info(f"Stochastic filter decision for {source_path or 'document'}: "
                    f"include={should_include}, reason='{reason}'")
        
        return should_include, reason, stochastic_details
    
    def get_output_paths(self, classification: str) -> Tuple[str, str, str]:
        """
        Get the appropriate output paths based on classification.
        
        Parameters
        ----------
        classification : str
            Document classification (discrete/continuous/both)
            
        Returns
        -------
        Tuple[str, str, str]
            Output directory, vector store filename, metadata filename
        """
        if classification == "discrete":
            return (
                self.cfg.paths.kb_discrete_output_dir,
                self.cfg.paths.kb_discrete_vector_store_filename,
                self.cfg.paths.kb_discrete_metadata_filename
            )
        elif classification == "continuous":
            return (
                self.cfg.paths.kb_continuous_output_dir,
                self.cfg.paths.kb_continuous_vector_store_filename,
                self.cfg.paths.kb_continuous_metadata_filename
            )
        else:  # both or unified
            return (
                self.cfg.paths.kb_output_dir,
                self.cfg.paths.kb_vector_store_filename,
                self.cfg.paths.kb_metadata_filename
            )
    
    def save_classification_report(self, classifications: List[Dict], output_path: str):
        """
        Save a detailed classification report.
        
        Parameters
        ----------
        classifications : List[Dict]
            List of classification results
        output_path : str
            Path to save the report
        """
        try:
            with open(output_path, 'w') as f:
                json.dump(classifications, f, indent=2, default=str)
            logging.info(f"Saved classification report to {output_path}")
        except Exception as e:
            logging.error(f"Failed to save classification report: {e}")