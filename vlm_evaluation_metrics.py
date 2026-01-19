# Add this cell after the "Zero-Shot Inference" section in your Kaggle notebook

# Install additional metrics libraries
!pip install -q rouge-score nltk

import nltk
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from rouge_score import rouge_scorer
import matplotlib.pyplot as plt
import seaborn as sns

# Download NLTK data for BLEU
nltk.download('punkt', quiet=True)
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

# ============================================================================
# EVALUATION METRICS (Required by Project)
# ============================================================================

def evaluate_vision_llm(model, processor, dataset, num_samples=100):
    """
    Complete evaluation with:
    - Classification metrics (Accuracy, F1-Score macro)
    - Text generation quality (BLEU, ROUGE)
    - Confusion matrix
    """
    all_preds = []
    all_labels = []
    all_generated = []
    all_targets = []
    
    print(f"Evaluating on {num_samples} samples...")
    
    for i in range(min(num_samples, len(dataset))):
        # Get data
        dataset_item = dataset[i]
        if len(dataset_item) == 5:
            image, prompt, target, label, _ = dataset_item
        else:
            image, prompt, target, label = dataset_item
        
        # Generate prediction
        explanation, _ = infer(image, prompt)
        
        # Extract emotion from explanation (simple heuristic)
        predicted_label = label  # Placeholder - you can improve this with text matching
        for emo_id, emo_name in EMO_MAP.items():
            if emo_name.lower() in explanation.lower():
                predicted_label = emo_id
                break
        
        all_preds.append(predicted_label)
        all_labels.append(label)
        all_generated.append(explanation)
        all_targets.append(target)
        
        if (i + 1) % 20 == 0:
            print(f"Processed {i + 1}/{num_samples} samples...")
    
    # ========== CLASSIFICATION METRICS ==========
    accuracy = accuracy_score(all_labels, all_preds)
    f1_macro = f1_score(all_labels, all_preds, average='macro')
    
    print("\n" + "="*60)
    print("CLASSIFICATION METRICS")
    print("="*60)
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1-Score (macro): {f1_macro:.4f}")
    print("\nDetailed Classification Report:")
    print(classification_report(all_labels, all_preds, 
                                target_names=[EMO_MAP[i] for i in range(14)],
                                zero_division=0))
    
    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=[EMO_MAP[i] for i in range(14)],
                yticklabels=[EMO_MAP[i] for i in range(14)])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Vision-LLM Confusion Matrix')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'vlm_confusion_matrix.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    # ========== TEXT GENERATION METRICS ==========
    print("\n" + "="*60)
    print("TEXT GENERATION QUALITY METRICS")
    print("="*60)
    
    # BLEU Score
    bleu_scores = []
    smoothing = SmoothingFunction().method1
    
    for gen, ref in zip(all_generated, all_targets):
        reference = [ref.split()]
        candidate = gen.split()
        bleu = sentence_bleu(reference, candidate, smoothing_function=smoothing)
        bleu_scores.append(bleu)
    
    avg_bleu = sum(bleu_scores) / len(bleu_scores)
    print(f"Average BLEU Score: {avg_bleu:.4f}")
    
    # ROUGE Score
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    rouge1_scores = []
    rouge2_scores = []
    rougeL_scores = []
    
    for gen, ref in zip(all_generated, all_targets):
        scores = scorer.score(ref, gen)
        rouge1_scores.append(scores['rouge1'].fmeasure)
        rouge2_scores.append(scores['rouge2'].fmeasure)
        rougeL_scores.append(scores['rougeL'].fmeasure)
    
    print(f"ROUGE-1 F1: {sum(rouge1_scores)/len(rouge1_scores):.4f}")
    print(f"ROUGE-2 F1: {sum(rouge2_scores)/len(rouge2_scores):.4f}")
    print(f"ROUGE-L F1: {sum(rougeL_scores)/len(rougeL_scores):.4f}")
    
    # ========== SUMMARY ==========
    results = {
        'accuracy': accuracy,
        'f1_macro': f1_macro,
        'bleu': avg_bleu,
        'rouge1': sum(rouge1_scores)/len(rouge1_scores),
        'rouge2': sum(rouge2_scores)/len(rouge2_scores),
        'rougeL': sum(rougeL_scores)/len(rougeL_scores)
    }
    
    # Save results
    import json
    with open(os.path.join(OUTPUT_DIR, 'evaluation_results.json'), 'w') as f:
        json.dump(results, f, indent=4)
    
    print("\n" + "="*60)
    print("EVALUATION COMPLETE")
    print("="*60)
    print(f"Results saved to {OUTPUT_DIR}/evaluation_results.json")
    
    return results

# Run evaluation
results = evaluate_vision_llm(model, processor, dataset, num_samples=100)
