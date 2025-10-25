"""
OCEAN-based training pipeline
Converts user feedback into training data and retrains the reward model
"""
import json
import time
from pathlib import Path
from src.utils import load_config, write_jsonl
from src.reward_model import train_reward
from src.generator_ollama import generate_motivation_response_ollama, check_ollama_running

def load_ocean_feedback(feedback_path="data/ocean_feedback.jsonl"):
    """Load all OCEAN feedback data"""
    if not Path(feedback_path).exists():
        return []
    with open(feedback_path) as f:
        return [json.loads(line) for line in f if line.strip()]

def generate_contrastive_response(prompt, ocean_profile, variation='opposite'):
    """
    Generate a contrastive response by modifying OCEAN dimensions

    Args:
        prompt: User's prompt
        ocean_profile: Original OCEAN profile dict
        variation: 'opposite' (invert all), 'random', or 'single_dim' (vary one dimension)

    Returns:
        tuple: (contrastive_ocean_profile, contrastive_response)
    """
    import random

    contrastive_ocean = ocean_profile.copy()

    if variation == 'opposite':
        # Invert all dimensions
        for dim in ['openness', 'conscientiousness', 'extraversion', 'agreeableness', 'neuroticism']:
            original_val = contrastive_ocean.get(dim, 0.5)
            # Invert around 0.5 midpoint
            contrastive_ocean[dim] = 1.0 - original_val

    elif variation == 'single_dim':
        # Vary a single dimension significantly
        dim_to_vary = random.choice(['openness', 'conscientiousness', 'extraversion', 'agreeableness', 'neuroticism'])
        original_val = contrastive_ocean[dim_to_vary]
        # Flip to opposite end
        contrastive_ocean[dim_to_vary] = 0.9 if original_val < 0.5 else 0.1

    elif variation == 'random':
        # Randomize all dimensions
        for dim in ['openness', 'conscientiousness', 'extraversion', 'agreeableness', 'neuroticism']:
            contrastive_ocean[dim] = random.uniform(0.1, 0.9)

    # Generate response with contrastive profile
    if check_ollama_running():
        contrastive_response = generate_motivation_response_ollama(prompt, contrastive_ocean)
    else:
        contrastive_response = "Contrastive response (Ollama not running)"

    return contrastive_ocean, contrastive_response

def convert_feedback_to_comparisons(feedback_entries, comparisons_path="data/comparisons.jsonl"):
    """
    Convert OCEAN feedback into comparison pairs for reward model training

    For each feedback entry:
    - Positive feedback: Generate a worse alternative and mark original as preferred
    - Negative feedback: Generate a better alternative and mark alternative as preferred
    """
    new_comparisons = []

    for i, entry in enumerate(feedback_entries):
        prompt = entry['prompt']
        response = entry['response']
        ocean_profile = entry['ocean_profile']
        feedback_type = entry['feedback']
        timestamp = entry.get('timestamp', time.time())

        # Generate contrastive response
        if feedback_type == 'positive':
            # Original response is good, generate a worse alternative
            contrastive_ocean, contrastive_response = generate_contrastive_response(
                prompt, ocean_profile, variation='opposite'
            )

            comparison = {
                "prompt_id": f"ocean_{int(timestamp)}_{i}",
                "a": response,  # Original (preferred)
                "b": contrastive_response,  # Contrastive (not preferred)
                "preferred": "a",
                "profile": "ocean_learned",
                "ocean_a": ocean_profile,
                "ocean_b": contrastive_ocean,
                "source": "ocean_feedback_positive"
            }

        else:  # negative feedback
            # Original response is bad, generate a better alternative
            contrastive_ocean, contrastive_response = generate_contrastive_response(
                prompt, ocean_profile, variation='opposite'
            )

            comparison = {
                "prompt_id": f"ocean_{int(timestamp)}_{i}",
                "a": contrastive_response,  # Contrastive (preferred)
                "b": response,  # Original (not preferred)
                "preferred": "a",
                "profile": "ocean_learned",
                "ocean_a": contrastive_ocean,
                "ocean_b": ocean_profile,
                "source": "ocean_feedback_negative"
            }

        new_comparisons.append(comparison)

    # Append to existing comparisons file
    Path(comparisons_path).parent.mkdir(parents=True, exist_ok=True)
    with open(comparisons_path, 'a') as f:
        for comp in new_comparisons:
            f.write(json.dumps(comp) + '\n')

    return new_comparisons

def train_from_ocean_feedback(min_feedback_count=3, auto_train=True):
    """
    Main training pipeline: Convert OCEAN feedback to comparisons and retrain reward model

    Args:
        min_feedback_count: Minimum feedback entries needed before training
        auto_train: Whether to automatically retrain the reward model

    Returns:
        dict: Training statistics
    """
    feedback_entries = load_ocean_feedback()

    if len(feedback_entries) < min_feedback_count:
        return {
            "status": "insufficient_data",
            "feedback_count": len(feedback_entries),
            "min_required": min_feedback_count,
            "message": f"Need at least {min_feedback_count} feedback entries to train. Currently have {len(feedback_entries)}."
        }

    # Check for new feedback that hasn't been processed
    processed_marker_path = "data/last_processed_feedback.json"
    last_processed_count = 0
    if Path(processed_marker_path).exists():
        with open(processed_marker_path) as f:
            marker = json.load(f)
            last_processed_count = marker.get('count', 0)

    new_feedback = feedback_entries[last_processed_count:]

    if len(new_feedback) == 0:
        return {
            "status": "no_new_data",
            "message": "All feedback has already been processed.",
            "total_feedback": len(feedback_entries),
            "processed": last_processed_count
        }

    print(f"Processing {len(new_feedback)} new feedback entries...")

    # Convert new feedback to comparison pairs
    new_comparisons = convert_feedback_to_comparisons(new_feedback)
    print(f"Generated {len(new_comparisons)} comparison pairs from feedback")

    # Update processed marker
    with open(processed_marker_path, 'w') as f:
        json.dump({
            "count": len(feedback_entries),
            "last_updated": time.time()
        }, f, indent=2)

    # Retrain reward model if requested
    training_result = None
    if auto_train:
        print("Retraining reward model with updated comparisons...")
        try:
            emb, clf = train_reward()
            training_result = {
                "status": "success",
                "message": "Reward model retrained successfully"
            }
            print("✅ Reward model retrained successfully!")
        except Exception as e:
            training_result = {
                "status": "error",
                "message": f"Training failed: {str(e)}"
            }
            print(f"❌ Training failed: {e}")

    return {
        "status": "success",
        "new_feedback_processed": len(new_feedback),
        "total_feedback": len(feedback_entries),
        "new_comparisons_generated": len(new_comparisons),
        "training_result": training_result
    }

def score_response_with_ocean(response, ocean_profile, emb=None, clf=None):
    """
    Score a response considering both reward model and OCEAN alignment

    Args:
        response: The generated response text
        ocean_profile: Target OCEAN profile dict
        emb: Sentence embedding model (optional)
        clf: Reward classifier (optional)

    Returns:
        float: Combined score (0.0-1.0)
    """
    from src.reward_model import load_reward, score_text

    if emb is None or clf is None:
        emb, clf = load_reward()

    # Get base reward score
    base_score = score_text(response, emb, clf)

    # OCEAN alignment is implicit in how the response was generated
    # The reward model learns from feedback which OCEAN profiles work
    return base_score

def get_training_stats():
    """Get statistics about the training data"""
    feedback_entries = load_ocean_feedback()

    cfg = load_config()
    comparisons_path = cfg['data_paths']['comparisons']

    comparisons = []
    if Path(comparisons_path).exists():
        with open(comparisons_path) as f:
            comparisons = [json.loads(line) for line in f if line.strip()]

    ocean_comparisons = [c for c in comparisons if c.get('source', '').startswith('ocean_feedback')]

    positive_feedback = len([f for f in feedback_entries if f.get('feedback') == 'positive'])
    negative_feedback = len([f for f in feedback_entries if f.get('feedback') == 'negative'])

    return {
        "total_feedback": len(feedback_entries),
        "positive_feedback": positive_feedback,
        "negative_feedback": negative_feedback,
        "total_comparisons": len(comparisons),
        "ocean_comparisons": len(ocean_comparisons),
        "has_reward_model": Path("models/reward/reward_clf.joblib").exists()
    }

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train reward model from OCEAN feedback")
    parser.add_argument("--min-feedback", type=int, default=3, help="Minimum feedback entries needed")
    parser.add_argument("--no-train", action="store_true", help="Generate comparisons but don't train")
    parser.add_argument("--stats", action="store_true", help="Show training statistics")

    args = parser.parse_args()

    if args.stats:
        stats = get_training_stats()
        print("\n📊 Training Statistics:")
        print(f"  Total feedback entries: {stats['total_feedback']}")
        print(f"    - Positive: {stats['positive_feedback']}")
        print(f"    - Negative: {stats['negative_feedback']}")
        print(f"  Total comparison pairs: {stats['total_comparisons']}")
        print(f"  OCEAN-generated comparisons: {stats['ocean_comparisons']}")
        print(f"  Reward model trained: {'✅ Yes' if stats['has_reward_model'] else '❌ No'}")
    else:
        result = train_from_ocean_feedback(
            min_feedback_count=args.min_feedback,
            auto_train=not args.no_train
        )
        print(f"\n📋 Training Result:")
        print(json.dumps(result, indent=2))
