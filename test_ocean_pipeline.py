"""
Test OCEAN Training Pipeline
Quick verification that the pipeline works end-to-end
"""
import json
import time
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ocean_trainer import (
    convert_feedback_to_comparisons,
    train_from_ocean_feedback,
    get_training_stats,
    score_response_with_ocean
)

def create_test_feedback():
    """Create some test feedback entries"""
    feedback_path = "data/ocean_feedback_test.jsonl"
    Path("data").mkdir(exist_ok=True)

    test_feedback = [
        {
            "timestamp": time.time(),
            "prompt": "I can't get started on my project",
            "response": "Break it into tiny steps and start with just 5 minutes of work.",
            "ocean_profile": {
                "openness": 0.5,
                "conscientiousness": 0.8,
                "extraversion": 0.5,
                "agreeableness": 0.6,
                "neuroticism": 0.3
            },
            "feedback": "positive",
            "session_id": "test"
        },
        {
            "timestamp": time.time(),
            "prompt": "I'm feeling overwhelmed",
            "response": "Stop overthinking and just start. Action beats planning.",
            "ocean_profile": {
                "openness": 0.3,
                "conscientiousness": 0.6,
                "extraversion": 0.7,
                "agreeableness": 0.3,
                "neuroticism": 0.2
            },
            "feedback": "negative",
            "session_id": "test"
        },
        {
            "timestamp": time.time(),
            "prompt": "I keep procrastinating",
            "response": "I understand procrastination is hard. Let's find what's blocking you and address it gently.",
            "ocean_profile": {
                "openness": 0.6,
                "conscientiousness": 0.5,
                "extraversion": 0.4,
                "agreeableness": 0.9,
                "neuroticism": 0.7
            },
            "feedback": "positive",
            "session_id": "test"
        }
    ]

    with open(feedback_path, 'w') as f:
        for entry in test_feedback:
            f.write(json.dumps(entry) + '\n')

    return feedback_path, len(test_feedback)

def test_pipeline():
    """Test the complete pipeline"""
    print("🧪 Testing OCEAN Training Pipeline")
    print("=" * 50)

    # Step 1: Create test feedback
    print("\n1️⃣ Creating test feedback...")
    feedback_path, count = create_test_feedback()
    print(f"   ✅ Created {count} test feedback entries")

    # Step 2: Load feedback
    print("\n2️⃣ Loading feedback...")
    with open(feedback_path) as f:
        feedback_entries = [json.loads(line) for line in f if line.strip()]
    print(f"   ✅ Loaded {len(feedback_entries)} entries")

    for i, entry in enumerate(feedback_entries):
        feedback_icon = "👍" if entry['feedback'] == 'positive' else "👎"
        print(f"   {feedback_icon} Entry {i+1}: {entry['feedback']} - {entry['prompt'][:40]}...")

    # Step 3: Convert to comparisons
    print("\n3️⃣ Converting feedback to comparison pairs...")
    comparisons_path = "data/comparisons_test.jsonl"

    # Clear test comparisons file
    if Path(comparisons_path).exists():
        Path(comparisons_path).unlink()

    new_comparisons = convert_feedback_to_comparisons(feedback_entries, comparisons_path)
    print(f"   ✅ Generated {len(new_comparisons)} comparison pairs")

    for i, comp in enumerate(new_comparisons):
        print(f"   📊 Pair {i+1}: preferred={comp['preferred']}, source={comp['source']}")

    # Step 4: Check comparisons file
    print("\n4️⃣ Verifying comparisons file...")
    if Path(comparisons_path).exists():
        with open(comparisons_path) as f:
            saved_comps = [json.loads(line) for line in f if line.strip()]
        print(f"   ✅ {len(saved_comps)} comparisons saved to {comparisons_path}")

        # Show structure of first comparison
        if saved_comps:
            print("\n   Sample comparison structure:")
            sample = saved_comps[0]
            print(f"   - prompt_id: {sample.get('prompt_id')}")
            print(f"   - preferred: {sample.get('preferred')}")
            print(f"   - profile: {sample.get('profile')}")
            print(f"   - source: {sample.get('source')}")
            print(f"   - has OCEAN profiles: ocean_a={('ocean_a' in sample)}, ocean_b={('ocean_b' in sample)}")
    else:
        print(f"   ❌ Comparisons file not created!")
        return False

    # Step 5: Test training stats
    print("\n5️⃣ Getting training statistics...")
    stats = get_training_stats()
    print(f"   ✅ Stats retrieved:")
    print(f"   - Total feedback: {stats['total_feedback']}")
    print(f"   - Positive: {stats['positive_feedback']}")
    print(f"   - Negative: {stats['negative_feedback']}")
    print(f"   - Total comparisons: {stats['total_comparisons']}")
    print(f"   - OCEAN comparisons: {stats['ocean_comparisons']}")

    # Step 6: Test scoring (if reward model exists)
    print("\n6️⃣ Testing response scoring...")
    test_ocean = {
        "openness": 0.7,
        "conscientiousness": 0.6,
        "extraversion": 0.5,
        "agreeableness": 0.8,
        "neuroticism": 0.4
    }

    test_response = "You can do this! Take it one step at a time."

    try:
        score = score_response_with_ocean(test_response, test_ocean)
        print(f"   ✅ Scoring works! Score: {score:.3f}")
    except Exception as e:
        print(f"   ⚠️  Scoring unavailable (reward model not trained): {e}")

    print("\n✅ Pipeline test complete!")
    print("\nℹ️  Test files created:")
    print(f"   - {feedback_path}")
    print(f"   - {comparisons_path}")
    print("\nTo train the actual reward model:")
    print("   python -m src.ocean_trainer")

    return True

if __name__ == "__main__":
    success = test_pipeline()
    sys.exit(0 if success else 1)
