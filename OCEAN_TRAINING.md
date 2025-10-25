# OCEAN Training Pipeline

## Overview

The OCEAN Training Pipeline enables the system to learn your personal communication preferences by training a reward model based on your feedback to responses generated with different OCEAN personality profiles.

## How It Works

### The 5-Dimensional OCEAN Model

The system uses the Big Five (OCEAN) personality dimensions to generate personalized motivation:

1. **O**penness - Creativity vs. Practicality
2. **C**onscientiousness - Structure vs. Flexibility
3. **E**xtraversion - Energetic vs. Calm
4. **A**greeableness - Warm vs. Direct
5. **N**euroticism - Reassuring vs. Confident

Each dimension is a score from 0.0 to 1.0.

### The Training Loop

```
1. GENERATE → System creates responses using OCEAN profiles
                ↓
2. FEEDBACK → You rate responses (👍/👎)
                ↓
3. ADJUST → Your learned OCEAN profile updates
                ↓
4. CREATE PAIRS → System generates contrastive responses
                ↓
5. TRAIN → Reward model learns your preferences
                ↓
6. IMPROVE → Future responses better match your style
```

## Using the Pipeline

### In the Web App

1. **Chat Mode** - Interact with different OCEAN profiles
2. **Give Feedback** - Click 👍 or 👎 on responses
3. **Training Tab** - Click "Train from OCEAN Feedback" when prompted
4. **Your Learned Profile** - View in sidebar, automatically improves

### Command Line Training

```bash
# Show training statistics
python -m src.ocean_trainer --stats

# Train from feedback (auto-train)
python -m src.ocean_trainer

# Generate comparisons only (no training)
python -m src.ocean_trainer --no-train

# Or use the script
./scripts/train_ocean.sh
```

## How Training Works

### Step 1: Collect Feedback
When you give feedback (👍/👎), the system saves:
- Your prompt
- The response generated
- The OCEAN profile used
- Your feedback (positive/negative)

This is stored in `data/ocean_feedback.jsonl`.

### Step 2: Generate Contrastive Pairs
For each feedback entry, the system creates a comparison pair:

**Positive Feedback (👍):**
- Original response (OCEAN: O=0.9, C=0.5, E=0.7, A=0.8, N=0.3)
- Contrastive response (OCEAN: O=0.1, C=0.5, E=0.3, A=0.2, N=0.7)
- Mark: Original is preferred

**Negative Feedback (👎):**
- Contrastive response (opposite OCEAN profile)
- Original response
- Mark: Contrastive is preferred

These pairs are appended to `data/comparisons.jsonl`.

### Step 3: Train Reward Model
The reward model uses:
- **Sentence embeddings** - Converts text to numerical vectors
- **Logistic Regression** - Learns which responses you prefer
- **Difference encoding** - Compares response pairs

Training happens when you have ≥3 feedback entries.

### Step 4: Score Future Responses
Once trained, the reward model can:
- Score any generated response (0.0-1.0)
- Rank multiple candidates
- Guide which OCEAN profiles work best for you

## Your Learned OCEAN Profile

The system maintains a personalized OCEAN profile that evolves with feedback:

```python
{
    "openness": 0.75,           # You prefer creative approaches
    "conscientiousness": 0.60,   # Moderate structure
    "extraversion": 0.45,        # Calm, reflective tone
    "agreeableness": 0.85,       # Warm, supportive style
    "neuroticism": 0.70,         # Reassuring, anxiety-aware
    "feedback_count": 12         # Based on 12 interactions
}
```

### Profile Updates

- **Positive feedback**: Moves toward the working profile (learning rate: 0.15)
- **Negative feedback**: Moves away from the non-working profile (learning rate: 0.10)
- **Bounded**: All values stay between 0.0 and 1.0

## Pipeline Files

### Data Files
- `data/ocean_feedback.jsonl` - Raw feedback from users
- `data/comparisons.jsonl` - Training pairs (original + OCEAN-generated)
- `data/user_ocean_profile.json` - Your learned OCEAN profile
- `data/last_processed_feedback.json` - Training checkpoint

### Model Files
- `models/reward/reward_clf.joblib` - Trained reward classifier

### Code Files
- `src/ocean_trainer.py` - Training pipeline implementation
- `src/reward_model.py` - Reward model training & inference
- `src/generator_ollama.py` - OCEAN-based response generation
- `src/demo_app_enhanced.py` - Web interface

## Advanced Usage

### Manual Training Control

```python
from src.ocean_trainer import train_from_ocean_feedback

# Train with custom settings
result = train_from_ocean_feedback(
    min_feedback_count=5,  # Require more feedback
    auto_train=True        # Automatically retrain model
)

print(result)
```

### Check Training Stats

```python
from src.ocean_trainer import get_training_stats

stats = get_training_stats()
print(f"Total feedback: {stats['total_feedback']}")
print(f"Training pairs: {stats['total_comparisons']}")
print(f"Model trained: {stats['has_reward_model']}")
```

### Score Responses

```python
from src.ocean_trainer import score_response_with_ocean

ocean_profile = {
    "openness": 0.8,
    "conscientiousness": 0.6,
    "extraversion": 0.5,
    "agreeableness": 0.7,
    "neuroticism": 0.4
}

score = score_response_with_ocean(
    "You've got this! Break it into small steps.",
    ocean_profile
)
print(f"Score: {score:.3f}")
```

## Benefits

1. **Personalized Learning** - Adapts to YOUR communication style
2. **Scientific Foundation** - Based on validated Big Five psychology
3. **Continuous Improvement** - Gets better with more feedback
4. **Transparent Process** - You can see how your profile evolves
5. **Flexible Control** - Manual or automatic training

## Training Tips

- **Give varied feedback** - Try different OCEAN profiles
- **Be consistent** - Rate similar styles similarly
- **Accumulate data** - More feedback = better learning
- **Retrain regularly** - Every 5-10 feedback entries
- **Check your profile** - See how your preferences evolve

## Troubleshooting

**Q: Training says "insufficient data"**
- Need at least 3 feedback entries
- Keep giving 👍/👎 feedback in Chat Mode

**Q: Reward model not updating responses**
- Make sure you clicked "Train from OCEAN Feedback"
- Check Training Dashboard for status

**Q: Ollama not generating contrastive responses**
- Start Ollama: `ollama run mistral`
- System will generate placeholders if Ollama unavailable

**Q: Want to reset and start over?**
- Delete `data/ocean_feedback.jsonl`
- Delete `data/user_ocean_profile.json`
- Click "Reset Learned Profile" in sidebar

## Next Steps

1. Give feedback on at least 5 responses
2. Train the reward model
3. Use "Your Learned Profile" preset in Chat Mode
4. Continue giving feedback to refine
5. Watch your OCEAN profile evolve!

## Technical Details

**Reward Model Architecture:**
- Embeddings: `sentence-transformers/all-MiniLM-L6-v2` (384 dim)
- Classifier: Logistic Regression (L2 regularization)
- Training: Pairwise preference learning
- Input: Embedding difference vector (response_a - response_b)
- Output: Probability that response_a is preferred

**OCEAN Profile Learning:**
- Update method: Gradient-based adjustment
- Positive learning rate: 0.15
- Negative learning rate: 0.10
- Constraints: [0.0, 1.0] bounds per dimension
- Initialization: Balanced (0.5 for all dimensions)
