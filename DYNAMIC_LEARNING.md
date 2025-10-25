# Dynamic Learning Rate System

## Overview

The MotivateMe AI now uses a **dynamic learning rate** that automatically adjusts based on the number of feedback responses you've provided. This accelerates the initial learning process while providing stability as your profile matures.

## How It Works

### Exponential Decay Formula
```
learning_rate = base_rate × (decay_factor ^ (feedback_count / 10))
learning_rate = max(min_rate, calculated_rate)
```

**Parameters:**
- `base_rate`: 0.25 (25%) - Maximum adjustment for first responses
- `min_rate`: 0.05 (5%) - Minimum adjustment to maintain adaptability
- `decay_factor`: 0.85 - Controls speed of decay

### Learning Phases

| Feedback Count | Learning Rate | Phase | Behavior |
|---------------|---------------|-------|----------|
| 0-10 | 21-25% | **Early Learning** 🔥 | High impact - Rapid profile formation |
| 11-20 | 18-21% | **Building** ⚡ | Strong adjustments - Profile taking shape |
| 21-40 | 13-18% | **Refinement** 📊 | Moderate changes - Fine-tuning preferences |
| 41-60 | 9-13% | **Stabilization** 📉 | Smaller adjustments - Profile maturing |
| 60+ | 5-9% | **Maintenance** 🎯 | Minimal tweaks - Preserving learned preferences |

## Benefits

### 🚀 Faster Initial Learning
- First 10 responses have 4-5x more impact than later responses
- Your profile converges quickly to your actual preferences
- Reduced time to personalization

### 🎯 Long-term Stability
- Prevents profile drift after it's well-established
- Late responses fine-tune rather than overhaul
- Maintains consistency in personalized responses

### 📈 Natural Learning Curve
- Mimics human learning patterns
- Early experiences shape foundation
- Later experiences refine details

## Example Impact

```
Response #1:  Adjustment = 25% → Profile shifts significantly
Response #10: Adjustment = 21% → Still learning your style
Response #30: Adjustment = 15% → Refining preferences
Response #50: Adjustment = 11% → Minor adjustments
Response #100: Adjustment = 5%  → Fine-tuning only
```

## Visual Learning Rate

The app now displays your current learning rate in the "Your Progress" tab, showing:
- Current adjustment strength percentage
- Number of responses collected
- Learning phase indicator
- Progress bar visualization

## Technical Details

### Positive vs Negative Feedback
- **Positive feedback**: Uses full learning rate (moves toward preferred style)
- **Negative feedback**: Uses 50% of learning rate (moves away more cautiously)

### Profile Updates
Each OCEAN dimension is adjusted independently:
```python
adjustment = (target_value - current_value) × learning_rate
new_value = current_value + adjustment
```

### Stored Metadata
Your profile now includes:
- `feedback_count`: Total responses
- `last_learning_rate`: Rate used in last update
- `last_updated`: Timestamp of last change

## Testing

Run the test script to see the learning rate curve:
```bash
python3 test_dynamic_learning.py
```

This will show how the learning rate decreases over time and explain the impact at each stage.
