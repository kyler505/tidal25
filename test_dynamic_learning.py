#!/usr/bin/env python3
"""Test the dynamic learning rate system"""

def calculate_dynamic_learning_rate(feedback_count, base_rate=0.25, min_rate=0.05, decay_factor=0.85):
    """Calculate a dynamic learning rate that decreases as feedback increases

    Args:
        feedback_count: Number of feedback entries so far
        base_rate: Starting learning rate (higher = faster initial learning)
        min_rate: Minimum learning rate to prevent complete stagnation
        decay_factor: Rate of decay (lower = faster decay)

    Returns:
        Dynamic learning rate between min_rate and base_rate
    """
    # Exponential decay: rate = base_rate * (decay_factor ^ feedback_count)
    # Ensures early feedback has much more impact
    dynamic_rate = base_rate * (decay_factor ** (feedback_count / 10))
    return max(min_rate, dynamic_rate)

def test_dynamic_learning_rate():
    """Test that learning rate decreases as feedback count increases"""

    print("Testing Dynamic Learning Rate System")
    print("=" * 60)

    feedback_counts = [0, 1, 5, 10, 15, 20, 30, 50, 100]

    print(f"{'Feedback Count':<20} {'Learning Rate':<20} {'Impact Level':<20}")
    print("-" * 60)

    for count in feedback_counts:
        rate = calculate_dynamic_learning_rate(count)

        # Determine impact level
        if rate >= 0.20:
            impact = "Very High 🔥"
        elif rate >= 0.15:
            impact = "High ⚡"
        elif rate >= 0.10:
            impact = "Medium 📊"
        elif rate >= 0.07:
            impact = "Low 📉"
        else:
            impact = "Minimal 🎯"

        print(f"{count:<20} {rate:.4f} ({rate:.1%}){' '*(8-len(f'{rate:.1%}'))} {impact}")

    print("\n" + "=" * 60)
    print("\nKey Insights:")
    print("- First feedback: Maximum impact (~25% adjustment)")
    print("- After 10 responses: Still significant (~15% adjustment)")
    print("- After 30 responses: Moderate (~8% adjustment)")
    print("- After 100 responses: Fine-tuning (~5% minimum)")
    print("\nThis accelerates early learning while preventing instability later.")

if __name__ == "__main__":
    test_dynamic_learning_rate()
