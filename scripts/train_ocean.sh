#!/bin/bash
# Train the reward model from OCEAN feedback

echo "🧠 OCEAN Training Pipeline"
echo "=========================="
echo ""

# Check if we're in the right directory
if [ ! -f "src/ocean_trainer.py" ]; then
    echo "❌ Error: Must run from project root directory"
    exit 1
fi

# Show stats first
echo "📊 Current Training Statistics:"
python -m src.ocean_trainer --stats
echo ""

# Ask user to proceed
echo "🔄 Ready to train from OCEAN feedback?"
echo "   This will:"
echo "   1. Convert feedback to comparison pairs"
echo "   2. Generate contrastive responses"
echo "   3. Retrain the reward model"
echo ""
read -p "Continue? (y/n) " -n 1 -r
echo ""

if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "🚀 Starting training..."
    python -m src.ocean_trainer

    if [ $? -eq 0 ]; then
        echo ""
        echo "✅ Training complete!"
        echo "📊 Updated statistics:"
        python -m src.ocean_trainer --stats
    else
        echo ""
        echo "❌ Training failed. Check errors above."
        exit 1
    fi
else
    echo "⏸️  Training cancelled."
    exit 0
fi
