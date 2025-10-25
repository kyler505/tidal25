# MotivateMe AI

A personalized AI motivation assistant that learns your unique personality using OCEAN (Big Five) personality dimensions and adapts its responses to match your communication preferences.

## Features

- 🎯 **Personalized Motivation**: Generate motivational responses tailored to your personality
- 🧠 **OCEAN Personality Learning**: Automatically learns your Big Five personality traits through feedback
- 📊 **Interactive Profile Building**: Compare different response styles to refine your profile
- 🎨 **Modern UI**: Clean, dark-themed interface built with Streamlit
- 🔄 **Continuous Learning**: Your profile improves with every interaction

## Quick Start

1. **Install dependencies**:
```bash
pip install -r requirements.txt
```

2. **Set up your API key**:
   - Get a Gemini API key from [Google AI Studio](https://makersuite.google.com/app/apikey)
   - Add to `config/settings.yaml`:
```yaml
gemini_api_key: "your-api-key-here"
```

3. **Run the app**:
```bash
streamlit run src/demo_app_enhanced.py
```

## How It Works

### OCEAN Personality Model
The system uses the Big Five (OCEAN) personality dimensions:
- **Openness**: Creativity and openness to new experiences
- **Conscientiousness**: Organization and goal-directed behavior
- **Extraversion**: Social energy and enthusiasm
- **Agreeableness**: Empathy and cooperation
- **Neuroticism**: Emotional sensitivity and stress response

### Learning Process
1. Start with a balanced profile (0.5 for each dimension)
2. Provide feedback on generated responses (helpful/not helpful)
3. Compare different response styles in "Build Your Profile"
4. Your profile adjusts automatically based on preferences
5. Future responses are personalized to your learned profile

## Project Structure

```
├── config/
│   └── settings.yaml           # Configuration and API keys
├── data/
│   ├── ocean_feedback.jsonl    # Feedback history
│   └── user_ocean_profile.json # Your learned personality profile
├── src/
│   ├── demo_app_enhanced.py    # Main Streamlit app
│   ├── generator_gemini.py     # Gemini API integration
│   ├── ocean_trainer.py        # OCEAN learning system
│   ├── reward_model.py         # Reward model training
│   └── utils.py                # Utility functions
├── scripts/
│   └── train_ocean.sh          # Training script
└── test_ocean_pipeline.py      # Pipeline tests
```

## Documentation

- [OCEAN Training Guide](OCEAN_TRAINING.md) - Detailed information about the personality learning system
- [Generation Guide](GENERATION_README.md) - Response generation documentation
- [Quick Start](QUICKSTART.md) - Getting started guide

## Technologies

- **Streamlit**: Web interface
- **Google Gemini**: LLM for response generation
- **scikit-learn**: Machine learning for reward models
- **sentence-transformers**: Text embeddings

## License

MIT License

