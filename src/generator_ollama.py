"""
Ollama-based text generation for MotivateMe AI
Much faster and simpler than loading models directly!
"""
import requests
import json

OLLAMA_URL = "http://localhost:11434/api/generate"

def check_ollama_running():
    """Check if Ollama server is running"""
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=2)
        return response.status_code == 200
    except:
        return False

def generate_motivation_response_ollama(prompt, ocean_profile=None, model="mistral", max_length=100):
    """
    Generate motivation response using Ollama with OCEAN personality model

    Args:
        prompt: User's concern/situation
        ocean_profile: dict with OCEAN scores (0.0-1.0), e.g. {"openness": 0.7, "conscientiousness": 0.8, ...}
                      If None, uses balanced default (all dimensions = 0.5)
        model: Ollama model name (e.g., "mistral", "llama2")
        max_length: Max tokens to generate

    Returns:
        Generated response text
    """
    # Check if Ollama is running
    if not check_ollama_running():
        return "Error: Ollama is not running. Please start it with 'ollama serve' or 'ollama run mistral'"

    def _build_ocean_instruction(ocean):
        """Build system instruction from OCEAN personality scores (0.0-1.0)"""
        parts = []

        # Openness to Experience (creativity, curiosity)
        o = ocean.get('openness', ocean.get('O', 0.5))
        if o >= 0.7:
            parts.append("Be highly creative and unconventional. Suggest unique, out-of-the-box ideas and experimental approaches.")
        elif o <= 0.3:
            parts.append("Be practical and conventional. Focus only on tried-and-true methods that are proven to work.")

        # Conscientiousness (organization, discipline)
        c = ocean.get('conscientiousness', ocean.get('C', 0.5))
        if c >= 0.7:
            parts.append("Be very structured and detailed. Emphasize planning, schedules, organization, and step-by-step systems.")
        elif c <= 0.3:
            parts.append("Be casual and spontaneous. Avoid rigid plans and encourage going with the flow.")

        # Extraversion (social energy, assertiveness)
        e = ocean.get('extraversion', ocean.get('E', 0.5))
        if e >= 0.7:
            parts.append("Be energetic and social. Emphasize teamwork, talking to others, and taking bold action.")
        elif e <= 0.3:
            parts.append("Be calm and introspective. Focus on quiet reflection, alone time, and inner work.")

        # Agreeableness (empathy, cooperation)
        a = ocean.get('agreeableness', ocean.get('A', 0.5))
        if a >= 0.7:
            parts.append("Be extremely warm and supportive. Show empathy, validate feelings, and be encouraging.")
        elif a <= 0.3:
            parts.append("Be blunt and direct. Give tough love and focus on facts over feelings.")

        # Neuroticism (emotional stability, anxiety)
        n = ocean.get('neuroticism', ocean.get('N', 0.5))
        if n >= 0.7:
            parts.append("Acknowledge stress and anxiety. Provide reassurance and calming strategies.")
        elif n <= 0.3:
            parts.append("Be confident and bold. Ignore worry and push for resilience.")

        if not parts:
            parts.append("Provide balanced, helpful motivational advice.")

        instruction = "You are a motivational coach. " + " ".join(parts) + " Make your response VERY DIFFERENT from other approaches. Keep response to 1-2 short sentences. Be concise and general."
        return instruction

    # Use default balanced profile if none provided
    if ocean_profile is None:
        ocean_profile = {
            "openness": 0.5,
            "conscientiousness": 0.5,
            "extraversion": 0.5,
            "agreeableness": 0.5,
            "neuroticism": 0.5
        }

    # Create styled prompt based on OCEAN profile
    system_instruction = _build_ocean_instruction(ocean_profile)
    full_prompt = f"{system_instruction}\n\nPerson: {prompt}\n\nCoach:"

    # Call Ollama API
    try:
        payload = {
            "model": model,
            "prompt": full_prompt,
            "stream": False,
            "options": {
                "temperature": 0.8,
                "top_p": 0.9,
                "top_k": 40,
                "num_predict": max_length
            }
        }

        response = requests.post(OLLAMA_URL, json=payload, timeout=30)

        if response.status_code == 200:
            result = response.json()
            generated_text = result.get('response', '').strip()

            # Clean up response
            if '\n' in generated_text:
                # Take first paragraph
                generated_text = generated_text.split('\n')[0].strip()

            # Limit to first 2 sentences for conciseness
            sentences = generated_text.split('. ')
            if len(sentences) > 2:
                generated_text = '. '.join(sentences[:2]) + '.'

            return generated_text if generated_text else "Unable to generate response. Please try again."
        else:
            return f"Error: Ollama returned status {response.status_code}"

    except requests.exceptions.Timeout:
        return "Error: Generation timed out. Ollama might be slow - try again."
    except Exception as e:
        return f"Error: {str(e)}"

def generate_motivation_scenario_ollama(model="mistral"):
    """
    Generate a realistic, relatable motivation/challenge scenario using Ollama

    Returns:
        Generated scenario text (e.g., "I'm struggling with...")
    """
    if not check_ollama_running():
        # Fallback to a default scenario if Ollama is not running
        import random
        fallback_scenarios = [
            "How do I stay focused when I have a lot on my mind?",
            "What's the best way to deal with setbacks?",
            "How can I build better daily habits?"
        ]
        return random.choice(fallback_scenarios)

    system_instruction = """Generate a short, general question about motivation, productivity, or personal growth that anyone could relate to.
Keep it simple and universal - avoid specific jobs, hobbies, or life situations. Make it conversational and natural.
Examples of GOOD questions:
- How do I stop procrastinating?
- What's the best way to stay motivated?
- How can I be more consistent with my goals?
- How do I handle feeling overwhelmed?
- What should I do when I feel stuck?
Output ONLY the question itself, nothing else."""

    try:
        payload = {
            "model": model,
            "prompt": system_instruction,
            "stream": False,
            "options": {
                "temperature": 0.9,  # Higher temperature for more variety
                "top_p": 0.95,
                "num_predict": 50
            }
        }

        response = requests.post(OLLAMA_URL, json=payload, timeout=15)

        if response.status_code == 200:
            result = response.json()
            generated_text = result.get('response', '').strip()

            # Clean up the response - remove any preamble
            # Remove quotes if present
            generated_text = generated_text.strip('"\'')

            # Remove common preambles
            preambles = [
                "here's a fresh question:",
                "here's a new question:",
                "here's one:",
                "here is a question:",
                "how about:",
                "try this:",
            ]
            for preamble in preambles:
                if generated_text.lower().startswith(preamble):
                    generated_text = generated_text[len(preamble):].strip()

            # Remove quotes again after preamble removal
            generated_text = generated_text.strip('"\'')

            # Take only the first sentence
            if '.' in generated_text:
                generated_text = generated_text.split('.')[0].strip()
            if '?' not in generated_text and not generated_text.endswith('?'):
                generated_text += '?'

            return generated_text
        else:
            # Fallback on error
            import random
            fallback_scenarios = [
                "How do I overcome perfectionism?",
                "What's the best way to stay on track?",
                "How can I build momentum?"
            ]
            return random.choice(fallback_scenarios)

    except Exception as e:
        # Fallback on any error
        import random
        fallback_scenarios = [
            "How do I stay focused?",
            "What helps with motivation?",
            "How can I be more disciplined?"
        ]
        return random.choice(fallback_scenarios)

def list_ollama_models():
    """List available Ollama models"""
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            data = response.json()
            models = [model['name'] for model in data.get('models', [])]
            return models
        return []
    except:
        return []

if __name__ == "__main__":
    # Test the OCEAN personality generator
    print("Testing OCEAN Personality Generator...")
    print(f"Ollama running: {check_ollama_running()}")
    print(f"Available models: {list_ollama_models()}")
    print()

    test_prompt = "I have an assignment due at midnight and I can't get started"

    print("=== Testing Default Balanced Profile ===")
    response = generate_motivation_response_ollama(test_prompt)
    print(response)
    print()

    print("=== Testing High Conscientiousness + Low Neuroticism (Structured & Confident) ===")
    ocean_structured = {
        "openness": 0.5,
        "conscientiousness": 0.9,
        "extraversion": 0.5,
        "agreeableness": 0.6,
        "neuroticism": 0.2
    }
    response = generate_motivation_response_ollama(test_prompt, ocean_structured)
    print(response)
    print()

    print("=== Testing High Agreeableness + High Neuroticism (Empathetic & Reassuring) ===")
    ocean_empathetic = {
        "openness": 0.6,
        "conscientiousness": 0.4,
        "extraversion": 0.3,
        "agreeableness": 0.9,
        "neuroticism": 0.8
    }
    response = generate_motivation_response_ollama(test_prompt, ocean_empathetic)
    print(response)
    print()

    print("=== Testing High Openness + High Extraversion (Creative & Energetic) ===")
    ocean_creative = {
        "openness": 0.9,
        "conscientiousness": 0.4,
        "extraversion": 0.9,
        "agreeableness": 0.6,
        "neuroticism": 0.3
    }
    response = generate_motivation_response_ollama(test_prompt, ocean_creative)
    print(response)
