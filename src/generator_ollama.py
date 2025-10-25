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

def generate_motivation_response_ollama(prompt, profile_style="neutral", model="mistral", max_length=150):
    """
    Generate motivation response using Ollama
    
    Args:
        prompt: User's concern/situation
        profile_style: "transitional" or "disciplinarian" or "neutral"
        model: Ollama model name (e.g., "mistral", "llama2")
        max_length: Max tokens to generate
    
    Returns:
        Generated response text
    """
    # Check if Ollama is running
    if not check_ollama_running():
        return "Error: Ollama is not running. Please start it with 'ollama serve' or 'ollama run mistral'"
    
    # Create styled prompt based on profile
    if profile_style == "transitional":
        system_instruction = "You are a compassionate motivational coach. First acknowledge feelings with empathy, then provide practical, realistic advice. Be warm but honest. Keep response to 2-3 sentences."
        full_prompt = f"{system_instruction}\n\nPerson: {prompt}\n\nCoach:"
    elif profile_style == "disciplinarian":
        system_instruction = "You are a tough-love motivational coach. Give direct, no-nonsense advice. Be blunt and action-focused. No sugar-coating. Keep response to 2-3 sentences."
        full_prompt = f"{system_instruction}\n\nPerson: {prompt}\n\nCoach:"
    else:
        full_prompt = f"Provide brief motivational advice for: {prompt}"
    
    # Call Ollama API
    try:
        payload = {
            "model": model,
            "prompt": full_prompt,
            "stream": False,
            "options": {
                "temperature": 0.7,
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
            
            # Limit to first 3 sentences for conciseness
            sentences = generated_text.split('. ')
            if len(sentences) > 3:
                generated_text = '. '.join(sentences[:3]) + '.'
            
            return generated_text if generated_text else "Unable to generate response. Please try again."
        else:
            return f"Error: Ollama returned status {response.status_code}"
            
    except requests.exceptions.Timeout:
        return "Error: Generation timed out. Ollama might be slow - try again."
    except Exception as e:
        return f"Error: {str(e)}"

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
    # Test the Ollama generator
    print("Testing Ollama Generator...")
    print(f"Ollama running: {check_ollama_running()}")
    print(f"Available models: {list_ollama_models()}")
    print()
    
    test_prompt = "I have an assignment due at midnight and I can't get started"
    
    print("=== Testing Transitional Style ===")
    response = generate_motivation_response_ollama(test_prompt, "transitional")
    print(response)
    print()
    
    print("=== Testing Disciplinarian Style ===")
    response = generate_motivation_response_ollama(test_prompt, "disciplinarian")
    print(response)

