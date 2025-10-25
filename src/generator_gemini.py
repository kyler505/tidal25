"""
Gemini-based text generation for MotivateMe AI
Using Google's Gemini Flash for fast, efficient responses
"""
import google.generativeai as genai
import os

# Configure Gemini API
genai.configure(api_key="AIzaSyBvXTbzsSB6mkdFdNN8g-Ah4G3IPPr9cFY")

# Using Gemini 2.5 Flash for generation
model = genai.GenerativeModel('gemini-2.0-flash-exp')

def check_gemini_available():
    """Check if Gemini API is accessible"""
    try:
        # Simple test to see if API is working
        response = model.generate_content("test")
        return True
    except:
        return False

def generate_motivation_response_gemini(prompt, ocean_profile=None):
    """
    Generate motivation response using Gemini with OCEAN personality model

    Args:
        prompt: User's concern/situation
        ocean_profile: dict with OCEAN scores (0.0-1.0)

    Returns:
        Generated response text
    """
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

    try:
        response = model.generate_content(
            full_prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0.8,
                max_output_tokens=100,
            )
        )

        generated_text = response.text.strip()

        # Clean up response
        if '\n' in generated_text:
            generated_text = generated_text.split('\n')[0].strip()

        # Limit to first 2 sentences
        sentences = generated_text.split('. ')
        if len(sentences) > 2:
            generated_text = '. '.join(sentences[:2]) + '.'

        return generated_text if generated_text else "Unable to generate response. Please try again."

    except Exception as e:
        return f"Error generating response: {str(e)}"


def generate_motivation_scenario_gemini():
    """
    Generate a realistic, relatable motivation question using Gemini

    Returns:
        Generated question text
    """
    import random

    # Fallback scenarios in case of error
    fallback_scenarios = [
        "How do I stay focused when I have a lot on my mind?",
        "What's the best way to deal with setbacks?",
        "How can I build better daily habits?"
    ]

    try:
        system_instruction = """Generate a short, general question about motivation, productivity, or personal growth that anyone could relate to.
Keep it simple and universal - avoid specific jobs, hobbies, or life situations. Make it conversational and natural.
Examples of GOOD questions:
- How do I stop procrastinating?
- What's the best way to stay motivated?
- How can I be more consistent with my goals?
- How do I handle feeling overwhelmed?
- What should I do when I feel stuck?
Output ONLY the question itself, nothing else."""

        response = model.generate_content(
            system_instruction,
            generation_config=genai.types.GenerationConfig(
                temperature=0.9,
                max_output_tokens=50,
            )
        )

        generated_text = response.text.strip()

        # Clean up the response - remove any preamble
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

        # Remove quotes again
        generated_text = generated_text.strip('"\'')

        # Take only the first sentence
        if '.' in generated_text:
            generated_text = generated_text.split('.')[0].strip()
        if '?' not in generated_text and not generated_text.endswith('?'):
            generated_text += '?'

        return generated_text if generated_text else random.choice(fallback_scenarios)

    except Exception as e:
        return random.choice(fallback_scenarios)


if __name__ == "__main__":
    # Test the Gemini generator
    print("Testing Gemini Generator...")
    print(f"Gemini available: {check_gemini_available()}")
    print()

    test_prompt = "How do I stay motivated?"

    print("=== Testing Default Balanced Profile ===")
    response = generate_motivation_response_gemini(test_prompt)
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
    response = generate_motivation_response_gemini(test_prompt, ocean_structured)
    print(response)
    print()

    print("=== Testing Scenario Generation ===")
    scenario = generate_motivation_scenario_gemini()
    print(scenario)
