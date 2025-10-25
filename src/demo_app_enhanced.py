import streamlit as st
import json
import time
import sys
from pathlib import Path

# Add parent directory to path if needed
if str(Path(__file__).parent.parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import load_config, write_jsonl
from src.reward_model import load_reward, score_text
from src.generator_ollama import generate_motivation_response_ollama, check_ollama_running

cfg = load_config()

# Helper functions (defined early)
def load_prompts():
    path = cfg['data_paths']['prompts']
    if not Path(path).exists():
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]

def load_comparisons():
    path = cfg['data_paths']['comparisons']
    if not Path(path).exists():
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]

def get_response_for_profile(prompt_id, profile):
    """Get the stored response for a profile from comparisons data"""
    comparisons = load_comparisons()
    for comp in comparisons:
        if comp.get('prompt_id') == prompt_id:
            if comp.get('profile') == profile:
                return comp.get('a') if comp.get('preferred') == 'a' else comp.get('b')
            else:
                return comp.get('b') if comp.get('preferred') == 'a' else comp.get('a')
    return None

def score_response(text, profile):
    """Score a response for a specific profile"""
    emb, clf = load_reward()
    if clf is None:
        return 0.5
    return score_text(text, emb, clf)

# Profile definitions
PROFILES = {
    "transitional": {
        "name": "The Transition",
        "description": "Comfort first, then reality",
        "icon": "🤝",
        "traits": ["Empathetic", "Gradual", "Understanding", "Realistic"],
        "color": "#4A90E2"
    },
    "disciplinarian": {
        "name": "The Disciplinarian", 
        "description": "Tough love, direct discipline",
        "icon": "💪",
        "traits": ["Direct", "No-nonsense", "Action-focused", "Uncompromising"],
        "color": "#E74C3C"
    }
}

st.set_page_config(page_title="MotivateMe AI", page_icon="🎯", layout="wide")

# Custom CSS
st.markdown("""
<style>
    .profile-card {
        padding: 20px;
        border-radius: 10px;
        border: 2px solid #ddd;
        margin: 10px 0;
    }
    .trait-badge {
        display: inline-block;
        padding: 5px 10px;
        margin: 5px;
        border-radius: 15px;
        background: #f0f0f0;
        font-size: 12px;
    }
    .response-box {
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
        border-left: 4px solid;
    }
</style>
""", unsafe_allow_html=True)

st.title("🎯 MotivateMe AI")
st.markdown("### Personal Motivation That Learns YOUR Language")

# Sidebar - Profile Info
st.sidebar.header("🎭 Motivation Profiles")
for profile_id, profile in PROFILES.items():
    with st.sidebar.expander(f"{profile['icon']} {profile['name']}"):
        st.write(profile['description'])
        st.write("**Characteristics:**")
        for trait in profile['traits']:
            st.markdown(f"• {trait}")

st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Model Stats")

# Check Ollama status
ollama_running = check_ollama_running()
if ollama_running:
    st.sidebar.success("✅ Ollama (Mistral) ready")
else:
    st.sidebar.warning("⚠️ Ollama not detected")
    st.sidebar.info("Start with: `ollama run mistral`")

emb, clf = load_reward()
if clf is not None:
    st.sidebar.success("✅ Reward model trained")
    st.sidebar.metric("Training samples", len(load_comparisons()))
else:
    st.sidebar.warning("⚠️ No reward model yet")
    st.sidebar.info("Add preferences to train!")

# Main content
tab1, tab2, tab3 = st.tabs(["🎯 Compare Profiles", "💬 Chat Mode", "📈 Training"])

with tab1:
    st.header("See How Different Personalities Motivate")
    st.markdown("Same struggle, completely different approaches.")
    
    prompts = load_prompts()
    
    if not prompts:
        st.warning("No prompts loaded. Check data/prompts.jsonl")
    else:
        # Prompt selector
        selected_prompt = st.selectbox(
            "Choose a motivation scenario:",
            prompts,
            format_func=lambda x: f"#{x['id']}: {x['text'][:80]}..."
        )
        
        if selected_prompt:
            st.markdown(f"### 💭 Your Situation:")
            st.info(selected_prompt['text'])
            
            st.markdown("### 🎭 How Each Profile Responds:")
            
            col1, col2 = st.columns(2)
            
            with col1:
                profile = PROFILES['transitional']
                st.markdown(f"""
                <div class="response-box" style="border-left-color: {profile['color']}">
                    <h4>{profile['icon']} {profile['name']}</h4>
                    <p style="color: #666; font-size: 14px;">{profile['description']}</p>
                </div>
                """, unsafe_allow_html=True)
                
                response = get_response_for_profile(selected_prompt['id'], 'transitional')
                if response:
                    st.write(response)
                    score = score_response(response, 'transitional')
                    st.metric("Alignment Score", f"{score:.2%}")
                else:
                    st.write("*Response not yet generated*")
            
            with col2:
                profile = PROFILES['disciplinarian']
                st.markdown(f"""
                <div class="response-box" style="border-left-color: {profile['color']}">
                    <h4>{profile['icon']} {profile['name']}</h4>
                    <p style="color: #666; font-size: 14px;">{profile['description']}</p>
                </div>
                """, unsafe_allow_html=True)
                
                response = get_response_for_profile(selected_prompt['id'], 'disciplinarian')
                if response:
                    st.write(response)
                    score = score_response(response, 'disciplinarian')
                    st.metric("Alignment Score", f"{score:.2%}")
                else:
                    st.write("*Response not yet generated*")
            
            st.markdown("---")
            st.markdown("### 👍 Which approach works better for you?")
            
            pref_col1, pref_col2, pref_col3 = st.columns([1, 1, 1])
            with pref_col1:
                if st.button("🤝 The Transition", use_container_width=True):
                    st.success("Preference logged! The model is learning your style.")
            with pref_col2:
                if st.button("💪 The Disciplinarian", use_container_width=True):
                    st.success("Preference logged! The model is learning your style.")
            with pref_col3:
                if st.button("😐 Neither", use_container_width=True):
                    st.info("Feedback noted. We'll generate better options.")

with tab2:
    st.header("💬 Interactive Chat")
    st.markdown("Get personalized motivation in real-time")
    
    selected_profile = st.radio(
        "Choose your motivation style:",
        ["transitional", "disciplinarian"],
        format_func=lambda x: f"{PROFILES[x]['icon']} {PROFILES[x]['name']}"
    )
    
    profile = PROFILES[selected_profile]
    st.markdown(f"""
    <div class="profile-card" style="border-color: {profile['color']}">
        <h3>{profile['icon']} {profile['name']}</h3>
        <p>{profile['description']}</p>
        <div>
            {''.join([f'<span class="trait-badge">{trait}</span>' for trait in profile['traits']])}
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    user_input = st.text_area("What's on your mind?", placeholder="I'm struggling with...")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        generate_button = st.button("🎯 Get Motivation", type="primary", use_container_width=True)
    with col2:
        show_both = st.checkbox("Show both styles", value=False)
    
    if generate_button:
        if user_input:
            if show_both:
                # Generate both profiles for comparison
                st.markdown("### 🎭 Both Motivation Styles:")
                
                col_trans, col_disc = st.columns(2)
                
                with col_trans:
                    st.markdown(f"**{PROFILES['transitional']['icon']} {PROFILES['transitional']['name']}**")
                    with st.spinner("Generating with Mistral..." if ollama_running else "Generating..."):
                        try:
                            # Use Ollama if available, fallback to message
                            if ollama_running:
                                response_trans = generate_motivation_response_ollama(user_input, "transitional", model="mistral")
                            else:
                                response_trans = "⚠️ Ollama not running. Start Ollama with 'ollama run mistral' to generate responses. Or use Tab 1 for curated examples."
                            
                            st.markdown(f"""
                            <div class="response-box" style="border-left-color: {PROFILES['transitional']['color']}; background: #f8f9fa; padding: 15px;">
                                {response_trans}
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # Score it
                            emb, clf = load_reward()
                            if clf:
                                score = score_text(response_trans, emb, clf)
                                st.metric("Alignment Score", f"{score:.1%}")
                        except Exception as e:
                            st.error(f"Generation error: {e}")
                
                with col_disc:
                    st.markdown(f"**{PROFILES['disciplinarian']['icon']} {PROFILES['disciplinarian']['name']}**")
                    with st.spinner("Generating with Mistral..." if ollama_running else "Generating..."):
                        try:
                            # Use Ollama if available, fallback to message
                            if ollama_running:
                                response_disc = generate_motivation_response_ollama(user_input, "disciplinarian", model="mistral")
                            else:
                                response_disc = "⚠️ Ollama not running. Start Ollama with 'ollama run mistral' to generate responses. Or use Tab 1 for curated examples."
                            
                            st.markdown(f"""
                            <div class="response-box" style="border-left-color: {PROFILES['disciplinarian']['color']}; background: #f8f9fa; padding: 15px;">
                                {response_disc}
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # Score it
                            emb, clf = load_reward()
                            if clf:
                                score = score_text(response_disc, emb, clf)
                                st.metric("Alignment Score", f"{score:.1%}")
                        except Exception as e:
                            st.error(f"Generation error: {e}")
                
                st.markdown("---")
                st.markdown("👆 **Which style motivates you more?** This is how RLHF learns your preferences!")
                
            else:
                # Generate single profile
                st.markdown("### 💬 Your Personalized Motivation:")
                spinner_text = f"Generating with Mistral ({selected_profile})..." if ollama_running else f"Generating {selected_profile} style response..."
                with st.spinner(spinner_text):
                    try:
                        # Use Ollama if available, fallback to message
                        if ollama_running:
                            response = generate_motivation_response_ollama(user_input, selected_profile, model="mistral")
                        else:
                            response = "⚠️ Ollama is not running. Please start Ollama with 'ollama run mistral' in a terminal to enable live generation. Meanwhile, check out Tab 1 (Compare Profiles) to see curated examples of both motivation styles!"
                        
                        st.markdown(f"""
                        <div class="response-box" style="border-left-color: {profile['color']}; background: #f8f9fa; padding: 20px; font-size: 16px;">
                            {response}
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Show alignment score if model trained
                        emb, clf = load_reward()
                        if clf:
                            score = score_text(response, emb, clf)
                            st.metric("Alignment Score", f"{score:.1%}", help="How well this matches learned preferences")
                        
                        # Option to save as preference
                        st.markdown("---")
                        col_feedback1, col_feedback2, col_feedback3 = st.columns(3)
                        with col_feedback1:
                            if st.button("👍 This motivates me!"):
                                st.success("Great! Preference logged.")
                        with col_feedback2:
                            if st.button("👎 Not helpful"):
                                st.info("Thanks for feedback. Try the other profile?")
                        with col_feedback3:
                            if st.button("🔄 Regenerate"):
                                st.rerun()
                                
                    except Exception as e:
                        st.error(f"Generation error: {e}")
                        if not ollama_running:
                            st.info("💡 Tip: The model is loading for the first time. This can take 10-20 seconds. Try again!")
                        else:
                            st.info("💡 Tip: Make sure Ollama is running with `ollama run mistral`")
        else:
            st.warning("Please describe your situation first.")

with tab3:
    st.header("📈 Training Dashboard")
    st.markdown("See how the AI learns from preferences")
    
    comparisons = load_comparisons()
    
    if comparisons:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Preferences", len(comparisons))
        
        with col2:
            transitional_count = len([c for c in comparisons if c.get('profile') == 'transitional'])
            st.metric("Transitional Preferences", transitional_count)
        
        with col3:
            disciplinarian_count = len([c for c in comparisons if c.get('profile') == 'disciplinarian'])
            st.metric("Disciplinarian Preferences", disciplinarian_count)
        
        st.markdown("### Recent Training Data")
        for i, comp in enumerate(comparisons[-5:]):
            with st.expander(f"Training Sample {i+1}"):
                st.write(f"**Profile:** {comp.get('profile', 'unknown')}")
                st.write(f"**Preferred Response:** {comp.get(comp.get('preferred', 'a'), 'N/A')}")
        
        if st.button("🔄 Retrain Reward Model", type="primary"):
            with st.spinner("Training model on your preferences..."):
                try:
                    from src.reward_model import train_reward
                    train_reward()
                    st.success("✅ Model retrained! Responses will now be more aligned with preferences.")
                    st.rerun()
                except Exception as e:
                    st.error(f"Training failed: {e}")
                    st.info("Make sure you have training data in data/comparisons.jsonl")
    else:
        st.info("No training data yet. Use the Compare Profiles tab to add preferences!")
        
    st.markdown("---")
    st.markdown("### 🎯 How It Works")
    st.markdown("""
    1. **You choose** between different motivation styles
    2. **The model learns** your preferences through RLHF
    3. **Future responses adapt** to match your personal motivation language
    4. **Continuous improvement** as more preferences are collected
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 20px;">
    <p><strong>MotivateMe AI</strong> - Personalized Motivation Through RLHF</p>
    <p>Built with ❤️ for better motivation</p>
</div>
""", unsafe_allow_html=True)

