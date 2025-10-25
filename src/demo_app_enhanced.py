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
from src.generator_gemini import generate_motivation_response_gemini, check_gemini_available, generate_motivation_scenario_gemini
from src.ocean_trainer import train_from_ocean_feedback, get_training_stats, score_response_with_ocean

# Page config MUST be first Streamlit command
st.set_page_config(
    page_title="MotivateMe AI",
    page_icon="✦",
    layout="centered",
    initial_sidebar_state="collapsed",
    menu_items={
        'Get Help': None,
        'Report a bug': None,
        'About': None
    }
)

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

def load_user_ocean_profile():
    """Load the user's learned OCEAN profile from file"""
    profile_path = "data/user_ocean_profile.json"
    if Path(profile_path).exists():
        with open(profile_path) as f:
            return json.load(f)
    else:
        # Default balanced profile
        return {
            "openness": 0.5,
            "conscientiousness": 0.5,
            "extraversion": 0.5,
            "agreeableness": 0.5,
            "neuroticism": 0.5,
            "feedback_count": 0,
            "last_updated": None
        }

def save_user_ocean_profile(ocean_profile):
    """Save the user's OCEAN profile to file"""
    profile_path = "data/user_ocean_profile.json"
    Path("data").mkdir(exist_ok=True)
    ocean_profile["last_updated"] = time.time()
    with open(profile_path, 'w') as f:
        json.dump(ocean_profile, f, indent=2)

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

def update_ocean_from_feedback(current_ocean, feedback_type, response_ocean, learning_rate=None):
    """Update user's OCEAN profile based on feedback with dynamic learning rate

    Args:
        current_ocean: User's current OCEAN scores
        feedback_type: 'positive' or 'negative'
        response_ocean: OCEAN profile that generated the response
        learning_rate: Optional fixed learning rate (if None, uses dynamic rate)
    """
    updated_ocean = current_ocean.copy()
    feedback_count = current_ocean.get('feedback_count', 0)

    # Use dynamic learning rate if not specified
    if learning_rate is None:
        learning_rate = calculate_dynamic_learning_rate(feedback_count)

    # Store the learning rate used for this update (for debugging/transparency)
    updated_ocean['last_learning_rate'] = learning_rate

    for dimension in ['openness', 'conscientiousness', 'extraversion', 'agreeableness', 'neuroticism']:
        current_val = updated_ocean.get(dimension, 0.5)
        response_val = response_ocean.get(dimension, 0.5)

        if feedback_type == 'positive':
            # Move toward the response profile that worked
            adjustment = (response_val - current_val) * learning_rate
        else:
            # Move away from the response profile that didn't work
            adjustment = -(response_val - current_val) * learning_rate * 0.5  # Smaller negative adjustment

        # Apply adjustment with bounds checking
        new_val = current_val + adjustment
        updated_ocean[dimension] = max(0.0, min(1.0, new_val))

    updated_ocean['feedback_count'] = feedback_count + 1
    return updated_ocean

def save_feedback_data(prompt, response, ocean_profile, feedback_type):
    """Save feedback data and auto-train the reward model"""
    feedback_entry = {
        "timestamp": time.time(),
        "prompt": prompt,
        "response": response,
        "ocean_profile": ocean_profile,
        "feedback": feedback_type,
        "session_id": st.session_state.get('session_id', 'unknown')
    }

    feedback_path = "data/ocean_feedback.jsonl"
    Path("data").mkdir(exist_ok=True)
    with open(feedback_path, 'a') as f:
        f.write(json.dumps(feedback_entry) + '\n')

    # Auto-train pipeline in background when we have enough feedback
    # Train every 3 feedback entries to keep model updated
    try:
        with open(feedback_path) as f:
            feedback_count = sum(1 for _ in f)

        # Auto-train every 3 feedback entries (minimal threshold for training)
        if feedback_count >= 3 and feedback_count % 3 == 0:
            # Run training silently in background
            try:
                result = train_from_ocean_feedback(min_feedback_count=3, auto_train=True)
                if result.get('status') == 'success':
                    # Mark that we successfully trained
                    st.session_state.last_training_count = feedback_count
                    st.session_state.training_success = True
            except Exception as e:
                # Silent fail - don't interrupt user experience
                print(f"Background training failed: {e}")
    except:
        pass

# OCEAN personality dimensions
OCEAN_DIMENSIONS = {
    "openness": {
        "name": "Openness to Experience",
        "description": "Creativity, curiosity, and openness to new ideas",
        "low_desc": "Practical, traditional approaches",
        "high_desc": "Creative, novel solutions"
    },
    "conscientiousness": {
        "name": "Conscientiousness",
        "description": "Organization, discipline, and goal-directed behavior",
        "low_desc": "Flexible, spontaneous methods",
        "high_desc": "Structured, organized plans"
    },
    "extraversion": {
        "name": "Extraversion",
        "description": "Social energy, assertiveness, and enthusiasm",
        "low_desc": "Calm, reflective approach",
        "high_desc": "Energetic, social strategies"
    },
    "agreeableness": {
        "name": "Agreeableness",
        "description": "Empathy, cooperation, and trust",
        "low_desc": "Direct, honest feedback",
        "high_desc": "Warm, supportive validation"
    },
    "neuroticism": {
        "name": "Neuroticism",
        "description": "Emotional sensitivity and stress responsiveness",
        "low_desc": "Confident, resilient focus",
        "high_desc": "Reassuring, anxiety-aware support"
    }
}

# Initialize session state
if 'session_id' not in st.session_state:
    st.session_state.session_id = str(int(time.time()))

if 'user_ocean_profile' not in st.session_state:
    st.session_state.user_ocean_profile = load_user_ocean_profile()

if 'learned_profile_dirty' not in st.session_state:
    st.session_state.learned_profile_dirty = False

if 'should_retrain' not in st.session_state:
    st.session_state.should_retrain = False

if 'last_training_count' not in st.session_state:
    st.session_state.last_training_count = 0

if 'training_success' not in st.session_state:
    st.session_state.training_success = False

if 'profile_questionnaire_complete' not in st.session_state:
    st.session_state.profile_questionnaire_complete = False

# Modern dark theme styling matching Figma CSS
st.markdown("""
<style>
    /* CSS Variables from Figma */
    :root {
        --background: #0a0a0a;
        --foreground: #ededed;
        --card: #0a0a0a;
        --card-foreground: #ededed;
        --primary: #ededed;
        --primary-foreground: #0a0a0a;
        --secondary: #1a1a1a;
        --secondary-foreground: #ededed;
        --muted: #1a1a1a;
        --muted-foreground: #888888;
        --accent: #1a1a1a;
        --accent-foreground: #ededed;
        --border: #222222;
        --input: #1a1a1a;
        --ring: #333333;
        --font-weight-medium: 500;
        --font-weight-normal: 400;
        --radius: 0.625rem;
    }

    /* Global app background */
    .stApp {
        background-color: var(--background);
        color: var(--foreground);
    }

    .main .block-container {
        padding-top: 0rem;
        padding-bottom: 0rem;
        max-width: 640px;
        margin: 0 auto;
    }

    /* Remove top spacing */
    .block-container {
        padding-top: 0rem !important;
    }

    /* Header styling */
    h1 {
        font-size: 1.5rem !important;
        font-weight: var(--font-weight-medium) !important;
        margin-bottom: 0 !important;
        margin-top: 0 !important;
        padding-top: 1rem !important;
        color: var(--foreground);
    }

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
        border-bottom: 1px solid var(--border);
    }

    .stTabs [data-baseweb="tab"] {
        padding: 0.75rem 0;
        color: var(--muted-foreground);
        font-weight: var(--font-weight-normal);
        background-color: transparent;
        border: none;
    }

    .stTabs [aria-selected="true"] {
        color: var(--foreground);
        border-bottom: 2px solid var(--foreground);
    }

    /* Card/Box styling */
    .profile-badge {
        background: var(--secondary);
        padding: 1rem;
        border-radius: var(--radius);
        border: 1px solid var(--border);
        margin: 1.5rem 0;
        display: flex;
        align-items: center;
        gap: 0.75rem;
    }

    .profile-badge-icon {
        font-size: 1.5rem;
        display: flex;
        align-items: center;
        justify-content: center;
        color: var(--foreground);
    }

    .profile-badge-icon svg {
        width: 24px;
        height: 24px;
        stroke: var(--foreground);
    }

    .profile-badge-text {
        flex: 1;
    }

    .profile-badge-title {
        font-size: 0.875rem;
        font-weight: var(--font-weight-medium);
        color: var(--foreground);
        margin: 0;
    }

    .profile-badge-subtitle {
        font-size: 0.75rem;
        color: var(--muted-foreground);
        margin: 0;
    }

    /* Input styling */
    .stTextArea textarea {
        background-color: var(--input) !important;
        border: 1px solid var(--border) !important;
        border-radius: var(--radius) !important;
        color: var(--foreground) !important;
        font-size: 0.95rem !important;
        font-weight: var(--font-weight-normal) !important;
    }

    .stTextArea textarea::placeholder {
        color: var(--muted-foreground) !important;
    }

    /* Dropdown styling */
    .stSelectbox > div > div {
        background-color: var(--input) !important;
        border: 1px solid var(--border) !important;
        border-radius: var(--radius) !important;
        color: var(--foreground) !important;
    }

    /* Button styling */
    .stButton > button {
        background-color: var(--muted) !important;
        color: var(--foreground) !important;
        border: none !important;
        border-radius: var(--radius) !important;
        padding: 0.5rem 2rem !important;
        font-weight: var(--font-weight-medium) !important;
        transition: all 0.2s !important;
    }

    .stButton > button:hover {
        background-color: var(--accent) !important;
        opacity: 0.9;
    }

    .stButton > button[kind="primary"] {
        background-color: var(--primary) !important;
        color: var(--primary-foreground) !important;
    }

    .stButton > button[kind="primary"]:hover {
        opacity: 0.9;
    }

    /* Comparison cards */
    .comparison-card {
        background: var(--secondary);
        padding: 1.5rem;
        border-radius: var(--radius);
        border: 1px solid var(--border);
        margin: 1rem 0;
    }

    /* Comparison cards in Build Your Profile tab need min-height */
    .comparison-card.comparison-response {
        min-height: auto;
    }

    .comparison-header {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: calc(var(--radius) - 4px);
        font-size: 0.75rem;
        font-weight: 600;
        margin-bottom: 0.75rem;
    }

    .style-a {
        background: rgba(138, 43, 226, 0.2);
        color: #BA68C8;
    }

    .style-b {
        background: rgba(0, 188, 212, 0.2);
        color: #4DD0E1;
    }

    .comparison-subtitle {
        font-size: 0.8rem;
        color: var(--muted-foreground);
        margin-bottom: 1rem;
    }

    .comparison-text {
        font-size: 0.9rem;
        color: var(--card-foreground);
        line-height: 1.6;
    }

    /* Scenario box */
    .scenario-box {
        background: var(--secondary);
        padding: 1rem 1.25rem;
        border-radius: var(--radius);
        border-left: 3px solid var(--muted-foreground);
        margin: 1.5rem 0;
    }

    .scenario-label {
        font-size: 0.75rem;
        color: var(--muted-foreground);
        margin-bottom: 0.25rem;
    }

    .scenario-text {
        font-size: 0.95rem;
        color: var(--foreground);
        font-weight: var(--font-weight-medium);
    }

    /* Stats cards */
    .stat-card {
        background: var(--secondary);
        padding: 1.25rem;
        border-radius: var(--radius);
        border: 1px solid var(--border);
        text-align: center;
    }

    .stat-icon {
        font-size: 1.5rem;
        margin-bottom: 0.5rem;
        display: flex;
        align-items: center;
        justify-content: center;
    }

    .stat-icon svg {
        width: 24px;
        height: 24px;
        stroke: var(--muted-foreground);
    }

    .stat-label {
        font-size: 0.75rem;
        color: var(--muted-foreground);
        margin-bottom: 0.25rem;
    }

    .stat-value {
        font-size: 1.5rem;
        font-weight: 600;
        color: var(--foreground);
    }

    .stat-status {
        font-size: 1.5rem;
        color: #BA68C8;
        font-weight: 600;
    }

    /* OCEAN dimension slider */
    .ocean-dimension {
        position: relative;
        background: linear-gradient(135deg, rgba(138, 43, 226, 0.12), rgba(0, 188, 212, 0.05));
        padding: 1.1rem 1.35rem;
        border-radius: calc(var(--radius) + 0.25rem);
        border: 1px solid rgba(255, 255, 255, 0.05);
        margin: 0.85rem 0;
        box-shadow: inset 0 0 0 1px rgba(255, 255, 255, 0.02), 0 10px 30px rgba(0, 0, 0, 0.35);
        -webkit-backdrop-filter: blur(14px);
        backdrop-filter: blur(14px);
        overflow: hidden;
    }

    .ocean-dimension::after {
        content: "";
        position: absolute;
        inset: 0;
        background: radial-gradient(circle at top right, rgba(255, 255, 255, 0.08), transparent 55%);
        pointer-events: none;
    }

    .ocean-dimension-content {
        position: relative;
        z-index: 1;
    }

    .ocean-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 0.75rem;
    }

    .ocean-header-left {
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    .ocean-name {
        font-size: 0.95rem;
        color: var(--foreground);
        font-weight: 600;
        letter-spacing: 0.01em;
    }

    .ocean-info-icon svg {
        width: 16px;
        height: 16px;
        stroke: rgba(255, 255, 255, 0.35);
    }

    .ocean-score-badge {
        font-size: 0.75rem;
        font-weight: 600;
        border-radius: 999px;
        padding: 0.2rem 0.6rem;
        background: rgba(255, 255, 255, 0.08);
        color: var(--foreground);
        letter-spacing: 0.01em;
    }

    .ocean-score-positive {
        background: rgba(76, 175, 80, 0.18);
        color: #7DFF9C;
        box-shadow: 0 0 10px rgba(125, 255, 156, 0.2);
    }

    .ocean-score-negative {
        background: rgba(239, 83, 80, 0.16);
        color: #FF8989;
        box-shadow: 0 0 10px rgba(255, 137, 137, 0.18);
    }

    .ocean-score-neutral {
        background: rgba(255, 255, 255, 0.08);
        color: rgba(255, 255, 255, 0.6);
    }

    .ocean-progress-container {
        position: relative;
        margin: 0.85rem 0 0.55rem 0;
    }

    .ocean-track {
        position: relative;
        height: 12px;
        border-radius: 999px;
        background: rgba(255, 255, 255, 0.08);
        box-shadow: inset 0 1px 2px rgba(0, 0, 0, 0.35);
        overflow: visible;
    }

    .ocean-track-fill {
        position: absolute;
        top: 0;
        left: 0;
        height: 100%;
        border-radius: inherit;
        background: linear-gradient(90deg, #8A2BE2 0%, #00BCD4 100%);
        box-shadow: 0 8px 18px rgba(0, 188, 212, 0.28);
    }

    .ocean-indicator {
        position: absolute;
        top: 50%;
        left: 50%;
        transform: translate(-50%, -18%);
        display: flex;
        flex-direction: column;
        align-items: center;
        z-index: 2;
        pointer-events: none;
    }

    .ocean-indicator-dot {
        width: 18px;
        height: 18px;
        background: #FFFFFF;
        border-radius: 50%;
        border: 3px solid rgba(10, 10, 10, 0.6);
        box-shadow: 0 6px 14px rgba(0, 0, 0, 0.45);
    }

    .ocean-indicator-value {
        margin-top: 0.45rem;
        font-size: 0.72rem;
        font-weight: 600;
        color: #EDEDED;
        background: rgba(0, 0, 0, 0.55);
        padding: 0.18rem 0.6rem;
        border-radius: 999px;
        letter-spacing: 0.02em;
        white-space: nowrap;
    }

    .ocean-labels {
        display: flex;
        justify-content: space-between;
        align-items: center;
        font-size: 0.72rem;
        color: rgba(255, 255, 255, 0.45);
        margin-top: 1.1rem;
    }

    .ocean-labels span:first-child {
        text-align: left;
    }

    .ocean-labels span:last-child {
        text-align: right;
    }

    .ocean-description {
        font-size: 0.76rem;
        color: rgba(255, 255, 255, 0.55);
        margin-top: 0.65rem;
    }

    /* Metric styling */
    [data-testid="stMetricValue"] {
        font-size: 1.75rem !important;
        font-weight: 600 !important;
        color: var(--foreground) !important;
    }

    /* Expander styling */
    .streamlit-expanderHeader {
        background-color: var(--secondary) !important;
        border: 1px solid var(--border) !important;
        border-radius: var(--radius) !important;
        color: var(--foreground) !important;
    }

    /* Caption styling */
    .stCaption {
        color: var(--muted-foreground) !important;
    }

    /* Hide Streamlit branding and menu */
    #MainMenu {visibility: hidden !important;}
    footer {visibility: hidden !important;}
    header {visibility: hidden !important;}

    /* Alternative method to hide header */
    [data-testid="stHeader"] {
        display: none !important;
    }

    /* Smooth transitions */
    * {
        transition: background-color 0.2s ease, border-color 0.2s ease;
    }
</style>
""", unsafe_allow_html=True)

# App logo header
st.image("resources/motivateme.png")
st.markdown("")

# Get user profile info
user_profile = st.session_state.user_ocean_profile
feedback_count = user_profile.get('feedback_count', 0)

# Check Gemini status
gemini_available = check_gemini_available()
emb, clf = load_reward()

tab1, tab2, tab3 = st.tabs(["Get Motivation", "Build Your Profile", "Your Progress"])

with tab1:
    # Profile building badge at top
    st.markdown(f"""
    <div class="profile-badge">
        <div class="profile-badge-icon">
            <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                <circle cx="12" cy="12" r="10"></circle>
                <polyline points="12 6 12 12 16 14"></polyline>
            </svg>
        </div>
        <div class="profile-badge-text">
            <p class="profile-badge-title">Profile Building</p>
            <p class="profile-badge-subtitle">{feedback_count} interactions recorded. Keep giving feedback to improve personalization!</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Input section
    st.markdown("#### What are you struggling with?")
    user_input = st.text_area(
        "What are you struggling with?",
        placeholder="I'm struggling with...",
        height=120,
        label_visibility="collapsed"
    )

    st.markdown("#### Choose Your Motivation Style")
    st.caption("Select motivation personality")

    # Profile selector with expandable view
    profile_options = []
    if feedback_count >= 3:
        profile_options.append("Your Learned Profile")
    profile_options.extend(["Balanced", "Empathetic", "Structured", "Creative"])

    selected = st.selectbox(
        "Choose Your Motivation Style",
        profile_options,
        label_visibility="collapsed"
    )

    # Expandable profile view
    with st.expander("View OCEAN Dimensions", expanded=False):
        for dim in OCEAN_DIMENSIONS.keys():
            score = user_profile.get(dim, 0.5)
            col1, col2 = st.columns([3, 1])
            with col1:
                st.caption(OCEAN_DIMENSIONS[dim]['name'])
            with col2:
                st.caption(f"{score:.2f}")
            st.progress(score)

    # Generate button
    st.markdown("")
    if st.button("Generate Motivation", type="primary", use_container_width=True):
        # Map selection to OCEAN profile
        profile_map = {
            "Your Learned Profile": {k: v for k, v in user_profile.items() if k in OCEAN_DIMENSIONS},
            "Balanced": {"openness": 0.5, "conscientiousness": 0.5, "extraversion": 0.5, "agreeableness": 0.5, "neuroticism": 0.5},
            "Empathetic": {"openness": 0.6, "conscientiousness": 0.4, "extraversion": 0.3, "agreeableness": 0.9, "neuroticism": 0.8},
            "Structured": {"openness": 0.4, "conscientiousness": 0.9, "extraversion": 0.6, "agreeableness": 0.5, "neuroticism": 0.2},
            "Creative": {"openness": 0.9, "conscientiousness": 0.4, "extraversion": 0.8, "agreeableness": 0.7, "neuroticism": 0.3}
        }

        selected_ocean = profile_map.get(selected, profile_map["Balanced"])

        if user_input:
            with st.spinner("Generating..."):
                try:
                    response = generate_motivation_response_gemini(user_input, selected_ocean)

                    st.markdown(f"""
                    <div class="comparison-card">
                        <div class="comparison-text">{response}</div>
                    </div>
                    """, unsafe_allow_html=True)

                    # Simple feedback
                    st.markdown("")
                    col_yes, col_no = st.columns(2)

                    # Store for feedback
                    st.session_state.last_response = {
                        'prompt': user_input,
                        'response': response,
                        'ocean': selected_ocean
                    }

                    with col_yes:
                        if st.button("✓ Yes, this helped", use_container_width=True):
                            updated = update_ocean_from_feedback(user_profile, 'positive', selected_ocean)
                            st.session_state.user_ocean_profile = updated
                            save_user_ocean_profile(updated)
                            save_feedback_data(user_input, response, selected_ocean, 'positive')
                            st.success("Thanks! Profile updated.")
                            time.sleep(1)
                            st.rerun()

                    with col_no:
                        if st.button("✕ Not helpful", use_container_width=True):
                            updated = update_ocean_from_feedback(user_profile, 'negative', selected_ocean)
                            st.session_state.user_ocean_profile = updated
                            save_user_ocean_profile(updated)
                            save_feedback_data(user_input, response, selected_ocean, 'negative')
                            st.info("Thanks for the feedback!")
                            time.sleep(1)
                            st.rerun()

                except Exception as e:
                    st.error(f"Error: {e}")
        else:
            st.warning("Please enter what you're struggling with first.")

with tab2:
    # Building profile badge
    st.markdown(f"""
    <div class="profile-badge">
        <div class="profile-badge-icon">
            <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                <path d="M12 2v20M2 12h20"></path>
            </svg>
        </div>
        <div class="profile-badge-text">
            <p class="profile-badge-title">Building Your Profile</p>
            <p class="profile-badge-subtitle">{feedback_count} interactions recorded. Keep going!</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### Compare these two responses")
    st.caption("Your choice trains the AI to understand your preferences")

    # Initialize
    if 'current_scenario' not in st.session_state:
        st.session_state.current_scenario = generate_motivation_scenario_gemini()
    if 'preset_rotation' not in st.session_state:
        st.session_state.preset_rotation = 0

    # Pick dimension to test
    dims = list(OCEAN_DIMENSIONS.keys())
    test_dim = dims[st.session_state.preset_rotation % len(dims)]

    # Create contrasting profiles
    ocean_a = {k: 0.5 for k in dims}
    ocean_b = ocean_a.copy()
    ocean_a[test_dim] = 0.9
    ocean_b[test_dim] = 0.1

    # Generate responses if needed
    if 'auto_response_a' not in st.session_state:
        with st.spinner("Generating comparison..."):
            st.session_state.auto_response_a = generate_motivation_response_gemini(
                st.session_state.current_scenario, ocean_a
            )
            st.session_state.auto_response_b = generate_motivation_response_gemini(
                st.session_state.current_scenario, ocean_b
            )
            st.session_state.auto_ocean_a = ocean_a
            st.session_state.auto_ocean_b = ocean_b

    # Display scenario
    st.markdown(f"""
    <div class="scenario-box">
        <div class="scenario-label">Situation:</div>
        <div class="scenario-text">{st.session_state.current_scenario}</div>
    </div>
    """, unsafe_allow_html=True)

    st.caption(f"Testing: {OCEAN_DIMENSIONS[test_dim]['name']} ({OCEAN_DIMENSIONS[test_dim]['description']})")
    st.markdown("")

    # Show responses side by side
    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f"""
        <div class="comparison-card comparison-response">
            <span class="comparison-header style-a">Option A</span>
            <p class="comparison-text">{st.session_state.auto_response_a}</p>
        </div>
        """, unsafe_allow_html=True)

        if st.button("Select this response", key="choose_a", type="primary", use_container_width=True):
            save_feedback_data(
                st.session_state.current_scenario,
                st.session_state.auto_response_a,
                st.session_state.auto_ocean_a,
                'positive'
            )
            updated = update_ocean_from_feedback(
                user_profile, 'positive', st.session_state.auto_ocean_a
            )
            st.session_state.user_ocean_profile = updated
            save_user_ocean_profile(updated)

            # Reset for next
            st.session_state.current_scenario = generate_motivation_scenario_gemini()
            st.session_state.preset_rotation += 1
            del st.session_state.auto_response_a
            del st.session_state.auto_response_b
            st.rerun()

    with col2:
        st.markdown(f"""
        <div class="comparison-card comparison-response">
            <span class="comparison-header style-b">Option B</span>
            <p class="comparison-text">{st.session_state.auto_response_b}</p>
        </div>
        """, unsafe_allow_html=True)

        if st.button("Select this response", key="choose_b", type="primary", use_container_width=True):
            save_feedback_data(
                st.session_state.current_scenario,
                st.session_state.auto_response_b,
                st.session_state.auto_ocean_b,
                'positive'
            )
            updated = update_ocean_from_feedback(
                user_profile, 'positive', st.session_state.auto_ocean_b
            )
            st.session_state.user_ocean_profile = updated
            save_user_ocean_profile(updated)

            # Reset for next
            st.session_state.current_scenario = generate_motivation_scenario_gemini()
            st.session_state.preset_rotation += 1
            del st.session_state.auto_response_a
            del st.session_state.auto_response_b
            st.rerun()

    st.markdown("")
    st.markdown(f"<p style='text-align: center; color: #808080; font-size: 0.9rem;'>Test Progress: {feedback_count} comparisons completed</p>", unsafe_allow_html=True)

with tab3:
    st.markdown("### Your Learning Progress")
    st.caption("See how your personality profile has evolved")

    training_stats = get_training_stats()

    # Stats cards in grid
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-icon">
                <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                    <polyline points="23 6 13.5 15.5 8.5 10.5 1 18"></polyline>
                    <polyline points="17 6 23 6 23 12"></polyline>
                </svg>
            </div>
            <div class="stat-label">Total Feedback</div>
            <div class="stat-value">{feedback_count}</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-icon">
                <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                    <polyline points="20 6 9 17 4 12"></polyline>
                </svg>
            </div>
            <div class="stat-label">Positive</div>
            <div class="stat-value">{training_stats['positive_feedback']}</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-icon">
                <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                    <line x1="18" y1="6" x2="6" y2="18"></line>
                    <line x1="6" y1="6" x2="18" y2="18"></line>
                </svg>
            </div>
            <div class="stat-label">Negative</div>
            <div class="stat-value">{training_stats['negative_feedback']}</div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-icon">
                <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                    <polyline points="22 12 18 12 15 21 9 3 6 12 2 12"></polyline>
                </svg>
            </div>
            <div class="stat-label">Status</div>
            <div class="stat-status">Active</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("")
    st.markdown("")

    if feedback_count > 0:
        st.markdown("### Your Current OCEAN Profile")
        st.caption("These dimensions shape how I communicate with you")

        # Show OCEAN dimensions with custom styling
        for dim in OCEAN_DIMENSIONS.keys():
            score = user_profile.get(dim, 0.5)
            delta = score - 0.5
            delta_rounded = round(delta, 2)
            score_percentage = max(0.0, min(100.0, score * 100))
            indicator_position = max(2.5, min(97.5, score_percentage))
            delta_display = f"{delta_rounded:+.2f}"
            score_display = f"{score:.2f}"

            if score >= 0.6:
                description_text = OCEAN_DIMENSIONS[dim]['high_desc']
            elif score <= 0.4:
                description_text = OCEAN_DIMENSIONS[dim]['low_desc']
            else:
                description_text = "Balanced approach"

            if delta_rounded > 0.02:
                delta_class = "ocean-score-badge ocean-score-positive"
            elif delta_rounded < -0.02:
                delta_class = "ocean-score-badge ocean-score-negative"
            else:
                delta_class = "ocean-score-badge ocean-score-neutral"

            st.markdown(f"""
            <div class="ocean-dimension">
                <div class="ocean-dimension-content">
                    <div class="ocean-header">
                        <div class="ocean-header-left">
                            <span class="ocean-name" title="{OCEAN_DIMENSIONS[dim]['description']}">{OCEAN_DIMENSIONS[dim]['name']}</span>
                            <span class="ocean-info-icon">
                                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                                    <circle cx="12" cy="12" r="10"></circle>
                                    <path d="M9.09 9a3 3 0 0 1 5.83 1c0 2-3 3-3 3"></path>
                                    <line x1="12" y1="17" x2="12.01" y2="17"></line>
                                </svg>
                            </span>
                        </div>
                        <span class="{delta_class}">{delta_display}</span>
                    </div>
                    <div class="ocean-progress-container">
                        <div class="ocean-track">
                            <div class="ocean-track-fill" style="width: {score_percentage}%;"></div>
                            <div class="ocean-indicator" style="left: {indicator_position}%;">
                                <div class="ocean-indicator-dot"></div>
                                <div class="ocean-indicator-value">{score_display}</div>
                            </div>
                        </div>
                    </div>
                    <div class="ocean-labels">
                        <span>0.0</span>
                        <span>1.0</span>
                    </div>
                    <div class="ocean-description">{description_text}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("")

        # Show learning rate information
        current_learning_rate = calculate_dynamic_learning_rate(feedback_count)
        learning_progress = min(100, (feedback_count / 50) * 100)  # Scale to show progress toward refinement

        st.markdown("### Learning Rate")
        st.markdown(f"""
        <div class="comparison-card">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
                <p class="comparison-text" style="margin: 0;">Current adjustment strength: <strong>{current_learning_rate:.1%}</strong></p>
                <span style="font-size: 0.8rem; color: var(--muted-foreground);">{feedback_count} responses</span>
            </div>
            <div style="background: var(--muted); height: 8px; border-radius: 4px; overflow: hidden; margin-top: 0.5rem;">
                <div style="background: linear-gradient(90deg, #8A2BE2 0%, #00BCD4 100%); height: 100%; width: {learning_progress}%;"></div>
            </div>
            <p style="font-size: 0.8rem; color: var(--muted-foreground); margin-top: 0.5rem;">
                {'Early learning phase - Your feedback has high impact!' if feedback_count < 10 else
                 'Building phase - Profile refining quickly' if feedback_count < 25 else
                 'Refinement phase - Making subtle adjustments'}
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("")
        st.markdown("### What This Means")
        st.markdown("""
        <div class="comparison-card">
            <p class="comparison-text">
            Your profile is balanced. Continue giving feedback to shape your preferences and help the AI understand your unique communication style.
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("")
        st.markdown("### Recent Feedback")
        st.markdown("""
        <div class="comparison-card">
            <p class="comparison-text" style="color: #808080;">
            No recent feedback to display
            </p>
        </div>
        """, unsafe_allow_html=True)

        # Reset button
        st.markdown("")
        st.markdown("")
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            if st.button("Reset Learned Profile", type="secondary", use_container_width=True):
                if 'confirm_reset' not in st.session_state:
                    st.session_state.confirm_reset = False

                if not st.session_state.confirm_reset:
                    st.session_state.confirm_reset = True
                    st.warning("Click again to confirm reset")
                else:
                    # Reset everything
                    default = {k: 0.5 for k in OCEAN_DIMENSIONS.keys()}
                    default['feedback_count'] = 0
                    default['last_updated'] = None

                    st.session_state.user_ocean_profile = default
                    save_user_ocean_profile(default)

                    # Clear data files
                    for file in ["data/ocean_feedback.jsonl", "data/comparisons.jsonl", "data/prompts.jsonl"]:
                        if Path(file).exists():
                            Path(file).unlink()

                    st.session_state.confirm_reset = False
                    st.success("Profile reset!")
                    time.sleep(1)
                    st.rerun()
    else:
        st.markdown("""
        <div class="comparison-card" style="text-align: center; padding: 3rem 2rem;">
            <div style="font-size: 3rem; margin-bottom: 1rem;">
                <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="display: inline-block;">
                    <line x1="18" y1="20" x2="18" y2="10"></line>
                    <line x1="12" y1="20" x2="12" y2="4"></line>
                    <line x1="6" y1="20" x2="6" y2="14"></line>
                </svg>
            </div>
            <h3 style="color: #FFFFFF; margin-bottom: 0.5rem;">No Data Yet</h3>
            <p class="comparison-text">Give feedback on the "Get Motivation" tab or complete comparisons in "Build Your Profile" to see your progress here.</p>
        </div>
        """, unsafe_allow_html=True)

# Footer
st.markdown("")
st.markdown("")
st.markdown("""
<div style='text-align: center; padding: 2rem 0; border-top: 1px solid #2E2E2E;'>
    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#505050" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="display: inline-block;">
        <circle cx="12" cy="12" r="10"></circle>
        <path d="M9.09 9a3 3 0 0 1 5.83 1c0 2-3 3-3 3"></path>
        <line x1="12" y1="17" x2="12.01" y2="17"></line>
    </svg>
</div>
""", unsafe_allow_html=True)
