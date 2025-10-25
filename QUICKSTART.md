# 🚀 Quick Start - MotivateMe AI

## Fix: "ModuleNotFoundError: No module named 'src'"

This happens when Python can't find the src module. Here are **3 ways to fix it**:

---

## ✅ **Method 1: Use the PowerShell Script (Easiest)**

```powershell
cd scaffold
.\run_demo.ps1
```

If you get a security error, run first:
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

---

## ✅ **Method 2: Manual Commands (Most Reliable)**

```powershell
# Navigate to scaffold directory
cd C:\Users\erixh\Downloads\tidal\scaffold

# Set Python path
$env:PYTHONPATH = "$PWD"

# Train the model
python src\reward_model.py

# Launch demo
streamlit run src\demo_app_enhanced.py
```

---

## ✅ **Method 3: Direct File Execution**

If the above don't work, run files directly:

```powershell
cd scaffold

# Change directory into src and run from there
cd src
python reward_model.py
cd ..

# Run streamlit from scaffold directory
streamlit run src\demo_app_enhanced.py
```

---

## 🎯 **What You'll See**

### Step 1: Training
```
Saved reward model to models/reward/reward_clf.joblib
```
✅ This means the AI learned from your 20 training examples!

### Step 2: Demo App
- Browser opens to `http://localhost:8501`
- You'll see 3 tabs:
  - **Compare Profiles** - Side-by-side motivation styles
  - **Chat Mode** - Interactive motivation
  - **Training Dashboard** - See the learning stats

---

## 🐛 **Troubleshooting**

### "No module named 'sentence_transformers'"
You haven't installed dependencies yet:
```powershell
cd scaffold
pip install -r requirements.txt
```

### "No such file or directory: 'config/settings.yaml'"
Make sure you're in the `scaffold` directory:
```powershell
cd C:\Users\erixh\Downloads\tidal\scaffold
pwd  # Should show: C:\Users\erixh\Downloads\tidal\scaffold
```

### "ModuleNotFoundError: No module named 'src'"
Use Method 2 above and make sure PYTHONPATH is set correctly.

### Demo app opens but shows errors
That's okay! You can still demo the concept. The UI will load even if some features fail.

---

## 📱 **For Your Presentation**

### Minimal Demo (if tech issues):
1. Open `data/comparisons.jsonl` in a text editor
2. **Read examples aloud** - The contrast is powerful even without the app!
3. Show the code to prove it's real

### Example to Read:
**Scenario:** "I've been procrastinating for 3 days"

**Transitional Style (You):**
> "I hear you - starting is always the hardest part, and three days of hesitation shows you care about doing it right. Now here's the reality: every day you wait makes it harder. Break it into the smallest possible first step and do that in the next 10 minutes."

**Disciplinarian Style (Teammate):**
> "Three days? Stop making excuses. You know what you need to do. Close this app, open your project, and start NOW. Not after you feel ready - you'll never feel ready. Discipline beats motivation every time."

**Ask judges:** "Which one would make YOU take action?"

The data alone proves the concept!

---

## 🎤 **Quick Presentation Checklist**

- [ ] Can you run `streamlit run src\demo_app_enhanced.py` successfully?
- [ ] Does the Compare Profiles tab show responses?
- [ ] Have you practiced reading 2-3 example pairs aloud?
- [ ] Do you have `data/comparisons.jsonl` open as backup?
- [ ] Can you explain RLHF in one sentence?

---

## 💡 **One-Sentence RLHF Explanation**

> "RLHF means the AI learns from human preferences - we show it examples of good vs bad responses, and it learns to generate more of what humans actually prefer."

---

**You've got this! 🚀**

The concept is strong, the data is ready, and even if the tech is temperamental, you can demo the idea compellingly.

