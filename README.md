# Chat with SQL (Text-to-Database Explorer & Schema Workspace)

A dual-provider, privacy-first Text-to-SQL application powered by **Google AI Studio (Gemini)**, **Local Ollama**, **Streamlit**, and **SQLite**. Designed specifically as a college viva companion and robust developer workspace, featuring dual workspace modes for read-only querying and human-in-the-loop schema & trigger management.

---

## 1. Dual AI Provider Setup

**Chat with SQL** supports two seamless AI backends selectable instantly from the Streamlit sidebar:
1. **Google AI Studio (Free API)** *(Recommended)*:
   - Uses zero-setup cloud models (`gemini-3.5-flash-lite`, `gemini-3.1-flash-lite`, `gemma-3-27b-it`, `gemini-3.8-flash`) with automatic fallback on rate limit (429).
   - **3 Secure Ways to Slot in `GEMINI_API_KEY`**:
     1. **Sidebar Password Box**: Paste your key directly into the encrypted password field in the Streamlit sidebar.
     2. **Environment Variable**: Set `GEMINI_API_KEY=your_key` in your OS environment.
     3. **`.env` File**: Copy `.env.example` to `.env` in `D:\chat-with-sql` and add your key.
2. **Local Ollama (Offline)**:
   - Runs 100% offline with zero cloud dependency (`qwen2.5-coder:3b`, `llama3.2:3b`).
   - Features **Low-End CPU Mode (2-core safe)** which caps `num_thread=2`, `num_ctx=2048`, and `num_predict=256` to keep dual-core student laptops fully responsive.

---

## 2. Quick Setup & Testing

### Step 1: Virtual Environment & Dependencies
```powershell
cd D:\chat-with-sql
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Step 2: Seed Database & Run Streamlit
```bash
python seed_db.py
streamlit run app.py
```

### Step 3: Run PyTest Suite
```bash
python -m pytest tests/test_app.py -v
```
