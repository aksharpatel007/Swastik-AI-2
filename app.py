import datetime
import random
import requests
import os
import time
import concurrent.futures
import webbrowser  # <--- ADDED: To open websites
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from flask_bcrypt import Bcrypt
import google.generativeai as genai
from openai import OpenAI
from dotenv import load_dotenv

# --- CONFIGURATION ---
load_dotenv()

app = Flask(__name__)
CORS(app)

# DATABASE
db_pass = os.getenv("DB_PASSWORD", "akshar")
db_port = os.getenv("DB_PORT", "5433")
app.config['SQLALCHEMY_DATABASE_URI'] = f'postgresql://postgres:{db_pass}@localhost:{db_port}/swastik_data'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)
bcrypt = Bcrypt(app)

# --- 1. GOOGLE GEMINI ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

# --- 2. GROQ ---
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
groq_client = OpenAI(
    api_key=GROQ_API_KEY,
    base_url="https://api.groq.com/openai/v1"
) if GROQ_API_KEY else None

# Other Clients
SAMBANOVA_KEY = os.getenv("SAMBANOVA_KEY")
sambanova_client = OpenAI(api_key=SAMBANOVA_KEY, base_url="https://api.sambanova.ai/v1") if SAMBANOVA_KEY else None

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
github_client = OpenAI(api_key=GITHUB_TOKEN, base_url="https://models.inference.ai.azure.com") if GITHUB_TOKEN else None

MISTRAL_KEY = os.getenv("MISTRAL_KEY")
mistral_client = OpenAI(api_key=MISTRAL_KEY, base_url="https://api.mistral.ai/v1") if MISTRAL_KEY else None

HF_TOKEN = os.getenv("HF_TOKEN")

# --- MODEL ROUTING MAP ---
MODEL_MAP = {
    "gemini": {"provider": "google", "id": "gemini-2.5-flash-lite"},
    "gemini-stable": {"provider": "google", "id": "gemini-1.5-flash"},
    "llama-3.3-70b": {"provider": "groq", "id": "llama-3.3-70b-versatile"},
    "mixtral-8x7b": {"provider": "groq", "id": "mixtral-8x7b-32768"},
    "gemma2-9b": {"provider": "groq", "id": "gemma2-9b-it"},
    "deepseek-r1": {"provider": "sambanova", "id": "DeepSeek-R1-Distill-Llama-70B"},
    "gpt-4o": {"provider": "github", "id": "gpt-4o"},
}

# --- SYSTEM COMMANDS ---
# Map keywords to URLs
COMMANDS = {
    "google": "https://www.google.com",
    "youtube": "https://www.youtube.com",
    "instagram": "https://www.instagram.com",
    "facebook": "https://www.facebook.com",
    "twitter": "https://twitter.com",
    "x": "https://twitter.com",
    "linkedin": "https://www.linkedin.com",
    "github": "https://github.com",
    "stackoverflow": "https://stackoverflow.com",
    "chatgpt": "https://chat.openai.com",
    "whatsapp": "https://web.whatsapp.com",
    "gmail": "https://mail.google.com",
    "lju": "https://www.ljku.edu.in",  # Added your college for convenience
    "maps": "https://maps.google.com",
    "spotify": "https://open.spotify.com"
}


def process_system_commands(text):
    """
    Checks if the user input is a system command (like 'open youtube').
    Returns the response string if a command matches, else None.
    """
    text_clean = text.lower().strip()

    # Logic for "Open [Website]"
    if text_clean.startswith("open "):
        site_key = text_clean.replace("open ", "").strip()

        # Check exact match in dictionary
        if site_key in COMMANDS:
            url = COMMANDS[site_key]
            webbrowser.open(url)  # Opens on the server/local machine
            return f"Opening {site_key.capitalize()} for you..."

        # Fallback: If it looks like a domain (e.g., "open example.com")
        if "." in site_key:
            url = f"https://{site_key}" if not site_key.startswith("http") else site_key
            webbrowser.open(url)
            return f"Opening {site_key}..."

    return None


# --- DATABASE MODELS ---
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(50), nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)


class ChatSession(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    title = db.Column(db.String(100))
    mode = db.Column(db.String(20), default="direct")
    created_at = db.Column(db.DateTime, default=datetime.datetime.utcnow)


class Message(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    session_id = db.Column(db.Integer, db.ForeignKey('chat_session.id'), nullable=False)
    sender = db.Column(db.String(20), nullable=False)
    model_used = db.Column(db.String(50), nullable=True)
    content = db.Column(db.Text, nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.datetime.utcnow)


class BattleVote(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    session_id = db.Column(db.Integer, db.ForeignKey('chat_session.id'), nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    model_a = db.Column(db.String(50), nullable=False)
    model_b = db.Column(db.String(50), nullable=False)
    winner = db.Column(db.String(50), nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.datetime.utcnow)


# --- AI GENERATION LOGIC ---
def generate_ai_response(model_key, prompt, retries=2):
    config = MODEL_MAP.get(model_key, MODEL_MAP["gemini"])
    provider = config["provider"]
    model_id = config["id"]

    for attempt in range(retries + 1):
        try:
            if provider == "google":
                if not GEMINI_API_KEY: return "Gemini API Key missing."
                model = genai.GenerativeModel(model_id)
                return model.generate_content(prompt).text

            elif provider == "groq":
                if not groq_client: return "Groq API Key missing."
                resp = groq_client.chat.completions.create(
                    model=model_id,
                    messages=[{"role": "user", "content": prompt}]
                )
                return resp.choices[0].message.content

            elif provider == "sambanova":
                if not sambanova_client: return "SambaNova Key missing."
                resp = sambanova_client.chat.completions.create(
                    model=model_id,
                    messages=[{"role": "user", "content": prompt}]
                )
                return resp.choices[0].message.content

            elif provider == "github":
                if not github_client: return "GitHub Token missing."
                resp = github_client.chat.completions.create(
                    model=model_id,
                    messages=[{"role": "user", "content": prompt}]
                )
                return resp.choices[0].message.content

        except Exception as e:
            err_msg = str(e).lower()
            if "429" in err_msg or "quota" in err_msg:
                if attempt < retries:
                    time.sleep(1 + attempt)
                    continue
                if provider == "google" and groq_client:
                    return generate_ai_response("llama-3.3-70b", prompt)
            return f"Error with {model_key}: {str(e)}"


# --- ROUTES ---
@app.route("/")
def home():
    return render_template("swastik_f.html")


@app.route("/register", methods=["POST"])
def register():
    data = request.json
    if User.query.filter_by(email=data['email']).first():
        return jsonify({"error": "Email already exists"}), 400
    hashed = bcrypt.generate_password_hash(data['password']).decode('utf-8')
    user = User(username=data['username'], email=data['email'], password=hashed)
    db.session.add(user)
    db.session.commit()
    return jsonify({"message": "Success", "username": user.username})


@app.route("/login", methods=["POST"])
def login():
    data = request.json
    user = User.query.filter_by(email=data['email']).first()
    if user and bcrypt.check_password_hash(user.password, data['password']):
        return jsonify({"message": "Success", "user_id": user.id, "username": user.username})
    return jsonify({"error": "Invalid credentials"}), 401


@app.route("/agent", methods=["POST"])
def agent():
    data = request.json
    user_id = data.get("user_id")
    text = data.get("input", "")
    model = data.get("model", "gemini")

    if not user_id: return jsonify({"error": "Login required"}), 401

    session_id = data.get("session_id")
    if not session_id:
        new_session = ChatSession(user_id=user_id, title=text[:30], mode="direct")
        db.session.add(new_session)
        db.session.commit()
        session_id = new_session.id

    # 1. Save User Message
    db.session.add(Message(session_id=session_id, sender="user", content=text))

    # 2. Check for System Commands (Open YouTube, etc.)
    # We do this BEFORE calling the AI to save time and API quota
    command_response = process_system_commands(text)

    if command_response:
        # If it was a command, use that response
        response_text = command_response
        model_used = "system_command"
    else:
        # If not a command, ask the AI
        response_text = generate_ai_response(model, text)
        model_used = model

    # 3. Save Bot Response
    db.session.add(Message(session_id=session_id, sender="bot", content=response_text, model_used=model_used))
    db.session.commit()

    return jsonify({"response": response_text, "session_id": session_id})


# --- MAIN ---
if __name__ == "__main__":
    with app.app_context():
        db.create_all()
    app.run(port=5001, debug=True)