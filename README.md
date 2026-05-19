# Swastik AI

Swastik AI is a Flask-based AI chat web application with user authentication, chat session storage, and support for multiple AI providers.

## Key Features

- User registration and login with password hashing
- Persistent chat sessions and message history using PostgreSQL
- Multi-model AI routing across Gemini, Groq, SambaNova, and GitHub-hosted models
- Simple command handling for opening common websites from chat input
- Single-page web interface served by Flask

## Technology Stack

- Python
- Flask
- Flask-CORS
- Flask-SQLAlchemy
- Flask-Bcrypt
- PostgreSQL
- OpenAI SDK (for compatible providers)
- Google Generative AI SDK
- Gunicorn

## Project Structure

- `app.py` - Main Flask application, API routes, model routing, and database models
- `swastik_f.html` - Frontend UI template
- `requirements.txt` - Python dependencies

## API Endpoints

- `GET /` - Loads the web interface
- `POST /register` - Creates a new user account
- `POST /login` - Authenticates an existing user
- `POST /agent` - Processes user input and returns model or system-command responses

## Environment Variables

Set these in a `.env` file or environment:

- `DB_PASSWORD` - PostgreSQL password
- `DB_PORT` - PostgreSQL port
- `GEMINI_API_KEY` - Google Gemini API key
- `GROQ_API_KEY` - Groq API key
- `SAMBANOVA_KEY` - SambaNova API key
- `GITHUB_TOKEN` - GitHub models inference token
- `MISTRAL_KEY` - Mistral API key
- `HF_TOKEN` - Hugging Face token

## Setup and Run

1. Clone the repository.
2. Create and activate a Python virtual environment.
3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Ensure PostgreSQL is running.
5. Create the required database. In the current version, the app uses a hardcoded database name in `app.py`, so the database must be named `swastik_data` (this limits deployment flexibility until made configurable).

```bash
createdb swastik_data
```

6. Configure environment variables.
7. Start the app:

```bash
python app.py
```

The application runs on `http://localhost:5001` by default.

## Notes

- Database tables are created automatically at startup using `db.create_all()` once the PostgreSQL database exists.
- In development mode, Flask debug is enabled in `app.py`; never enable debug mode in production because it can expose sensitive data and remote code execution paths through the debugger. Use Gunicorn with debug disabled for production.
