# AI Learning Assistant for Computer Science Students

一个面向计算机类大学生学习场景的智能问答系统原型。项目基于 Flask 构建 Web 应用，接入 DeepSeek Chat API，结合本地知识库检索、会话持久化、学习场景引导和流式回答，帮助学生在课程学习、编程调试、考试复习和项目实践中获得更结构化的辅导。

## Features

- User registration, login, profile editing, and avatar selection.
- Multi-session chat history stored in SQLite.
- DeepSeek API based answer generation.
- Server-Sent Events streaming for a ChatGPT-like response experience.
- Scene-aware tutoring modes for general study, exams, coding, and projects.
- Local FAQ and rule matching for simple high-confidence answers.
- RAG workflow powered by ChromaDB and `knowledge_base.txt`.
- Structured answer metadata such as summary, category, confidence, source, and scene reminders.
- Optional browser voice input on supported browsers.

## Architecture

```text
Flask app
├── Auth and profile routes
├── Chat and streaming routes
├── Scene guidance and local FAQ matching
├── DeepSeek API client
├── SQLite persistence
└── RAG integration
    ├── knowledge_base.txt
    ├── build_index.py
    └── rag_core.py / ask.py
```

Main files:

- `app.py` - Flask application, API routes, chat orchestration, persistence, and scene logic.
- `rag_core.py` - ChromaDB retrieval helpers.
- `build_index.py` - Builds the local vector index from `knowledge_base.txt`.
- `ask.py` - Command-line RAG question answering demo.
- `templates/` - Login, profile, and chat pages.
- `static/` - Frontend assets and default assistant avatars.

## Requirements

- Python 3.8 or newer. Python 3.11 is recommended.
- A DeepSeek API key.

Install dependencies:

```bash
pip install -r requirements.txt
```

## Configuration

Set environment variables before starting the app.

Linux / macOS:

```bash
export DEEPSEEK_API_KEY="your DeepSeek API key"
export FLASK_SECRET_KEY="a strong random secret"
```

Windows PowerShell:

```powershell
$env:DEEPSEEK_API_KEY="your DeepSeek API key"
$env:FLASK_SECRET_KEY="a strong random secret"
```

Optional:

```bash
export DEEPSEEK_MODEL="deepseek-chat"
```

## Run

Build the local knowledge-base index first if you want to use RAG:

```bash
python build_index.py
```

Start the web app:

```bash
python app.py
```

Open:

```text
http://127.0.0.1:5000
```

Run the command-line RAG demo:

```bash
python ask.py
```

## Runtime Data

The repository intentionally excludes runtime and private local data:

- `chat_history.db`
- `chat_history.db-wal`
- `chat_history.db-shm`
- `chroma_db/`
- uploaded user avatars
- `.env` files
- local reports, slides, backups, and deployment archives

This keeps the public repository focused on reusable source code while preventing chat history, generated indexes, and private project documents from being published accidentally.

## Roadmap

- Add automated tests for authentication, chat, and RAG flows.
- Add a production deployment guide using Gunicorn and Nginx.
- Improve knowledge-base ingestion and source citation quality.
- Add configurable prompt templates for different CS courses.
- Add evaluation scripts for answer quality and retrieval accuracy.

## License

This project is released under the MIT License.
