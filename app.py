import json
import os
import secrets
import sqlite3
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import requests
from flask import Flask, Response, jsonify, redirect, render_template, request, session, stream_with_context, url_for
from werkzeug.exceptions import HTTPException
from werkzeug.utils import secure_filename
from werkzeug.security import check_password_hash, generate_password_hash

from rag_core import RagNotReadyError, build_context, format_sources, retrieve

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY") or secrets.token_hex(32)

API_URL = "https://api.deepseek.com/v1/chat/completions"
MODEL_NAME = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")

MAX_RECENT_MESSAGES = 8
SUMMARY_TRIGGER_MESSAGES = 12
SUMMARY_MAX_CHARS = 1200
DEFAULT_TIMEOUT = 60
DB_PATH = os.path.join(os.path.dirname(__file__), "chat_history.db")
MAX_AVATAR_SIZE = 2 * 1024 * 1024
ALLOWED_AVATAR_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
USER_AVATAR_DIR = os.path.join(os.path.dirname(__file__), "static", "user_avatars")
AI_AVATAR_OPTIONS = [
    f"/static/ai_avatars/ai_avatar_{index:02d}.png"
    for index in range(1, 11)
]
DEFAULT_AI_AVATAR = AI_AVATAR_OPTIONS[0]

conversation_store: Dict[str, Dict[str, Any]] = {}


@app.errorhandler(Exception)
def handle_unexpected_error(error: Exception):
    if request.path.startswith("/auth/") or request.path.startswith("/profile/") or request.path == "/ask":
        if isinstance(error, HTTPException):
            return jsonify({"error": error.description}), error.code
        return jsonify({"error": f"服务器错误: {error}"}), 500
    if isinstance(error, HTTPException):
        return error
    raise error


@app.route("/favicon.ico")
def favicon():
    return "", 204

BASE_SYSTEM_PROMPT = """
你是“面向计算机类大学生学习场景的智能问答助手”。

你的目标：
1. 优先服务计算机类大学生在课程学习、编程实践、算法训练、项目开发、论文阅读、面试准备中的问题。
2. 回答强调教学性、步骤性、可验证性，避免只给结论不解释。
3. 如果用户问题存在歧义，先指出关键前提，再给出合理假设下的回答。
4. 涉及代码时，优先给出能运行或易于修改的示例，并解释核心思路。
5. 不编造课程安排、实验要求、论文结论或不存在的 API。
6. 当问题超出已知信息时，明确说明不确定性。

输出风格约束：
1. 默认使用中文回答，保留必要英文术语。
2. 优先使用清晰的小标题、步骤、示例、注意事项。
3. 如果用户只是要简短答案，保持精炼。
4. 不输出与学习场景无关的闲聊内容。
""".strip()


def get_db_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, timeout=10)
    conn.row_factory = sqlite3.Row
    return conn


def get_deepseek_api_key() -> str:
    return os.getenv("DEEPSEEK_API_KEY", "")


def is_deepseek_api_key_configured() -> bool:
    api_key = get_deepseek_api_key().strip()
    return bool(api_key)


def format_deepseek_error(exc: requests.RequestException) -> str:
    response = getattr(exc, "response", None)
    if response is not None and response.status_code == 401:
        return (
            "DeepSeek API 鉴权失败（401）。请确认已经在启动 Flask 的同一个 PowerShell "
            "窗口中设置了正确的 DEEPSEEK_API_KEY，然后重启 python app.py。"
        )
    return f"DeepSeek API 调用失败: {exc}"


def init_db() -> None:
    with get_db_connection() as conn:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=10000")
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS conversations (
                session_id TEXT PRIMARY KEY,
                user_id INTEGER,
                summary TEXT NOT NULL DEFAULT '',
                updated_at TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY(user_id) REFERENCES users(id)
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                email TEXT NOT NULL UNIQUE,
                username TEXT NOT NULL UNIQUE,
                nickname TEXT NOT NULL UNIQUE,
                avatar_url TEXT,
                password_hash TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY(session_id) REFERENCES conversations(session_id)
            )
            """
        )
        conversation_columns = {
            row["name"]
            for row in conn.execute("PRAGMA table_info(conversations)").fetchall()
        }
        if "user_id" not in conversation_columns:
            conn.execute("ALTER TABLE conversations ADD COLUMN user_id INTEGER")
        user_columns = {
            row["name"]
            for row in conn.execute("PRAGMA table_info(users)").fetchall()
        }
        if "avatar_url" not in user_columns:
            conn.execute("ALTER TABLE users ADD COLUMN avatar_url TEXT")


def get_current_user() -> Optional[Dict[str, Any]]:
    user_id = session.get("user_id")
    if not user_id:
        return None

    with get_db_connection() as conn:
        row = conn.execute(
            "SELECT id, email, username, nickname, avatar_url, created_at FROM users WHERE id = ?",
            (user_id,),
        ).fetchone()

    return dict(row) if row else None


def require_login_json() -> Optional[Any]:
    if get_current_user():
        return None
    return jsonify({"error": "请先登录"}), 401


def utc_now_iso() -> str:
    return datetime.utcnow().isoformat(timespec="seconds")


def load_conversation_state(session_id: str) -> Dict[str, Any]:
    with get_db_connection() as conn:
        conversation_row = conn.execute(
            "SELECT summary FROM conversations WHERE session_id = ?",
            (session_id,),
        ).fetchone()
        message_rows = conn.execute(
            "SELECT role, content FROM messages WHERE session_id = ? ORDER BY id ASC",
            (session_id,),
        ).fetchall()

    return {
        "messages": [
            {"role": row["role"], "content": row["content"]}
            for row in message_rows
        ],
        "summary": conversation_row["summary"] if conversation_row else "",
    }


def save_conversation_state(session_id: str, state: Dict[str, Any]) -> None:
    now = utc_now_iso()
    user_id = session.get("user_id")
    with get_db_connection() as conn:
        conn.execute(
            """
            INSERT INTO conversations (session_id, user_id, summary, updated_at, created_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(session_id) DO UPDATE SET
                user_id = excluded.user_id,
                summary = excluded.summary,
                updated_at = excluded.updated_at
            """,
            (session_id, user_id, state["summary"], now, now),
        )
        conn.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
        conn.executemany(
            """
            INSERT INTO messages (session_id, role, content, created_at)
            VALUES (?, ?, ?, ?)
            """,
            [
                (session_id, item["role"], item["content"], now)
                for item in state["messages"]
            ],
        )


def list_saved_sessions(limit: int = 20) -> List[Dict[str, Any]]:
    user_id = session.get("user_id")
    with get_db_connection() as conn:
        rows = conn.execute(
            """
            SELECT
                c.session_id,
                c.summary,
                c.updated_at,
                c.created_at,
                COUNT(m.id) AS message_count
            FROM conversations c
            LEFT JOIN messages m ON m.session_id = c.session_id
            WHERE c.user_id = ?
            GROUP BY c.session_id, c.summary, c.updated_at, c.created_at
            ORDER BY c.updated_at DESC
            LIMIT ?
            """,
            (user_id, limit),
        ).fetchall()

    sessions = []
    for row in rows:
        summary = row["summary"].strip()
        sessions.append(
            {
                "session_id": row["session_id"],
                "summary_preview": summary[:80] if summary else "暂无摘要，优先展示最近消息记录。",
                "updated_at": row["updated_at"],
                "created_at": row["created_at"],
                "message_count": row["message_count"],
            }
        )
    return sessions

SCENE_PROMPTS = {
    "general": {
        "label": "通用学习助手",
        "prompt": """
当前角色：通用学习助手。
适用于概念讲解、课程答疑、知识梳理。
回答时兼顾准确性与易懂性，优先帮助学生建立知识框架。
""".strip(),
    },
    "exam": {
        "label": "考研/考试辅导",
        "prompt": """
当前角色：考试辅导教练。
回答时突出考点、易错点、解题步骤和记忆方法。
如适合，给出“题型识别 -> 解法 -> 常见陷阱”的结构。
""".strip(),
    },
    "coding": {
        "label": "编程调试助手",
        "prompt": """
当前角色：编程调试助手。
回答时优先定位报错原因、复现条件、修复步骤和改进建议。
如用户提供代码，先指出问题，再给出修正版或排查顺序。
""".strip(),
    },
    "project": {
        "label": "项目实践导师",
        "prompt": """
当前角色：项目实践导师。
回答时优先从需求拆解、架构设计、技术选型、实施步骤、风险控制几个角度组织内容。
更关注方案落地性，而不是抽象定义。
""".strip(),
    },
}

STRUCTURED_RESPONSE_PROMPT = """
你必须返回一个 JSON 对象，不要输出 Markdown 代码块，不要输出 JSON 之外的任何内容。

JSON 必须包含以下字段：
- answer: string，给用户的最终回答
- summary: string，用一句话概括回答核心
- category: string，从以下类别中选择最合适的一项：
  ["概念讲解", "代码调试", "算法分析", "项目设计", "考试辅导", "学习规划", "职业发展", "其他"]
- confidence: string，只能为 ["高", "中", "低"] 之一

要求：
1. answer 字段可使用 Markdown。
2. summary 字段必须简短。
3. category 和 confidence 必须严格使用给定枚举值。
4. 如果用户输入不完整，也要在 answer 中说明缺失信息。
""".strip()

RAG_SYSTEM_PROMPT = """
你正在使用本地知识库回答问题。

要求：
1. 必须严格根据“知识库检索片段”回答。
2. 如果片段中没有明确答案，请说明“知识库中没有找到明确相关信息”。
3. 不要编造学校政策、学院规则或不存在的通知。
4. 涉及保研、奖学金、竞赛加分等正式规则时，提醒用户以学院和学校当年正式通知为准。
5. 回答最后要列出“引用来源”，使用片段中给出的 source 和 chunk_id。
""".strip()

LOW_CONFIDENCE_HINTS = (
    "信息不足",
    "缺少",
    "无法确定",
    "无法准确",
    "无法直接",
    "无法判断",
    "不能确定",
    "未提供",
    "没有提供",
    "需要更多信息",
    "需要补充",
    "题目不完整",
    "代码未提供",
    "无法保证",
    "不确定",
)

WEAK_INPUT_HINTS = (
    "未贴出",
    "没贴出",
    "不把代码贴出来",
    "没有代码",
    "没有题目",
    "没有截图",
)

UNCERTAINTY_SOFTENERS = (
    "通常",
    "一般来说",
    "可能",
    "也许",
    "大概率",
)

ANSWER_SUBSTANCE_HINTS = (
    "步骤",
    "原因",
    "示例",
    "代码",
    "复杂度",
    "结论",
    "建议",
    "可以",
    "需要",
    "如下",
)

SCENE_KEYWORDS = {
    "general": (
        "课程",
        "学习",
        "考试",
        "考研",
        "算法",
        "数据结构",
        "操作系统",
        "计算机网络",
        "数据库",
        "编程",
        "代码",
        "调试",
        "项目",
        "论文",
        "面试",
        "java",
        "python",
        "c++",
        "flask",
        "前端",
        "后端",
    ),
    "exam": (
        "考试",
        "考研",
        "408",
        "真题",
        "考点",
        "选择题",
        "简答题",
        "复习",
        "刷题",
        "知识点",
        "题型",
        "分数",
        "备考",
        "错题",
    ),
    "coding": (
        "代码",
        "报错",
        "异常",
        "bug",
        "调试",
        "编译",
        "运行",
        "函数",
        "接口",
        "java",
        "python",
        "c++",
        "javascript",
        "sql",
        "错误",
        "失败",
        "日志",
        "启动",
        "traceback",
        "stack",
    ),
    "project": (
        "项目",
        "系统",
        "需求",
        "模块",
        "架构",
        "设计",
        "数据库",
        "部署",
        "技术选型",
        "实现",
        "功能",
        "流程图",
        "原型",
        "页面",
        "业务",
        "接口",
        "数据库表",
    ),
}

SCENE_GUIDANCE = {
    "general": "当前场景更适合课程学习、概念理解、算法训练、编程实践等计算机类学习问题。你也可以把问题改成与课程知识、代码实现或技术原理相关的提问。",
    "exam": "当前场景更适合考试辅导。你可以把问题改成考点梳理、题型分析、真题讲解或复习规划。",
    "coding": "当前场景更适合代码调试与问题排查。你可以补充报错信息、代码片段、运行环境或预期结果。",
    "project": "当前场景更适合项目实践。你可以从需求分析、模块设计、技术选型、数据库设计或实施步骤来提问。",
}

SCENE_MISMATCH_MIN_CHARS = 8

FAQ_INTENT_HINTS = (
    "是什么",
    "什么是",
    "介绍",
    "概念",
    "区别",
    "差异",
)

FAQ_COMPLEXITY_HINTS = (
    "对比",
    "比较",
    "和",
    "与",
    "vs",
    "优缺点",
    "适合",
    "为什么",
    "怎么",
    "如何",
    "报错",
    "代码",
    "实现",
    "项目",
    "案例",
)

FAQ_KEYWORD_ALIASES = {
    "什么是": ("什么是", "是什么"),
}

LOCAL_FAQ_RULES = [
    {
        "keywords": ("c语言", "介绍"),
        "structured": {
            "answer": (
                "C语言是一种通用的过程式编程语言，由 Dennis Ritchie 在 1970 年代初为 Unix 系统开发。\n\n"
                "它的特点是运行效率高、语法相对简洁、能够直接操作内存，因此常用于操作系统、编译器、嵌入式开发和高性能程序。\n\n"
                "学习 C 语言时，建议重点掌握变量与控制流、函数、数组、指针、结构体、文件操作和内存管理。"
            ),
            "summary": "C语言是一种高效、接近底层、广泛用于系统开发的过程式语言。",
            "category": "概念讲解",
            "confidence": "高",
        },
    },
    {
        "keywords": ("进程", "线程", "区别"),
        "structured": {
            "answer": (
                "进程是资源分配的基本单位，线程是 CPU 调度的基本单位。\n\n"
                "可以把进程理解为一个正在运行的程序实例，而线程是这个程序内部实际执行任务的执行流。"
                "同一进程内的线程共享内存空间，切换开销较小；不同进程之间相互隔离，安全性更高。"
            ),
            "summary": "进程负责资源隔离，线程负责执行调度，同进程线程共享资源。",
            "category": "概念讲解",
            "confidence": "高",
        },
    },
    {
        "keywords": ("数据库", "什么是"),
        "structured": {
            "answer": (
                "数据库是按照一定结构组织、存储和管理数据的系统，用来支持数据的高效查询、更新和维护。\n\n"
                "常见数据库分为关系型数据库和非关系型数据库。前者如 MySQL、PostgreSQL，适合结构化数据；"
                "后者如 Redis、MongoDB，适合缓存、文档或高并发等场景。"
            ),
            "summary": "数据库是用于组织、存储和管理数据的系统。",
            "category": "概念讲解",
            "confidence": "高",
        },
    },
]


def validate_auth_fields(email: str, username: str, nickname: str, password: str) -> str:
    if not email or "@" not in email:
        return "请输入有效邮箱"
    if not username or len(username) < 3:
        return "用户名至少需要 3 个字符"
    if not nickname or len(nickname) < 2:
        return "昵称至少需要 2 个字符"
    if not password or len(password) < 6:
        return "密码至少需要 6 位"
    return ""


def save_user_avatar(file_storage: Any, required: bool = True) -> str:
    if not file_storage or not file_storage.filename:
        if required:
            raise ValueError("注册时请选择用户头像")
        return ""

    original_filename = secure_filename(file_storage.filename)
    _, extension = os.path.splitext(original_filename.lower())
    if extension not in ALLOWED_AVATAR_EXTENSIONS:
        raise ValueError("头像仅支持 JPG、PNG 或 WebP 格式")

    file_storage.seek(0, os.SEEK_END)
    file_size = file_storage.tell()
    file_storage.seek(0)
    if file_size <= 0:
        raise ValueError("头像文件不能为空")
    if file_size > MAX_AVATAR_SIZE:
        raise ValueError("头像文件不能超过 2MB")

    os.makedirs(USER_AVATAR_DIR, exist_ok=True)
    filename = f"user_avatar_{uuid.uuid4().hex}{extension}"
    file_path = os.path.join(USER_AVATAR_DIR, filename)
    file_storage.save(file_path)
    return f"/static/user_avatars/{filename}"


def get_session_id() -> str:
    if "chat_session_id" not in session:
        session["chat_session_id"] = str(uuid.uuid4())
    return session["chat_session_id"]


def get_conversation_state() -> Dict[str, Any]:
    session_id = get_session_id()
    if session_id not in conversation_store:
        conversation_store[session_id] = load_conversation_state(session_id)
    return conversation_store[session_id]


def switch_session(session_id: str) -> Dict[str, Any]:
    session["chat_session_id"] = session_id
    state = load_conversation_state(session_id)
    conversation_store[session_id] = state
    return state


def build_system_prompt(scene: str, structured_mode: bool) -> str:
    scene_prompt = SCENE_PROMPTS.get(scene, SCENE_PROMPTS["general"])["prompt"]
    prompt_parts = [BASE_SYSTEM_PROMPT, scene_prompt]
    if structured_mode:
        prompt_parts.append(STRUCTURED_RESPONSE_PROMPT)
    return "\n\n".join(prompt_parts)


def build_messages(state: Dict[str, Any], scene: str, structured_mode: bool) -> List[Dict[str, str]]:
    messages = [{"role": "system", "content": build_system_prompt(scene, structured_mode)}]

    if state["summary"]:
        messages.append(
            {
                "role": "system",
                "content": f"以下是此前多轮对话的摘要，请延续上下文但避免重复：\n{state['summary']}",
            }
        )

    messages.extend(state["messages"][-MAX_RECENT_MESSAGES:])
    return messages


def build_rag_messages(
    state: Dict[str, Any],
    scene: str,
    structured_mode: bool,
    rag_context: str,
) -> List[Dict[str, str]]:
    messages = build_messages(state, scene, structured_mode)
    messages.insert(
        1,
        {
            "role": "system",
            "content": f"{RAG_SYSTEM_PROMPT}\n\n【知识库检索片段】\n{rag_context}",
        },
    )
    return messages


def prepare_rag_context(question: str, top_k: int = 3) -> Tuple[str, List[Dict[str, Any]]]:
    docs, metadatas = retrieve(question, top_k=top_k)
    return build_context(docs, metadatas), format_sources(metadatas)


def call_deepseek(messages: List[Dict[str, str]], structured_mode: bool) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "model": MODEL_NAME,
        "messages": messages,
        "temperature": 0.5,
        "max_tokens": 1200,
    }
    if structured_mode:
        payload["response_format"] = {"type": "json_object"}

    headers = {
        "Authorization": f"Bearer {get_deepseek_api_key()}",
        "Content-Type": "application/json",
    }

    response = requests.post(
        API_URL,
        json=payload,
        headers=headers,
        timeout=DEFAULT_TIMEOUT,
    )
    response.raise_for_status()
    return response.json()


def iter_deepseek_content(messages: List[Dict[str, str]], structured_mode: bool):
    payload: Dict[str, Any] = {
        "model": MODEL_NAME,
        "messages": messages,
        "temperature": 0.5,
        "max_tokens": 1200,
        "stream": True,
    }
    if structured_mode:
        payload["response_format"] = {"type": "json_object"}

    headers = {
        "Authorization": f"Bearer {get_deepseek_api_key()}",
        "Content-Type": "application/json",
    }

    with requests.post(
        API_URL,
        json=payload,
        headers=headers,
        timeout=DEFAULT_TIMEOUT,
        stream=True,
    ) as response:
        response.raise_for_status()
        for line in response.iter_lines(decode_unicode=True):
            if not line or not line.startswith("data:"):
                continue

            data = line[len("data:") :].strip()
            if data == "[DONE]":
                break

            try:
                chunk = json.loads(data)
            except json.JSONDecodeError:
                continue

            delta = chunk.get("choices", [{}])[0].get("delta", {})
            content = delta.get("content")
            if content:
                yield content


def sse_event(event: str, data: Dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def extract_streaming_answer_delta(buffer: str, cursor: int) -> Tuple[str, int]:
    marker = '"answer"'
    key_index = buffer.find(marker)
    if key_index == -1:
        return "", cursor

    colon_index = buffer.find(":", key_index + len(marker))
    if colon_index == -1:
        return "", cursor

    start_quote = buffer.find('"', colon_index + 1)
    if start_quote == -1:
        return "", cursor

    if cursor < start_quote + 1:
        cursor = start_quote + 1

    chars = []
    index = cursor
    while index < len(buffer):
        char = buffer[index]
        if char == "\\":
            if index + 1 >= len(buffer):
                break
            escaped = buffer[index + 1]
            if escaped == "n":
                chars.append("\n")
            elif escaped == "r":
                chars.append("\r")
            elif escaped == "t":
                chars.append("\t")
            elif escaped in {'"', "\\", "/"}:
                chars.append(escaped)
            elif escaped == "u":
                if index + 5 >= len(buffer):
                    break
                hex_value = buffer[index + 2 : index + 6]
                try:
                    chars.append(chr(int(hex_value, 16)))
                except ValueError:
                    chars.append("\\u" + hex_value)
                index += 4
            else:
                chars.append(escaped)
            index += 2
            continue

        if char == '"':
            return "".join(chars), index

        chars.append(char)
        index += 1

    return "".join(chars), index


def strip_json_code_fence(raw_content: str) -> str:
    content = raw_content.strip()
    if not content.startswith("```"):
        return content

    lines = content.splitlines()
    if len(lines) >= 2 and lines[0].strip().startswith("```") and lines[-1].strip() == "```":
        return "\n".join(lines[1:-1]).strip()
    return content


def extract_json_object(raw_content: str) -> str:
    content = strip_json_code_fence(raw_content)
    start = content.find("{")
    end = content.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return content
    return content[start : end + 1]


def safe_parse_structured_content(raw_content: str) -> Dict[str, str]:
    content = raw_content.strip()
    if not content:
        return {
            "answer": "模型未返回可显示的回答内容，请重试或关闭结构化输出后再提问。",
            "summary": "模型返回内容为空。",
            "category": "其他",
            "confidence": "低",
        }

    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        try:
            parsed = json.loads(extract_json_object(content))
        except json.JSONDecodeError:
            return {
                "answer": content,
                "summary": "模型未按 JSON 返回，已降级为普通文本。",
                "category": "其他",
                "confidence": "低",
            }

    answer = str(parsed.get("answer", "")).strip() or "模型未提供 answer 字段。"
    summary = str(parsed.get("summary", "")).strip() or "未生成摘要。"
    category = str(parsed.get("category", "其他")).strip() or "其他"
    confidence = str(parsed.get("confidence", "低")).strip() or "低"

    if category not in {"概念讲解", "代码调试", "算法分析", "项目设计", "考试辅导", "学习规划", "职业发展", "其他"}:
        category = "其他"
    if confidence not in {"高", "中", "低"}:
        confidence = "低"

    confidence = normalize_confidence(answer, confidence)

    return {
        "answer": answer,
        "summary": summary,
        "category": category,
        "confidence": confidence,
    }


def normalize_confidence(answer: str, confidence: str) -> str:
    answer_lower = answer.lower()

    if any(hint in answer for hint in LOW_CONFIDENCE_HINTS + WEAK_INPUT_HINTS):
        return "低"

    if any(hint.lower() in answer_lower for hint in WEAK_INPUT_HINTS):
        return "低"

    has_softener = any(hint.lower() in answer_lower for hint in UNCERTAINTY_SOFTENERS)
    has_substance = any(hint.lower() in answer_lower for hint in ANSWER_SUBSTANCE_HINTS)
    if confidence == "高" and has_softener and not has_substance:
        return "中"

    return confidence


def detect_scene_mismatch(user_input: str, scene: str) -> str:
    if scene not in SCENE_KEYWORDS or scene == "general":
        return ""

    normalized_input = user_input.lower()
    compact_input = "".join(normalized_input.split())
    if len(compact_input) < SCENE_MISMATCH_MIN_CHARS:
        return ""

    matched_scene_keywords = [
        keyword for keyword in SCENE_KEYWORDS[scene]
        if keyword.lower() in normalized_input
    ]
    if matched_scene_keywords:
        return ""

    return (
        f"当前选择的是“{SCENE_PROMPTS[scene]['label']}”，"
        "但这条提问暂时没有明显命中该场景的关键词。"
        f"{SCENE_GUIDANCE[scene]}"
    )


def faq_keyword_matches(keyword: str, normalized_input: str) -> bool:
    aliases = FAQ_KEYWORD_ALIASES.get(keyword, (keyword,))
    return any(alias in normalized_input for alias in aliases)


def match_local_faq(user_input: str) -> Dict[str, str]:
    normalized_input = "".join(user_input.lower().split())
    for rule in LOCAL_FAQ_RULES:
        keywords = tuple(keyword.lower().replace(" ", "") for keyword in rule["keywords"])
        if not all(faq_keyword_matches(keyword, normalized_input) for keyword in keywords):
            continue

        has_faq_intent = any(hint in normalized_input for hint in FAQ_INTENT_HINTS)
        allowed_complexity_hints = set()
        if "区别" in keywords or "差异" in keywords:
            allowed_complexity_hints.update({"和", "与", "vs", "对比", "比较"})
        has_extra_complexity = any(
            hint in normalized_input
            and hint not in keywords
            and hint not in allowed_complexity_hints
            for hint in FAQ_COMPLEXITY_HINTS
        )
        if has_faq_intent and not has_extra_complexity:
            return dict(rule["structured"])
    return {}


def compress_history_if_needed(state: Dict[str, Any]) -> None:
    if len(state["messages"]) <= SUMMARY_TRIGGER_MESSAGES:
        return

    messages_to_summarize = state["messages"][:-MAX_RECENT_MESSAGES]
    if not messages_to_summarize:
        return

    transcript_lines = []
    for item in messages_to_summarize:
        role = "用户" if item["role"] == "user" else "助手"
        transcript_lines.append(f"{role}: {item['content']}")
    transcript = "\n".join(transcript_lines)

    summary_prompt = [
        {
            "role": "system",
            "content": (
                "请将以下多轮对话压缩为简洁上下文摘要，保留："
                "用户目标、关键约束、已经给出的结论、未解决的问题。"
                f"输出纯文本，控制在 {SUMMARY_MAX_CHARS} 字以内。"
            ),
        },
        {"role": "user", "content": transcript},
    ]

    try:
        response_data = call_deepseek(summary_prompt, structured_mode=False)
        summary_text = response_data["choices"][0]["message"]["content"].strip()
    except Exception:
        summary_text = transcript[-SUMMARY_MAX_CHARS:]

    state["summary"] = summary_text[:SUMMARY_MAX_CHARS]
    state["messages"] = state["messages"][-MAX_RECENT_MESSAGES:]


@app.route("/auth")
def auth_page():
    if get_current_user():
        return redirect(url_for("index"))
    return render_template("auth.html")


@app.route("/auth/register", methods=["POST"])
def register():
    if request.form or request.files:
        email = str(request.form.get("email", "")).strip().lower()
        username = str(request.form.get("username", "")).strip()
        nickname = str(request.form.get("nickname", "")).strip()
        password = str(request.form.get("password", ""))
        avatar_file = request.files.get("avatar")
    else:
        request_data = request.get_json(silent=True) or {}
        email = str(request_data.get("email", "")).strip().lower()
        username = str(request_data.get("username", "")).strip()
        nickname = str(request_data.get("nickname", "")).strip()
        password = str(request_data.get("password", ""))
        avatar_file = None

    error = validate_auth_fields(email, username, nickname, password)
    if error:
        return jsonify({"error": error}), 400
    try:
        avatar_url = save_user_avatar(avatar_file)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    now = utc_now_iso()
    try:
        with get_db_connection() as conn:
            conn.execute(
                """
                INSERT INTO users (email, username, nickname, avatar_url, password_hash, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (email, username, nickname, avatar_url, generate_password_hash(password), now),
            )
    except sqlite3.IntegrityError:
        if avatar_url:
            avatar_path = os.path.join(os.path.dirname(__file__), avatar_url.lstrip("/").replace("/", os.sep))
            if os.path.exists(avatar_path):
                os.remove(avatar_path)
        return jsonify({"error": "邮箱、用户名或昵称已存在"}), 409

    return jsonify({"ok": True, "message": "register success"})


@app.route("/auth/login", methods=["POST"])
def login():
    request_data = request.get_json(silent=True) or {}
    nickname = str(request_data.get("nickname", "")).strip()
    password = str(request_data.get("password", ""))

    if not nickname or not password:
        return jsonify({"error": "昵称和密码不能为空"}), 400

    with get_db_connection() as conn:
        user = conn.execute(
            "SELECT id, email, username, nickname, avatar_url, password_hash, created_at FROM users WHERE nickname = ?",
            (nickname,),
        ).fetchone()

    if not user or not check_password_hash(user["password_hash"], password):
        return jsonify({"error": "昵称或密码错误"}), 401

    session.clear()
    session["user_id"] = user["id"]
    session["chat_session_id"] = str(uuid.uuid4())

    return jsonify(
        {
            "ok": True,
            "message": "login success",
            "user": {
                "id": user["id"],
                "email": user["email"],
                "username": user["username"],
                "nickname": user["nickname"],
                "avatar_url": user["avatar_url"],
            },
        }
    )


@app.route("/auth/logout", methods=["POST"])
def logout():
    session.clear()
    return jsonify({"ok": True})


@app.route("/auth/me", methods=["GET"])
def auth_me():
    user = get_current_user()
    if not user:
        return jsonify({"authenticated": False}), 401
    return jsonify({"authenticated": True, "user": user})


@app.route("/profile")
def profile_page():
    current_user = get_current_user()
    if not current_user:
        return redirect(url_for("auth_page"))
    return render_template(
        "profile.html",
        current_user=current_user,
        default_avatar=DEFAULT_AI_AVATAR,
    )


@app.route("/profile/update", methods=["POST"])
def update_profile():
    auth_error = require_login_json()
    if auth_error:
        return auth_error

    current_user = get_current_user()
    if not current_user:
        return jsonify({"error": "请先登录"}), 401

    nickname = str(request.form.get("nickname", "")).strip()
    avatar_file = request.files.get("avatar")

    if not nickname or len(nickname) < 2:
        return jsonify({"error": "昵称至少需要 2 个字符"}), 400

    try:
        new_avatar_url = save_user_avatar(avatar_file, required=False)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    old_avatar_url = current_user.get("avatar_url") or ""
    try:
        with get_db_connection() as conn:
            existing = conn.execute(
                "SELECT id FROM users WHERE nickname = ? AND id != ?",
                (nickname, current_user["id"]),
            ).fetchone()
            if existing:
                if new_avatar_url:
                    new_avatar_path = os.path.join(
                        os.path.dirname(__file__),
                        new_avatar_url.lstrip("/").replace("/", os.sep),
                    )
                    if os.path.exists(new_avatar_path):
                        os.remove(new_avatar_path)
                return jsonify({"error": "昵称已存在"}), 409

            avatar_url = new_avatar_url or old_avatar_url
            conn.execute(
                "UPDATE users SET nickname = ?, avatar_url = ? WHERE id = ?",
                (nickname, avatar_url, current_user["id"]),
            )
    except sqlite3.Error as exc:
        return jsonify({"error": f"资料更新失败: {exc}"}), 500

    if new_avatar_url and old_avatar_url and old_avatar_url.startswith("/static/user_avatars/"):
        old_avatar_path = os.path.join(
            os.path.dirname(__file__),
            old_avatar_url.lstrip("/").replace("/", os.sep),
        )
        if os.path.exists(old_avatar_path):
            os.remove(old_avatar_path)

    updated_user = get_current_user()
    return jsonify({"ok": True, "message": "profile updated", "user": updated_user})


@app.route("/")
def index():
    current_user = get_current_user()
    if not current_user:
        return redirect(url_for("auth_page"))
    return render_template(
        "index.html",
        scenes={key: value["label"] for key, value in SCENE_PROMPTS.items()},
        current_user=current_user,
        ai_avatar_options=AI_AVATAR_OPTIONS,
        default_ai_avatar=DEFAULT_AI_AVATAR,
    )


@app.route("/sessions", methods=["GET"])
def get_sessions():
    auth_error = require_login_json()
    if auth_error:
        return auth_error
    return jsonify(
        {
            "sessions": list_saved_sessions(),
            "current_session_id": get_session_id(),
        }
    )


@app.route("/sessions/load", methods=["POST"])
def load_session():
    auth_error = require_login_json()
    if auth_error:
        return auth_error
    request_data = request.get_json(silent=True) or {}
    session_id = str(request_data.get("session_id", "")).strip()
    if not session_id:
        return jsonify({"error": "session_id 不能为空"}), 400

    with get_db_connection() as conn:
        row = conn.execute(
            "SELECT session_id FROM conversations WHERE session_id = ? AND user_id = ?",
            (session_id, session.get("user_id")),
        ).fetchone()
    if not row:
        return jsonify({"error": "会话不存在或无权访问"}), 404

    state = switch_session(session_id)
    return jsonify(
        {
            "ok": True,
            "session_id": session_id,
            "messages": state["messages"],
            "has_summary": bool(state["summary"]),
            "history_count": len(state["messages"]),
        }
    )


@app.route("/chat", methods=["POST"])
def chat():
    auth_error = require_login_json()
    if auth_error:
        return auth_error
    request_data = request.get_json(silent=True) or {}
    user_input = str(request_data.get("message", "")).strip()
    scene = str(request_data.get("scene", "general")).strip()
    structured_mode = bool(request_data.get("structured_mode", True))
    rag_mode = bool(request_data.get("rag_mode", False))

    if not user_input:
        return jsonify({"error": "message 不能为空"}), 400
    if scene not in SCENE_PROMPTS:
        scene = "general"
    if not is_deepseek_api_key_configured():
        return jsonify({"error": "请先配置 DEEPSEEK_API_KEY 环境变量"}), 500
    scene_reminder = detect_scene_mismatch(user_input, scene)
    rag_context = ""
    rag_sources: List[Dict[str, Any]] = []
    if rag_mode:
        try:
            rag_context, rag_sources = prepare_rag_context(user_input, top_k=3)
        except RagNotReadyError as exc:
            return jsonify({"error": str(exc)}), 500

    state = get_conversation_state()
    session_id = get_session_id()
    state["messages"].append({"role": "user", "content": user_input})
    compress_history_if_needed(state)

    local_faq_result = {} if rag_mode else match_local_faq(user_input)
    source = "knowledge_base" if rag_mode else "local_faq" if local_faq_result else "deepseek"

    if local_faq_result:
        parsed_content = (
            local_faq_result
            if structured_mode
            else {
                "answer": local_faq_result["answer"],
                "summary": "本地 FAQ 命中，未额外生成摘要。",
                "category": local_faq_result["category"],
                "confidence": local_faq_result["confidence"],
            }
        )
    else:
        try:
            messages = (
                build_rag_messages(state, scene, structured_mode, rag_context)
                if rag_mode
                else build_messages(state, scene, structured_mode)
            )
            response_data = call_deepseek(
                messages,
                structured_mode=structured_mode,
            )
            raw_content = response_data["choices"][0]["message"]["content"]
        except requests.RequestException as exc:
            return jsonify({"error": format_deepseek_error(exc)}), 502

        parsed_content = (
            safe_parse_structured_content(raw_content)
            if structured_mode
            else {
                "answer": raw_content,
                "summary": "普通文本模式未生成摘要。",
                "category": "其他",
                "confidence": "中",
            }
        )

    state["messages"].append({"role": "assistant", "content": parsed_content["answer"]})
    save_conversation_state(session_id, state)

    return jsonify(
        {
            "reply": parsed_content["answer"],
            "structured": parsed_content,
            "scene": scene,
            "structured_mode": structured_mode,
            "history_count": len(state["messages"]),
            "has_summary": bool(state["summary"]),
            "session_id": session_id,
            "scene_reminder": scene_reminder,
            "source": source,
            "rag_mode": rag_mode,
            "sources": rag_sources,
        }
    )


@app.route("/chat/stream", methods=["POST"])
def chat_stream():
    auth_error = require_login_json()
    if auth_error:
        return auth_error
    request_data = request.get_json(silent=True) or {}
    user_input = str(request_data.get("message", "")).strip()
    scene = str(request_data.get("scene", "general")).strip()
    structured_mode = bool(request_data.get("structured_mode", True))
    rag_mode = bool(request_data.get("rag_mode", False))

    if not user_input:
        return jsonify({"error": "message 不能为空"}), 400
    if scene not in SCENE_PROMPTS:
        scene = "general"
    if not is_deepseek_api_key_configured():
        return jsonify({"error": "请先配置 DEEPSEEK_API_KEY 环境变量"}), 500

    scene_reminder = detect_scene_mismatch(user_input, scene)
    rag_context = ""
    rag_sources: List[Dict[str, Any]] = []
    if rag_mode:
        try:
            rag_context, rag_sources = prepare_rag_context(user_input, top_k=3)
        except RagNotReadyError as exc:
            return jsonify({"error": str(exc)}), 500

    state = get_conversation_state()
    session_id = get_session_id()
    state["messages"].append({"role": "user", "content": user_input})
    compress_history_if_needed(state)

    local_faq_result = {} if rag_mode else match_local_faq(user_input)
    source = "knowledge_base" if rag_mode else "local_faq" if local_faq_result else "deepseek"

    @stream_with_context
    def generate():
        raw_content = ""
        streamed_answer = ""
        answer_cursor = 0

        yield sse_event(
            "start",
            {
                "scene": scene,
                "structured_mode": structured_mode,
                "has_summary": bool(state["summary"]),
                "session_id": session_id,
                "scene_reminder": scene_reminder,
                "source": source,
                "rag_mode": rag_mode,
                "sources": rag_sources,
            },
        )

        if local_faq_result:
            parsed_content = (
                local_faq_result
                if structured_mode
                else {
                    "answer": local_faq_result["answer"],
                    "summary": "本地 FAQ 命中，未额外生成摘要。",
                    "category": local_faq_result["category"],
                    "confidence": local_faq_result["confidence"],
                }
            )
            streamed_answer = parsed_content["answer"]
            yield sse_event("delta", {"content": streamed_answer})
        else:
            try:
                messages = (
                    build_rag_messages(state, scene, structured_mode, rag_context)
                    if rag_mode
                    else build_messages(state, scene, structured_mode)
                )
                for content in iter_deepseek_content(
                    messages,
                    structured_mode=structured_mode,
                ):
                    raw_content += content
                    if structured_mode:
                        delta, answer_cursor = extract_streaming_answer_delta(raw_content, answer_cursor)
                    else:
                        delta = content

                    if delta:
                        streamed_answer += delta
                        yield sse_event("delta", {"content": delta})
            except requests.RequestException as exc:
                yield sse_event("error", {"error": format_deepseek_error(exc)})
                return

            parsed_content = (
                safe_parse_structured_content(raw_content)
                if structured_mode
                else {
                    "answer": raw_content,
                    "summary": "普通文本模式未生成摘要。",
                    "category": "其他",
                    "confidence": "中",
                }
            )

            if structured_mode and not streamed_answer:
                streamed_answer = parsed_content["answer"]
                yield sse_event("delta", {"content": streamed_answer})

        state["messages"].append({"role": "assistant", "content": parsed_content["answer"]})
        save_conversation_state(session_id, state)

        yield sse_event(
            "done",
            {
                "reply": parsed_content["answer"],
                "structured": parsed_content,
                "scene": scene,
                "structured_mode": structured_mode,
                "history_count": len(state["messages"]),
                "has_summary": bool(state["summary"]),
                "session_id": session_id,
                "scene_reminder": scene_reminder,
                "source": source,
                "rag_mode": rag_mode,
                "sources": rag_sources,
            },
        )

    return Response(
        generate(),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.route("/ask", methods=["POST"])
def ask_knowledge_base():
    auth_error = require_login_json()
    if auth_error:
        return auth_error

    request_data = request.get_json(silent=True) or {}
    question = str(request_data.get("question", "")).strip()
    top_k = int(request_data.get("top_k", 3) or 3)

    if not question:
        return jsonify({"error": "question 不能为空"}), 400
    if not is_deepseek_api_key_configured():
        return jsonify({"error": "请先配置 DEEPSEEK_API_KEY 环境变量"}), 500

    try:
        rag_context, rag_sources = prepare_rag_context(question, top_k=top_k)
        response_data = call_deepseek(
            [
                {
                    "role": "system",
                    "content": f"{RAG_SYSTEM_PROMPT}\n\n【知识库检索片段】\n{rag_context}",
                },
                {"role": "user", "content": question},
            ],
            structured_mode=False,
        )
    except RagNotReadyError as exc:
        return jsonify({"error": str(exc)}), 500
    except requests.RequestException as exc:
        return jsonify({"error": format_deepseek_error(exc)}), 502

    answer = response_data["choices"][0]["message"]["content"].strip()
    return jsonify({"answer": answer, "sources": rag_sources})


@app.route("/reset", methods=["POST"])
def reset_chat():
    auth_error = require_login_json()
    if auth_error:
        return auth_error
    session_id = get_session_id()
    conversation_store[session_id] = {"messages": [], "summary": ""}
    save_conversation_state(session_id, conversation_store[session_id])
    return jsonify({"ok": True})


init_db()

if __name__ == "__main__":
    app.run(debug=True)
