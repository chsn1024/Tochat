import json
import os
import uuid
from typing import Any, Dict, List

import requests
from flask import Flask, jsonify, render_template, request, session

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "dev-secret-key")

API_URL = "https://api.deepseek.com/v1/chat/completions"
API_KEY = os.getenv("DEEPSEEK_API_KEY", "yourapi")
MODEL_NAME = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")

MAX_RECENT_MESSAGES = 8
SUMMARY_TRIGGER_MESSAGES = 12
SUMMARY_MAX_CHARS = 1200
DEFAULT_TIMEOUT = 60

conversation_store: Dict[str, Dict[str, Any]] = {}

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


def get_session_id() -> str:
    if "chat_session_id" not in session:
        session["chat_session_id"] = str(uuid.uuid4())
    return session["chat_session_id"]


def get_conversation_state() -> Dict[str, Any]:
    session_id = get_session_id()
    if session_id not in conversation_store:
        conversation_store[session_id] = {
            "messages": [],
            "summary": "",
        }
    return conversation_store[session_id]


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
        "Authorization": f"Bearer {API_KEY}",
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


def safe_parse_structured_content(raw_content: str) -> Dict[str, str]:
    try:
        parsed = json.loads(raw_content)
    except json.JSONDecodeError:
        return {
            "answer": raw_content,
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

    return {
        "answer": answer,
        "summary": summary,
        "category": category,
        "confidence": confidence,
    }


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


@app.route("/")
def index():
    return render_template(
        "index.html",
        scenes={key: value["label"] for key, value in SCENE_PROMPTS.items()},
    )


@app.route("/chat", methods=["POST"])
def chat():
    request_data = request.get_json(silent=True) or {}
    user_input = str(request_data.get("message", "")).strip()
    scene = str(request_data.get("scene", "general")).strip()
    structured_mode = bool(request_data.get("structured_mode", True))

    if not user_input:
        return jsonify({"error": "message 不能为空"}), 400
    if scene not in SCENE_PROMPTS:
        scene = "general"
    if API_KEY == "yourapi":
        return jsonify({"error": "请先配置 DEEPSEEK_API_KEY 环境变量"}), 500

    state = get_conversation_state()
    state["messages"].append({"role": "user", "content": user_input})
    compress_history_if_needed(state)

    try:
        response_data = call_deepseek(
            build_messages(state, scene, structured_mode),
            structured_mode=structured_mode,
        )
        raw_content = response_data["choices"][0]["message"]["content"]
    except requests.RequestException as exc:
        return jsonify({"error": f"DeepSeek API 调用失败: {exc}"}), 502

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

    return jsonify(
        {
            "reply": parsed_content["answer"],
            "structured": parsed_content,
            "scene": scene,
            "structured_mode": structured_mode,
            "history_count": len(state["messages"]),
            "has_summary": bool(state["summary"]),
        }
    )


@app.route("/reset", methods=["POST"])
def reset_chat():
    session_id = get_session_id()
    conversation_store[session_id] = {"messages": [], "summary": ""}
    return jsonify({"ok": True})


if __name__ == "__main__":
    app.run(debug=True)
