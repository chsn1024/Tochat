from flask import Flask, render_template, request, jsonify
import requests

app = Flask(__name__)

API_URL = "https://api.deepseek.com/v1/chat/completions"
API_KEY = "yourapi"  # 请替换为你的 API Key

# 聊天记录保存在用户 session 中（简化为全局变量）
chat_history = []

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    user_input = request.json["message"]
    chat_history.append({"role": "user", "content": user_input})

    payload = {
        "model": "deepseek-chat",
        "messages": chat_history,
        "temperature": 0.7,
        "max_tokens": 1024
    }

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }

    response = requests.post(API_URL, json=payload, headers=headers)
    data = response.json()

    assistant_reply = data["choices"][0]["message"]["content"]
    chat_history.append({"role": "assistant", "content": assistant_reply})

    return jsonify({"reply": assistant_reply})

if __name__ == "__main__":
    app.run(debug=True)