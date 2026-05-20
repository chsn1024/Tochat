import os

from openai import OpenAI

from rag_core import build_context, format_sources, retrieve


deepseek_client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com",
)


def ask_deepseek(question: str, context: str) -> str:
    prompt = f"""
你是一个大学新生经验问答助手。

请严格根据下面的知识库内容回答用户问题。
如果知识库中没有明确答案，请回答：“知识库中没有找到明确相关信息。”
不要编造学校政策。
如果涉及保研、奖学金、竞赛加分等正式规则，请提醒用户以学院和学校当年通知为准。

【知识库内容】
{context}

【用户问题】
{question}

请给出清晰、简洁、有帮助的回答，并在最后列出引用来源。
""".strip()

    response = deepseek_client.chat.completions.create(
        model=os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content


def main() -> None:
    while True:
        question = input("\n请输入问题，输入 q 退出：").strip()
        if question.lower() == "q":
            break
        if not question:
            continue

        docs, metadatas = retrieve(question, top_k=3)
        context = build_context(docs, metadatas)
        answer = ask_deepseek(question, context)

        print("\n回答：")
        print(answer)
        print("\n引用来源：")
        for source in format_sources(metadatas):
            print(f"- {source['source']}，chunk_id：{source['chunk_id']}")


if __name__ == "__main__":
    main()
