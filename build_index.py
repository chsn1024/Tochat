from rag_core import build_index


def main() -> None:
    chunk_count = build_index()
    print(f"知识库构建完成，共写入 {chunk_count} 个 chunk。")


if __name__ == "__main__":
    main()
