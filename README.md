# 面向计算机类大学生学习场景的智能问答助手

这是一个基于 `Flask + DeepSeek API` 的教学型聊天机器人示例项目，已扩展为适合大创项目展示的原型系统，重点加入了以下三类机制：

- 场景化角色与提示词约束机制
- 多轮对话上下文管理机制
- 结构化输出机制

## 当前可展示的创新点

### 1. 场景化角色与提示词约束机制

系统不再只有一个通用聊天入口，而是支持多种学习场景角色：

- 通用学习助手
- 考研/考试辅导
- 编程调试助手
- 项目实践导师

后端会根据前端选择的场景，动态拼接不同的 `system prompt`，从而约束模型回答风格、重点和输出结构。

### 2. 多轮对话上下文管理机制

系统采用“最近消息窗口 + 历史摘要压缩”的方式管理上下文：

- 保留最近若干轮原始消息
- 当消息数量超过阈值时，自动把较早对话压缩为摘要
- 在后续提问时，将“系统角色 + 历史摘要 + 最近对话”一起发送给模型

这样可以避免上下文无限增长，同时保留用户目标、关键约束和未解决问题。

### 3. 结构化输出机制

在结构化模式下，后端会请求 DeepSeek 返回 JSON 对象，并解析为：

- `answer`：最终回答
- `summary`：一句话摘要
- `category`：问题类别
- `confidence`：置信等级

如果模型未严格返回 JSON，后端会自动降级为普通文本并给出兜底字段，保证接口稳定。

## 系统架构

### 后端

文件：[app.py](/mnt/d/chat/app.py)

核心结构如下：

1. `conversation_store`
   按会话保存消息列表和历史摘要，替代原来的全局 `chat_history`。
2. `SCENE_PROMPTS`
   定义不同学习场景的角色提示词。
3. `build_messages()`
   构造发送给 DeepSeek 的最终消息，包括：

   - 基础系统角色
   - 场景提示词
   - 历史摘要
   - 最近多轮对话
4. `compress_history_if_needed()`
   在上下文过长时调用模型生成摘要，实现上下文压缩。
5. `safe_parse_structured_content()`
   解析结构化 JSON 输出，并在异常时自动降级。

### 前端

文件：[templates/index.html](/mnt/d/chat/templates/index.html)

新增能力：

- 场景选择器
- 结构化输出开关
- 会话重置按钮
- 结构化元信息展示区
- 更适合项目展示的界面布局

## 接口设计

### `POST /chat`

请求体示例：

```json
{
  "message": "请解释一下快速排序，并分析时间复杂度",
  "scene": "exam",
  "structured_mode": true
}
```

响应体示例：

```json
{
  "reply": "快速排序的核心思想是......",
  "structured": {
    "answer": "快速排序的核心思想是......",
    "summary": "快速排序通过分治与划分实现高效排序。",
    "category": "算法分析",
    "confidence": "高"
  },
  "scene": "exam",
  "structured_mode": true,
  "history_count": 6,
  "has_summary": false
}
```

### `POST /reset`

清空当前会话的消息和摘要。

## 运行方式

### 1. 运行环境

本项目需要 `Python 3.8` 或更高版本，推荐使用 `Python 3.11`。

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 配置环境变量

Linux / macOS:

```bash
export DEEPSEEK_API_KEY="你的 DeepSeek API Key"
export FLASK_SECRET_KEY="自定义 Flask Secret"
```

Windows PowerShell:

```powershell
$env:DEEPSEEK_API_KEY="你的 DeepSeek API Key"
$env:FLASK_SECRET_KEY="自定义 Flask Secret"
```

### 4. 启动项目

```bash
python app.py
```

然后访问：

```text
http://127.0.0.1:5000
```

## 适合在大创申报书中描述的设计思路

可以把本项目概括为“三层机制”：

- 提示词控制层：根据学习场景切换系统角色，控制模型回答边界和教学风格
- 上下文管理层：通过会话隔离、最近窗口保留、历史摘要压缩，提高多轮对话连续性
- 输出结构化层：将自然语言回答封装成标准 JSON 字段，便于后续做统计分析、学习画像和问答分类

这三层机制相比普通聊天机器人，更适合教学辅导、实验演示和后续功能扩展。

## 后续可继续扩展的方向

- 接入数据库或 Redis，替代内存态会话存储
- 增加知识库检索，实现课程资料增强问答
- 对 `category` 做统计分析，形成学生问题画像
- 根据 `confidence` 自动触发“补充提问”或“人工复核”机制
- 为不同课程单独设计提示词模板，例如数据结构、操作系统、计算机网络、数据库

## 本地知识库问答

项目支持基于 `knowledge_base.txt` 的 RAG 问答流程：

```bash
python build_index.py
python ask.py
```

网页端登录后打开“知识库增强回答”开关，即可让系统先检索本地知识库，再调用 DeepSeek 生成带引用来源的回答。

也可以直接调用接口：

```text
POST /ask
{
  "question": "大学绩点重要吗？",
  "top_k": 3
}
```
