# 任务指南：从零构建 My-LangChain

> 这是本仓库的**权威任务指南（Roadmap / Master Plan）**。
> 目标：用纯 Python 从零实现一个与当前主流 LangChain（`langchain-core` 1.x 用法）
> **使用方式一致**的框架，并在此过程中逐层理解其底层原理。
> 阅读顺序：本文件 → 每个阶段的 `docs/guide/NN-*.md` → `tests/` → `examples/`。

---

## 0. 总原则（Read First）

- **不依赖、不导入、不复制任何官方 LangChain 代码。**
  禁止出现在依赖里的包：`langchain`、`langchain-core`、`langchain-community`、
  `langchain-openai`、`langgraph`、`langsmith`。
  可以学习其**公开 API 用法**与设计思想，实现必须自己写。
- **第三方基础库允许使用**：`pydantic` v2、`httpx`、`numpy`、`pytest`、
  `pytest-asyncio`、`ruff`、`mypy`（这些不是 LangChain）。
- **一步一步来**：每个阶段必须"能运行 + 有测试 + 有讲解"，上一阶段验收通过
  才进入下一阶段。拒绝把 demo 拼起来充数。
- **测试驱动（TDD）**：先写失败测试 → 最小实现 → 通过 → 提交。
- **中文文档 + 英文代码**：指南/讲解用中文；标识符、docstring、commit 用英文。
- **不主动 git commit**，除非用户明确要求。
- 工程规范细节见 `AGENTS.md`（命令、导入分组、类型、命名、错误处理）。

### 每个阶段的标准结构

1. **目标**：这一阶段结束时能做什么。
2. **背景 / 要解决的问题**：为什么官方要这样设计。
3. **关键概念与设计取舍**：抽象、协议、数据结构。
4. **任务清单**：可勾选的原子任务（含文件路径）。
5. **验收标准**：必须通过的测试与可运行示例。
6. **讲解文档**：`docs/guide/NN-<topic>.md` 源码级剖析。
7. **复盘练习**：每阶段末尾留 1–3 道思考/扩展题。

---

## 1. 阶段总览

| 阶段 | 主题 | 交付的核心抽象 | 里程碑 |
|---|---|---|---|
| P0 | 工程基座 | 打包、ruff/mypy/pytest、异常、版本 | `pip install -e ".[dev]"` 与 `pytest` 可跑 |
| P1 | 消息与 Prompt | Message 体系、PromptValue、Prompt 模板 | 能把模板渲染成 messages |
| P2 | Runnable 与 LCEL | `Runnable`、`\|` 组合、Sequence/Lambda/Parallel/Passthrough/Branch | `prompt \| model \| parser` 端到端 |
| P3 | 模型接口 | `BaseChatModel`/`BaseLLM`、OpenAI 兼容、Fake 模型、流式 | 可真实/离线调用与流式输出 |
| P4 | 输出解析 | OutputParser 家族 | 文本/JSON/Pydantic 解析可用 |
| P5 | 结构化链 | LLMChain、Sequential、Router、fallbacks | 多步链可组合运行 |
| P6 | Memory | ChatMessageHistory、摘要记忆、RunnableWithMessageHistory | 多轮对话状态可持久于内存 |
| P7 | 回调与事件 | CallbackHandler、CallbackManager、astream_events | 全链路可观测、可流式事件 |
| P8 | 检索 / RAG | Document、Splitter、Embeddings、VectorStore、Retriever、retrieval chain | 本地文档问答可运行 |
| P9 | 工具与 Agent | `@tool`、bind_tools、AgentExecutor/ReAct | 模型可调用工具并完成任务 |
| P10 | 收口与打包 | 公共 API、示例库、异步一致性、发布准备 | 框架达到"可用"标准 |

> 说明：本文件为总览与阶段级任务清单。**开始某一阶段时**，再把该阶段细化为
> 逐文件、逐命令的 bite-sized 计划（写到 `docs/plans/`），避免一次性铺开过度细节。

---

## 2. 目标目录结构

```
My-LangChain/
  AGENTS.md
  TASKS.md                 # 本文件
  README.md
  pyproject.toml
  my_langchain/
    __init__.py
    core/                  # 异常、基础类型、Runnable 基座、config
    messages/              # 消息体系
    prompts/               # Prompt 模板与 PromptValue
    language_models/       # 模型抽象 + providers
    output_parsers/
    chains/
    memory/
    callbacks/
    documents/             # Document、loader、splitter
    embeddings/
    vectorstores/
    retriever/
    tools/
    agents/
  tests/                   # 与源码路径镜像，test_*.py
  examples/                # 每个阶段可运行脚本
  docs/
    guide/                 # 每阶段中文讲解
    plans/                 # 阶段详细计划
```

导入路径约定：单一包 `my_langchain`，子模块镜像 `langchain_core` 的内部结构，
例如 `from my_langchain.prompts import ChatPromptTemplate`。

---

## 3. 分阶段任务清单

### P0 工程基座

**目标**：建立可安装、可测试、可 lint 的 Python 包骨架。

- [x] 创建 `pyproject.toml`（PEP 621）
  - `[project]` name=`my-langchain`，`requires-python = ">=3.13"`
  - 运行时依赖：`pydantic>=2`、`httpx`
  - `[project.optional-dependencies].dev`：`pytest`、`pytest-asyncio`、`ruff`、`mypy`
  - `[tool.ruff]` line-length=100；`[tool.ruff.lint]` 选择 `E,F,I,UP,B,SIM,D`
  - `[tool.mypy]` `python_version="3.13"`、`strict` 相关项
  - `[tool.pytest.ini_options]` `testpaths=["tests"]`、`asyncio_mode="auto"`
  - `[tool.pytest.ini_options].markers` 注册 `integration`
- [x] 创建 `my_langchain/__init__.py`，导出 `__version__`
- [x] 创建 `my_langchain/core/__init__.py`
- [x] 创建 `my_langchain/core/exceptions.py`
  - `MyLangChainError`（基类），子类：`ProviderError`、`ParsingError`、
    `ConfigurationError`、`InputValidationError`
- [x] `tests/__init__.py` 与 `tests/core/test_exceptions.py`
- [x] 在 `README.md` 补充安装与运行说明（中文）

**验收**
- `pip install -e ".[dev]"` 成功
- `ruff check .`、`ruff format --check .`、`mypy my_langchain` 干净
- `pytest -q` 通过（至少异常层级测试）

---

### P1 消息与 Prompt

**目标**：能定义消息、组装聊天模板并渲染为消息列表。

**关键概念**：Message 不可变数据结构、`content` 与 `additional_kwargs`、
消息类型标识、chunk 合并、`PromptValue` 的双态（string / messages）。

- [x] `my_langchain/messages/__init__.py`
- [x] `my_langchain/messages/base.py`
  - `BaseMessage`（`content`、`additional_kwargs`、`response_metadata`、`id`、`type`）
  - `HumanMessage`、`AIMessage`、`SystemMessage`、`FunctionMessage`、`ToolMessage`
  - `ChatMessage(role=...)`
  - `message_to_dict()` / `messages_from_dict()`
- [x] `my_langchain/messages/chunk.py`
  - `BaseMessageChunk`、`AIMessageChunk`，实现 `__add__`（chunk 级合并）
- [x] `my_langchain/prompts/prompt_values.py`
  - `PromptValue`（`to_string`/`to_messages`）、`StringPromptValue`、`ChatPromptValue`
- [x] `my_langchain/prompts/prompt_template.py`
  - `PromptTemplate`（`from_template`、`format`、输入变量校验）
- [x] `my_langchain/prompts/chat.py`
  - `ChatPromptTemplate`、`MessagesPlaceholder`、`from_messages`、`format_messages`
  - `partial()` 支持
- [x] `my_langchain/prompts/few_shot.py`
  - `FewShotPromptTemplate`、`FewShotChatMessagePromptTemplate`
- [x] `tests/messages/test_base.py`、`tests/prompts/test_chat.py` 等

**验收**
- 用 `ChatPromptTemplate` 把 `{topic}`/`{history}` 渲染为正确的消息列表
- `AIMessageChunk.__add__` 合并结果符合预期
- 缺变量时抛出清晰的 `InputValidationError`

---

### P2 Runnable 与 LCEL 运行时（框架心脏）

**目标**：实现统一的 `Runnable` 协议与 `|` 组合，端到端跑通
`prompt | model | parser`。

**关键概念**：统一接口、同步/异步/流式/批量四态、`RunnableConfig` 透传、
惰性组合（组合即 Runnable）、输入输出类型推断。

- [ ] `my_langchain/core/runnables/__init__.py`
- [ ] `my_langchain/core/runnables/config.py`
  - `RunnableConfig`（TypedDict）：`tags`、`metadata`、`callbacks`、`run_name`、
    `max_concurrency`、`recursion_limit`、`configurable`
  - `ensure_config()`、`merge_configs()`
- [ ] `my_langchain/core/runnables/base.py`
  - `Runnable` 抽象：`invoke`/`batch`/`stream`/`ainvoke`/`abatch`/`astream`/
    `astream_log`、`with_config`、`with_types`、`with_retry`、`with_fallbacks`、
    `bind`、`__or__`/`pipe`、`assign`、`pick`、`map`
  - 默认实现基于最小原语，子类只需实现 `invoke` 与 `stream`（async 同理）
- [ ] `my_langchain/core/runnables/sequence.py`：`RunnableSequence`
- [ ] `my_langchain/core/runnables/lambda.py`：`RunnableLambda`
- [ ] `my_langchain/core/runnables/parallel.py`：`RunnableParallel`、`RunnableMap`
- [ ] `my_langchain/core/runnables/passthrough.py`：`RunnablePassthrough`、`assign`
- [ ] `my_langchain/core/runnables/branch.py`：`RunnableBranch`
- [ ] `my_langchain/core/runnables/each.py`：`RunnableEach`
- [ ] `my_langchain/core/runnables/retry.py`：`RunnableRetry`
- [ ] `my_langchain/core/runnables/fallbacks.py`：`RunnableWithFallbacks`
- [ ] `tests/core/runnables/test_sequence.py` 等

**验收**
- `chain = prompt | fake_model | StrOutputParser()` 的
  `invoke`/`batch`/`stream`/`ainvoke`/`astream` 全部行为一致
- `RunnableParallel` 并发执行且 `max_concurrency` 生效（测试可断言）
- 组合后的输入输出类型标注可通过 `mypy`

---

### P3 模型接口

**目标**：定义语言模型抽象，实现 OpenAI 兼容 provider 与离线 Fake 模型。

- [ ] `my_langchain/language_models/base.py`：`BaseLanguageModel`
- [ ] `my_langchain/language_models/chat_models.py`
  - `BaseChatModel`（`invoke`/`stream`/`generate`/`bind_tools`/`with_structured_output`）
  - `SimpleChatModel`
- [ ] `my_langchain/language_models/llms.py`：`BaseLLM`、`LLM`
- [ ] `my_langchain/language_models/results.py`：`LLMResult`、`ChatResult`、
  `Generation`、`ChatGeneration`、`ChatGenerationChunk`、token 用量
- [ ] `my_langchain/language_models/fake.py`
  - `FakeListChatModel`、`FakeListLLM`（确定性、供测试与示例）
- [ ] `my_langchain/language_models/providers/openai.py`
  - `ChatOpenAI` / `OpenAI`（基于 `httpx`，`OPENAI_API_KEY` 环境变量）
  - 支持 `stream`、`ainvoke`、`astream`、`temperature`、`base_url`
- [ ] `tests/language_models/test_fake.py`（离线）、
  `tests/language_models/test_openai.py`（标 `@pytest.mark.integration`）

**验收**
- 无 API Key 时，`FakeListChatModel` 可完成 §P2 端到端链
- 有 Key 时，`ChatOpenAI` 的真实调用与流式输出可运行

---

### P4 输出解析

**目标**：把模型输出解析为字符串、JSON、Pydantic 对象。

- [ ] `my_langchain/output_parsers/base.py`：`BaseOutputParser`、
  `BaseTransformOutputParser`（支持流式解析）
- [ ] `StrOutputParser`、`CommaSeparatedListOutputParser`
- [ ] `JsonOutputParser`（可含 schema）、`PydanticOutputParser`
- [ ] `OutputFixingParser`、`RetryOutputParser`
- [ ] 与 LCEL 互操作：`model | parser`、`model.with_structured_output(Schema)`
- [ ] `tests/output_parsers/*`

**验收**
- 非法 JSON 触发 `ParsingError`；`OutputFixingParser` 能用模型自修复

---

### P5 结构化链

- [ ] `my_langchain/chains/base.py`：`Chain` 基类与 `LLMChain`
- [ ] `SequentialChain`、`SimpleSequentialChain`
- [ ] `RouterChain`（基于条件选择分支）
- [ ] `my_langchain/chains/fallbacks.py`：带降级的链
- [ ] `tests/chains/*`

**验收**：多步链可组合运行，失败时按 fallback 顺序降级。

---

### P6 Memory 与对话状态

- [ ] `my_langchain/memory/chat_history.py`
  - `BaseChatMessageHistory`、`InMemoryChatMessageHistory`
  - `messages`、`add_user_message`、`add_ai_message`、`clear`
- [ ] `my_langchain/memory/buffer.py`：`ConversationBufferMemory`
- [ ] `my_langchain/memory/summary.py`：`ConversationSummaryMemory`
- [ ] `my_langchain/core/runnables/history.py`：`RunnableWithMessageHistory`
  （`get_session_history(session_id)`）
- [ ] `tests/memory/*`

**验收**：同一 `session_id` 多轮调用，模型可看到历史并正确裁剪/摘要。

---

### P7 回调、流式与事件

- [ ] `my_langchain/callbacks/base.py`：`BaseCallbackHandler`（含各 `on_*` 钩子）
- [ ] `my_langchain/callbacks/manager.py`：`CallbackManager`、
  `CallbackManagerForLLMRun`/`ForChainRun`
- [ ] `my_langchain/callbacks/stdout.py`：`StdOutCallbackHandler`
- [ ] `Runnable.astream_events`：统一事件流（chain/llm/parser/tool 事件）
- [ ] token 级流式回调 `on_llm_new_token`
- [ ] `tests/callbacks/*`

**验收**：一次链式调用能采集中间事件，并按顺序流式输出 token。

---

### P8 检索 / RAG

- [ ] `my_langchain/documents/base.py`：`Document`、`BaseLoader`、`TextLoader`
- [ ] `my_langchain/documents/splitters.py`：`CharacterTextSplitter`、
  `RecursiveCharacterTextSplitter`（含 overlap）
- [ ] `my_langchain/embeddings/base.py` + `fake.py` + `openai.py`：`Embeddings`
- [ ] `my_langchain/vectorstores/base.py`：`VectorStore`、
  `InMemoryVectorStore`（余弦相似度检索）
- [ ] `my_langchain/retriever/base.py`：`Retriever`、`VectorStoreRetriever`
- [ ] `my_langchain/chains/retrieval.py`：
  `create_stuff_documents_chain`、`create_retrieval_chain`
- [ ] `tests/vectorstores/*`、`tests/chains/test_retrieval.py`

**验收**：用本地 txt 构建索引，完成"检索 + 生成"问答（FakeEmbeddings + Fake 模型离线可跑）。

---

### P9 工具与 Agent

- [ ] `my_langchain/tools/base.py`：`BaseTool`、`Tool`、`StructuredTool`
- [ ] `my_langchain/tools/decorator.py`：`@tool`（从类型与 docstring 推断 schema）
- [ ] `my_langchain/tools/conversion.py`：`convert_to_openai_tool`
- [ ] `BaseChatModel.bind_tools`
- [ ] `my_langchain/agents/`：ReAct / structured-chat agent、
  `AgentExecutor`（循环：模型 → 工具 → 观察 → 再模型）
- [ ] `tests/tools/*`、`tests/agents/*`

**验收**：模型（Fake 固定动作）能调用工具、读到工具结果并给出最终答案。

---

### P10 收口与打包

- [ ] 公共 API 导出：各子包 `__init__.py` 精心维护 `__all__`，顶层 `my_langchain`
      聚合常用入口
- [ ] `examples/` 覆盖每个阶段的最小可运行脚本
- [ ] 异步一致性测试（sync/async 行为对齐）
- [ ] `@pytest.mark.integration` 离线/在线用例分离
- [ ] 文档：README 使用示例 + `docs/guide/` 全阶段讲解
- [ ] 版本与 CHANGELOG，发布前 `ruff`/`mypy`/`pytest` 全绿

**验收**：新用户按 README 能在 5 分钟内跑通 `prompt | model | parser` 与本地 RAG 示例。

---

## 4. 里程碑验收总表

- [x] **M1**：`pip install -e ".[dev]"` 后 `ruff`/`mypy`/`pytest` 全绿（P0）
- [x] **M2**：模板 → 消息渲染正确（P1）
- [ ] **M3**：`prompt | model | parser` 四态（sync/async/stream/batch）端到端（P2–P4）
- [ ] **M4**：多步链 + Memory 多轮对话（P5–P6）
- [ ] **M5**：全链路回调/事件流可观测（P7）
- [ ] **M6**：本地 RAG 问答可运行（P8）
- [ ] **M7**：工具调用 Agent 可完成任务（P9）
- [ ] **M8**：框架"可用"并具备完整示例与文档（P10）

---

## 5. 协作方式

- 每完成一个原子任务即勾选本文件中的复选框。
- 每阶段开始前，把该阶段展开为 `docs/plans/YYYY-MM-DD-P<NN>-<topic>.md`
  的逐文件/逐命令计划，再动手。
- 遇到设计分歧（抽象接口、是否引入依赖），先讨论再实现，保持"学到底层"的目标。
