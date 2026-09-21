# My-LangChain

从零开始构建的 LangChain 框架（教学用途）。

本项目用纯 Python 从零实现一个与主流 LangChain（`langchain-core` 风格）**使用方式一致**
的框架，并在实现过程中逐层讲解其底层原理。完整路线图见 [`TASKS.md`](TASKS.md)，
工程约定见 [`AGENTS.md`](AGENTS.md)。

## 环境要求

- Python >= 3.13

## 安装

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows
pip install -e ".[dev]"
```

## 开发命令

```bash
ruff check .          # lint
ruff format .         # format
mypy                  # 类型检查
pytest                # 运行全部测试
pytest -q             # 安静模式
pytest tests/core/test_exceptions.py::test_base_error_subclasses_exception
```

## 当前进度

- [x] **P0 工程基座**：打包、工具链、异常体系
- [x] **P1 消息与 Prompt**：Message/Chunk、PromptValue、PromptTemplate/ChatPromptTemplate/FewShot
- [ ] P2 Runnable 与 LCEL 运行时
- [ ] P3 模型接口
- [ ] P4 输出解析
- [ ] P5 结构化链
- [ ] P6 Memory
- [ ] P7 回调与事件
- [ ] P8 检索 / RAG
- [ ] P9 工具与 Agent
- [ ] P10 收口与打包

## 许可证

[Apache-2.0](LICENSE)
