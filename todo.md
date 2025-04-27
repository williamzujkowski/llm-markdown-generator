todo.md

## 3\. Phase 1: Generator Enhancement (LangChain Integration)

**Objective:** Refactor `llm-markdown-generator` using LangChain/LangSmith for improved reliability, structure, observability, and frontmatter extraction accuracy. Adhere strictly to `CLAUDE.md`.

*(This section follows the detailed plan previously generated, referencing the Python-specific standards in `CLAUDE.md`)*

  * **Sub-Phase 1.1: Setup, Configuration, and Basic Tracing**
      * [ ] **Install Dependencies:** Add `langchain`, LLM provider (`langchain-openai`, etc.), `langsmith`, `pydantic`, `python-dotenv`, optional `langgraph` to `requirements.txt`/`pyproject.toml`. Install using `pip install -r requirements.txt` within `venv`. (Ref: CLAUDE.md Sec 1)
      * [ ] **Configure LangSmith:** Set up `.env` with LangSmith/LLM API keys and project details. Load variables using `dotenv`. (Ref: CLAUDE.md Sec 6.6)
      * [ ] **Implement Basic Trace Test:** Refactor a single LLM call using LangChain integration (e.g., `ChatOpenAI`) and verify trace appears in LangSmith UI.
  * **Sub-Phase 1.2: Refactor Core Logic with LangChain Components**
      * [ ] **Identify Core Tasks:** Document the logical steps of the current generation process.
      * [ ] **Create LangChain Prompt Templates:** Externalize prompts using `PromptTemplate`/`ChatPromptTemplate` (store in `.llmconfig/prompt-templates/` or `prompts.py`). (Ref: CLAUDE.md Sec 6.2)
      * [ ] **Define Pydantic Output Schemas:** Create Pydantic models defining the **exact target frontmatter structure** (based on sample analysis) and any other structured LLM outputs. (Ref: CLAUDE.md Sec 2 - Type Annotations)
      * [ ] **Implement Output Parsers:** Use LangChain parsers (e.g., `PydanticOutputParser`) linked to Pydantic schemas for reliable structured data extraction (especially frontmatter). Include format instructions in relevant prompts.
      * [ ] **Refactor LLM Calls:** Replace direct API calls with LangChain LLM/ChatModel integrations.
      * [ ] **Build Core LCEL Chains:** Connect prompts, LLMs, parsers using LCEL (`|`) for distinct generation tasks. Ensure code adheres to PEP 8, Black formatting, isort, Google-style docstrings. (Ref: CLAUDE.md Sec 2, 3.1, 3.2)
      * [ ] **Orchestrate Chains:** Combine LCEL chains to replicate the overall workflow, passing context.
      * [ ] **Unit Testing:** Write `pytest` unit tests for new/refactored chains, prompts, parsers using mocking. Aim for \>85% coverage. (Ref: CLAUDE.md Sec 4.1, 4.2.1, 4.4.3)
  * **Sub-Phase 1.3: Implement RAG (Optional Enhancement)**
      * [ ] **Identify Context Sources & Setup Vector Store:** Choose store (e.g., ChromaDB), embedding model. Implement indexing script for relevant documents.
      * [ ] **Integrate Retriever into Chains:** Modify LCEL chains to fetch and include retrieved context in prompts using `RunnableParallel`, etc.
  * **Sub-Phase 1.4: Implement LangGraph (Conditional Enhancement)**
      * [ ] **Analyze Complexity:** Proceed only if workflow needs loops, conditions, agents, or human-in-the-loop.
      * [ ] **Model State, Nodes, Edges:** Define graph structure using `StateGraph`.
      * [ ] **Compile and Run Graph:** Replace chain orchestration with graph execution.
  * **Sub-Phase 1.5: Implement LangSmith Evaluation**
      * [ ] **Create Evaluation Datasets:** Create datasets in LangSmith with input examples and reference outputs (especially frontmatter).
      * [ ] **Configure Evaluators:** Define/select LangSmith evaluators (e.g., JSON comparison, criteria, custom functions) for quality checks.
      * [ ] **Implement Evaluation Script:** Create `evaluate.py` using LangSmith SDK to run evaluations against datasets.
      * [ ] **Analyze Results:** Use evaluation feedback to iterate and improve the generator.
  * **Sub-Phase 1.6: CI/CD Evaluation Check (Optional)**
      * [ ] **Integrate Evaluation Run:** Add step to GitHub Actions (or other CI/CD) to run `evaluate.py`.
      * [ ] **Check Results:** Potentially fail build if quality metrics drop below thresholds.