# Financial Agent - Edge Cases Analysis 📋

**Document Version**: 1.0  
**Last Updated**: December 24, 2025  
**Project**: Vietnamese Stock Market Investment Assistant (Financial Agent)

---

## 📖 Table of Contents

1. [Use Cases](#use-cases)
2. [Architecture Overview](#architecture-overview)
3. [Workflow Pipeline](#workflow-pipeline)
4. [Critical Edge Cases](#critical-edge-cases)
5. [Data Flow Edge Cases](#data-flow-edge-cases)
6. [File Handling Edge Cases](#file-handling-edge-cases)
7. [RAG & Retrieval Edge Cases](#rag--retrieval-edge-cases)
8. [Tool Execution Edge Cases](#tool-execution-edge-cases)
9. [State Management Edge Cases](#state-management-edge-cases)
10. [Error Handling Edge Cases](#error-handling-edge-cases)
11. [Performance & Scalability Edge Cases](#performance--scalability-edge-cases)
12. [Recommendation Matrix](#recommendation-matrix)

---

## Use Cases

### 👤 **Primary Users & Their Goals**

#### 1. **Individual Retail Investor** 📈
**Profile**: Individual traders, small portfolio holders
- **Goals**: 
  - Get real-time stock prices and technical analysis
  - Understand financial ratios and company fundamentals
  - Make quick trading decisions
  - Track historical price trends

**Key Interactions**:
```
1. Opens Frontend
2. Types: "What's TCB's current price and 20-day moving average?"
3. System:
   ├─ Extracts ticker: "TCB"
   ├─ Calls tools: get_current_price(), calculate_sma()
   ├─ Returns: Price $23.50, SMA-20 $23.25
   └─ Agent synthesizes: "TCB is trading above its 20-day average..."
4. User makes investment decision
```

**Success Criteria**:
- ✅ Response within 3 seconds
- ✅ Accurate price data (< 5 min old)
- ✅ Correct technical calculations
- ✅ Clear explanations

**Common Queries**:
- "Which banks have highest P/E ratio?"
- "Show me VCB's historical revenue"
- "Compare TCB vs VCB dividend yields"
- "Is FPT overvalued based on P/E?"

---

#### 2. **Financial Analyst** 📊
**Profile**: Professional analysts, fund managers, financial advisors
- **Goals**:
  - Analyze large datasets (quarterly reports, 5-year financials)
  - Compare multiple companies systematically
  - Extract insights from unstructured documents
  - Generate investment reports

**Key Interactions**:
```
1. Uploads Q3 2024 Financial Reports (5 PDF files)
2. Types: "Summarize revenue growth trends across all companies"
3. System:
   ├─ [EXTRACT_DATA] Processes all PDFs with OCR
   ├─ [INGEST_FILE] Stores chunks in personal RAG
   ├─ [AGENT] Searches RAG for revenue data
   ├─ Aggregates: "FPT +15%, VCB +8%, TCB +5%..."
   └─ Generates markdown report with tables
4. Analyst downloads report for client presentation
```

**Success Criteria**:
- ✅ Process 50MB PDF in < 1 minute
- ✅ Maintain document privacy (no mix with other users)
- ✅ Accurate data extraction (> 95% accuracy)
- ✅ Structured output (markdown tables, charts)

**Common Queries**:
- "What's the average profit margin across uploaded companies?"
- "Which company shows strongest growth in assets?"
- "Extract debt-to-equity ratios from all reports"
- "Create comparison table of top 5 metrics"

---

#### 3. **Risk Manager** 🛡️
**Profile**: Bank risk officers, fund risk managers
- **Goals**:
  - Monitor market volatility indicators (RSI, MACD)
  - Identify risk trends and anomalies
  - Set alerts for threshold breaches
  - Generate compliance reports

**Key Interactions**:
```
1. Asks: "Show RSI for top 20 stocks, flag any > 70 (overbought)"
2. System:
   ├─ Iterates through ticker list
   ├─ Calculates RSI for each
   ├─ Filters: RSI > 70
   ├─ Returns: 5 stocks above threshold
   └─ Risk manager sets watch alerts
3. Monitors portfolio risk exposure
```

**Success Criteria**:
- ✅ Batch analysis of 20+ stocks < 5 seconds
- ✅ Accurate RSI calculations matching Bloomberg
- ✅ Real-time data (not > 5 min old)
- ✅ Consistent methodology across all tickers

**Common Queries**:
- "Which stocks are in overbought territory (RSI > 70)?"
- "Calculate correlation between TCB and VCB prices"
- "Show volatility metrics for last 90 days"
- "Alert if any stock drops > 10% from 50-day MA"

---

#### 4. **Student/Educator** 🎓
**Profile**: Finance students, investment course instructors
- **Goals**:
  - Learn stock market concepts
  - Understand technical indicators
  - Practice analysis on real data
  - Create educational materials

**Key Interactions**:
```
1. Student asks: "Explain the difference between SMA-20 and SMA-50"
2. System:
   ├─ Provides definition and purpose
   ├─ Calculates both for example stock (e.g., FPT)
   ├─ Shows chart/data comparison
   ├─ Explains trend signal: "Bullish if SMA-20 > SMA-50"
3. Student learns by example
```

**Success Criteria**:
- ✅ Clear, educational explanations
- ✅ Visual data representation
- ✅ Real-world examples with actual stock data
- ✅ No jargon without explanation

**Common Queries**:
- "What is P/E ratio and how to interpret it?"
- "Show me how to calculate Fibonacci retracement"
- "Explain RSI overbought/oversold conditions"
- "Compare growth stock vs value stock characteristics"

---

### 🎯 **Use Case Workflows**

#### **UC1: Quick Price Check** ⚡
```
Input: User types "TCB price"
├─ Classification: Financial query
├─ Tool Selection: get_current_price()
├─ Execution: VnStock API call
├─ Response: "TCB is trading at $23.50"
Duration: < 1 second
Complexity: Simple
Tool Chain: 1 tool
```

---

#### **UC2: Technical Analysis Report** 📉
```
Input: User types "Analyze FPT stock technical setup"
├─ Classification: Technical analysis
├─ Tool Selection: [
│    - get_historical_data(FPT, last 3 months)
│    - calculate_sma(FPT, windows=[20, 50, 200])
│    - calculate_rsi(FPT, period=14)
│  ]
├─ Execution: 3 sequential tool calls
├─ Synthesis: LLM merges results → analysis
├─ Response: "FPT shows bullish setup: price above SMA-20..."
Duration: 3-5 seconds
Complexity: Medium
Tool Chain: 3 tools
```

---

#### **UC3: Multi-Document Financial Analysis** 📑
```
Input: User uploads 10 company Q3 reports (PDF)
       Query: "Which company has best profitability?"
       
Flow:
├─ [EXTRACT_DATA] 
│  ├─ Process 10 PDFs
│  ├─ OCR scanned pages
│  ├─ Extract text chunks
│  └─ Create embeddings
│
├─ [INGEST_FILE]
│  ├─ Store chunks in personal RAG
│  ├─ Index by company/section
│  └─ Record file metadata
│
├─ [AGENT]
│  ├─ Search RAG: "profitability metrics"
│  ├─ Retrieve net profit margins
│  ├─ LLM extracts and compares
│  └─ Ranks companies by profitability
│
└─ Response: "Company A: 25% margin, Company B: 22%, ..."

Duration: 60-120 seconds (mostly file processing)
Complexity: High
Tool Chain: RAG search + LLM synthesis
File Types: PDF (with OCR)
```

---

#### **UC4: Comparative Analysis** 🔄
```
Input: User asks "TCB vs VCB: which is cheaper by P/E?"

Flow:
├─ Extract tickers: TCB, VCB
├─ Tool calls (parallel):
│  ├─ get_company_info(TCB) → earnings
│  ├─ get_current_price(TCB) → price
│  ├─ get_company_info(VCB) → earnings
│  └─ get_current_price(VCB) → price
├─ LLM calculates P/E for both
├─ Compares and ranks
└─ Response: "VCB cheaper: P/E 8.5 vs TCB 10.2"

Duration: 3-5 seconds
Complexity: Medium
Tool Chain: 4 parallel tools
```

---

#### **UC5: Batch Analysis with Filtering** 📊
```
Input: "Show all stocks with RSI > 70 (overbought) and price > 100K"

Flow:
├─ Tool: get_all_stocks() → [100+ tickers]
├─ For each ticker:
│  ├─ get_current_price()
│  ├─ calculate_rsi()
│  └─ Filter by conditions
├─ Aggregate results
└─ Response: Table of 8 stocks meeting criteria

Duration: 10-20 seconds
Complexity: High (batch processing)
Tool Chain: Sequential iteration + filtering
Data: 100+ stocks processed
```

---

#### **UC6: Long Conversation with Context** 💬
```
Turn 1:
├─ User: "What's FPT's revenue?"
├─ Agent: Calls tool, returns $X billion
└─ Store message in history

Turn 2:
├─ User: "How does that compare to last year?"
├─ Agent: Needs FPT context from Turn 1
├─ Calls get_historical_data(FPT)
├─ Compares current vs previous
└─ Response: "Up 15% YoY"

Turn 3:
├─ User: "Is that growth rate sustainable?"
├─ Agent: Uses FPT context from Turn 1 & 2
├─ Reads financial fundamentals
├─ Assesses sustainability
└─ Response: Analysis based on metrics

Duration: 15 seconds (3 turns)
Complexity: High (multi-turn context)
Tool Chain: 2 tools across conversation
Context: Maintained across 3 turns
```

---

### 📋 **Edge Cases Within Use Cases**

#### **UC2 Extended: What if RSI calculation fails?**
```
User: "Analyze FPT stock"

Normal flow:
├─ get_historical_data(FPT) → ✓ 1000 data points
├─ calculate_sma(FPT) → ✓ Returns moving averages
├─ calculate_rsi(FPT) → ✗ ERROR: "Need 14+ data points"

Edge case:
- Newly listed stock with only 5 trading days
- RSI requires 14+ periods
- Tool fails
- Agent sees error, continues with SMA only
- Answer: "FPT too new for RSI, but SMA shows..."
- User: Partially satisfied (missing one indicator)
```

---

#### **UC3 Extended: What if document is corrupted?**
```
User: Uploads 10 files: 9 PDFs + 1 corrupted ZIP

Processing:
├─ File 1-8: ✓ Extract successful
├─ File 9: ✗ ZIP file → Unsupported format → Skip
├─ File 10: ✓ Extract successful
├─ Result: 9/10 files ingested
└─ User: Unaware that 1 file was skipped!

Risk: Incomplete analysis without user notification
```

---

#### **UC5 Extended: What if batch processing hangs?**
```
User: "Show all stocks with RSI > 70"
      (Requests analysis of 100+ stocks)

Scenario:
├─ Process 10 stocks: 2 seconds ✓
├─ Process 20 stocks: 4 seconds ✓
├─ Process 50 stocks: 10 seconds ✓
├─ Process 80 stocks: 30 seconds ⚠️
├─ Process 100+ stocks: > 60 seconds ❌
├─ User gives up, closes browser
├─ Backend keeps processing
└─ Resources wasted

Risk: No UI feedback, user doesn't know if it's working
```

---

#### **UC6 Extended: What if context grows too large?**
```
Turn 1-10: Normal conversation, tokens OK
Turn 11-30: Still fine, ~50K tokens
Turn 31-50: Getting large, ~100K tokens
Turn 51: User asks new question
├─ System tries to send all 50 turns to LLM
├─ Total: 125K tokens > LLM limit (128K)
├─ Error: 413 Payload Too Large
├─ Chat breaks: User must start new conversation
└─ Conversation history lost

Risk: Data loss after long conversation
```

---

### ✅ **Success Metrics by Use Case**

| Use Case | Metric | Target | Current Risk |
|----------|--------|--------|--------------|
| **UC1: Quick Price** | Latency | < 1s | ✓ Likely met |
| | Accuracy | 100% | ✓ API accurate |
| **UC2: Tech Analysis** | Latency | < 5s | 🟡 Borderline |
| | Tool accuracy | 99%+ | 🟡 RSI edge cases |
| **UC3: Multi-Doc** | File handling | 0 corruption | 🔴 No validation |
| | Privacy | User-isolated | 🔴 Default user_id |
| **UC4: Comparison** | Tool chaining | Works seamlessly | 🟡 No error recovery |
| **UC5: Batch** | Timeout | < 30s | 🔴 No timeout set |
| | Feedback | Real-time progress | ❌ Silent processing |
| **UC6: Conversation** | Context limit | No crash | 🔴 > 128K crashes |
| | History preservation | All 50+ turns | 🟡 Token overflow |

---

### 🎨 **User Journey Maps**

#### **Analyst's Day**
```
9:00 AM
├─ Morning: Check market overview
│  └─ "Which sectors are up today?"
│
10:00 AM
├─ Client request: Analyze 3 companies
│  ├─ Uploads 3 Q3 reports
│  ├─ System ingests and indexes
│  └─ Analyst prepares comparison
│
11:00 AM
├─ Client call: Present analysis
│  ├─ Queries agent for supporting data
│  ├─ "Show revenue breakdown by segment"
│  └─ Gets instant RAG results
│
2:00 PM
├─ Deep dive: Technical analysis
│  ├─ "Calculate correlation matrix for top 10 stocks"
│  ├─ Batch processing
│  └─ Generates report
│
4:00 PM
├─ Risk review: Set alerts
│  ├─ "Alert if RSI > 70 for any watched stock"
│  └─ Monitoring begins
│
5:00 PM
└─ End of day: Export findings
   └─ "Create markdown report of today's analysis"
```

**Pain Points**:
- ❌ File upload can't be cancelled (UC3 edge case)
- ❌ Batch queries have no progress bar (UC5 edge case)
- ❌ Long conversations eventually crash (UC6 edge case)

---

### 🚨 **High-Risk Use Cases**

#### **Risk UC1: Regulatory Compliance Report**
```
Scenario: Risk manager must generate audit report
Input: "Extract all risk metrics from uploaded documents"

Requirements:
- ✅ Data accuracy: 100% (regulatory requirement)
- ✅ Data source tracking (where each number came from)
- ✅ Timestamp of data (when was it fetched)
- ✅ Audit trail (who accessed what)

Current System:
- ❌ No source tracking
- ❌ No timestamp on tool results
- ❌ No audit logging
- ❌ Default user_id = potential data leak

Risk Impact: **Regulatory failure, fines**
```

---

#### **Risk UC2: Critical Trade Execution**
```
Scenario: Trader relies on agent for price check before trade
Input: "What's FPT's current price? I'm about to execute 10M share trade"

Requirements:
- ✅ Data freshness: < 1 minute old
- ✅ Data accuracy: ±0% error acceptable
- ✅ Responsiveness: < 2 seconds

Current System:
- ⚠️  API could return cached data 5+ minutes old
- ⚠️  No freshness indicator in response
- ✓ Response time likely OK

Risk Impact: **Trader executes at wrong price, loses money**
```

---

#### **Risk UC3: Portfolio Manager Multi-User Collision**
```
Scenario: 3 portfolio managers using system simultaneously
Manager A: Uploads portfolio files
Manager B: Uploads portfolio files
Manager C: Uploads portfolio files

All use system default user_id="default"

Risk:
- Manager A's portfolio visible to B & C
- Wrong investment decisions
- Data breach, fiduciary violation

Current System:
- ❌ Defaults to "default" if user_id missing
- ❌ No validation of user_id
- ❌ No multi-tenancy isolation tests

Risk Impact: **Legal liability, license revocation**
```

---

## ✨ **Recommended Enhancements**

### For Each Use Case:

| Use Case | Enhancement | Effort | Impact |
|----------|-------------|--------|--------|
| **UC1** | Add "Data as of" timestamp | Low | High |
| **UC2** | Cache intermediate tool results | Medium | Medium |
| **UC3** | Batch file validation | Low | High |
| **UC4** | Parallel tool execution | Medium | High |
| **UC5** | Progress bar + cancellation | Medium | High |
| **UC6** | Token counting + auto-summarization | High | High |

---

## Architecture Overview

The Financial Agent is a multi-layered system with the following components:

```
┌─────────────────────────────────────────────────────────────────┐
│                    FastAPI Server (main.py)                     │
│              (Port 8000, CORS + Security Headers)               │
└──────────────────────────┬──────────────────────────────────────┘
                           │
        ┌──────────────────┼──────────────────┐
        │                  │                  │
        ▼                  ▼                  ▼
   ┌─────────┐      ┌──────────┐      ┌──────────┐
   │ Upload  │      │   Chat   │      │Admin API │
   │Endpoint │      │ Endpoint │      │Endpoint  │
   └────┬────┘      └────┬─────┘      └────┬─────┘
        │                │                  │
        └────────────────┼──────────────────┘
                         │
                         ▼
        ┌────────────────────────────────┐
        │   LangGraph Workflow Engine     │
        │  (4-Node: Extract→Ingest→     │
        │   Agent→Tools)                 │
        └────────────┬───────────────────┘
                     │
        ┌────────────┴─────────────────┐
        │                              │
        ▼                              ▼
   ┌─────────────┐          ┌──────────────────┐
   │File Process │          │ RAG Service      │
   │Pipeline     │          │ - Personal RAG   │
   │- PDF,Excel  │          │ - Global RAG     │
   │- OCR        │          │ - Semantic Search│
   └──────┬──────┘          │ - Keyword Search │
          │                 └──────────────────┘
          │
          ▼
   ┌──────────────┐
   │Qdrant Vector │
   │Database      │
   └──────────────┘
```

### Core Components

| Component | Purpose | Key Features |
|-----------|---------|--------------|
| **FastAPI App** | REST API Server | Auth, Rate Limiting, CORS |
| **LangGraph Workflow** | Orchestration Engine | 4-node pipeline with conditional routing |
| **FinancialAgent** | Tool & LLM Manager | 8+ financial tools, LLM integration |
| **File Pipeline** | File Processing | PDF, Excel, Image (OCR) support |
| **RAG Service** | Semantic Search | Multi-collection, user-isolated storage |
| **Qdrant Vector DB** | Vector Storage | Persistent embeddings, metadata filtering |

---

## Workflow Pipeline

### High-Level Flow

```
USER INPUT
    │
    ▼
┌─────────────────────────┐
│   EXTRACT_DATA NODE     │  ← File uploaded?
│  - Process files (PDF,  │
│    Excel, Images)       │
│  - Extract text chunks  │
│  - Create structured    │
│    data                 │
└────────┬────────────────┘
         │
         ▼
┌─────────────────────────┐
│   INGEST_FILE NODE      │  ← Chunks extracted?
│  - Embed chunks         │
│  - Store in personal    │
│    RAG (user isolated)  │
│  - Record file IDs      │
└────────┬────────────────┘
         │
         ▼
┌─────────────────────────┐
│   AGENT NODE (Pass 1)   │  ← Initial decision
│  - Read user query      │
│  - Prepare RAG context  │
│  - Invoke LLM with tools│
│  - Detect tool calls    │
└────┬───────────────────┤
     │                   │
     │ Tools Called?     │
     ├─ YES: ▼           │
     │   ┌─────────────┐ │
     │   │ TOOLS NODE  │ │
     │   │ - Execute   │ │
     │   │ - Get results
     │   └──────┬──────┘ │
     │          │        │
     │          └────┬───┘
     │               │
     │ NO: ──────────┘
     │
     └──▶ Continue to
         AGENT NODE
         (Pass 2)
         │
         ▼
    ┌─────────────────┐
    │ AGENT NODE      │
    │ (Final Synthesis)
    │ - Merge RAG +   │
    │   Tool Results  │
    │ - Generate      │
    │   Final Answer  │
    └────────┬────────┘
             │
             ▼
         RETURN
       FINAL ANSWER
```

### State Evolution

```
Initial State
├── user_prompt: "What is TCB's SMA-20?"
├── uploaded_files: []
├── conversation_history: []
├── user_id: "user123"
└── session_id: "session456"
    │
    ├─ [EXTRACT_DATA] ─▶ extracted_file_data: null
    │
    ├─ [INGEST_FILE] ──▶ ingested_file_ids: []
    │
    ├─ [AGENT PASS 1] ─▶ conversation_history: [
    │                      HumanMessage(...),
    │                      AIMessage(tool_calls=[...])
    │                    ]
    │
    ├─ [TOOLS] ────────▶ conversation_history: [...,
    │                      ToolMessage(result=...)
    │                    ]
    │
    └─ [AGENT PASS 2] ─▶ conversation_history: [...,
                           AIMessage(content="Final answer...")
                         ]
                         generated_answer: "Final answer..."
```

---

## Critical Edge Cases

### 🔴 1. NULL/EMPTY STATE PROPAGATION

**Scenario**: What happens when key fields are None or empty?

| Field | Edge Case | Current Behavior | Risk | Severity |
|-------|-----------|------------------|------|----------|
| `user_prompt` | Empty string `""` | Agent proceeds with empty query | LLM gets no context, generates generic response | 🟡 MEDIUM |
| `uploaded_files` | `None` vs `[]` | Extract node skips both | State inconsistency | 🟡 MEDIUM |
| `conversation_history` | Empty `[]` | Agent creates first message | First message has no context | 🟢 LOW |
| `best_search_results` | `None` | RAG context not applied | Answers miss document insights | 🔴 HIGH |
| `user_id` | "default" (fallback) | Multi-user collisions possible | Data isolation breach | 🔴 HIGH |
| `session_id` | "default" (fallback) | Multiple sessions mixed in RAG | Conversation leakage | 🔴 HIGH |

**Recommended Fixes**:
```python
# ❌ Current: allows None propagation
user_id = state.get("user_id", "default")

# ✅ Better: validate required fields
user_id = state.get("user_id")
if not user_id:
    raise ValueError("user_id is required for RAG isolation")

# ✅ Best: use proper defaults with validation
user_id = state.get("user_id") or uuid4()  # Generate unique ID if missing
session_id = state.get("session_id") or uuid4()
```

---

### 🔴 2. FILE SIZE & TYPE VALIDATION BYPASS

**Scenario**: What if someone uploads a 10GB file or unsupported format?

**Current State**:
- Document Service has 50MB limit ✅
- Supported formats: PDF, DOCX, TXT, PNG, JPG ✅
- BUT: No validation at API upload endpoint ❌

**Edge Cases**:
1. **Zero-byte file**: `len(file_data) == 0`
   - Pipeline extracts no chunks
   - Ingest node processes empty list
   - User thinks file was processed
   - **Risk**: Silent failure

2. **Corrupted PDF**: File has `.pdf` extension but binary corruption
   - pdfplumber fails silently or throws exception
   - Pipeline catches exception
   - Returns `"success": false` in extracted_data
   - **Risk**: Partial ingestion (some chunks succeed, some fail)

3. **Image with no text**: Picture of a graph (no OCR text)
   - Tesseract returns empty string
   - Creates 0-content chunks
   - Wastes database space
   - **Risk**: Bloated vector store, poor search results

4. **Mixed file types in batch upload**:
   - 5 PDFs + 1 Excel + 1 corrupted ZIP
   - Pipeline processes sequentially
   - Stops at ZIP (unsupported)
   - Earlier files already ingested
   - Later files never processed
   - **Risk**: Incomplete ingestion with no user notification

**Example Code**:
```python
# From: src/core/langgraph_workflow.py, node_extract_data
for file_info in uploaded_files:
    try:
        result = pipeline.process(file_path, file_type, file_name)
        extracted_data[file_name] = {
            "success": True,
            "content": result.get("text", ""),  # ⚠️  Could be empty!
            "chunks": result.get("chunks", [])   # ⚠️  Could be []!
        }
    except Exception as e:
        # Silently logs error and continues
        extracted_data[file_name] = {"success": False, "error": str(e)}
```

---

### 🔴 3. TOOL EXECUTION CASCADING FAILURES

**Scenario**: What if a tool crashes during execution?

**Current Implementation**:
```python
# From node_tools in langgraph_workflow.py
for tool in tool_calls:
    try:
        # Execute tool
        result = tool_executor.invoke(...)
        messages.append(ToolMessage(content=result))
    except Exception as e:
        # Returns error message
        error_message = AIMessage(content=f"Tool error: {str(e)}")
        messages.append(error_message)
```

**Edge Cases**:

| Scenario | Tool Example | Current Behavior | Issue |
|----------|--------------|------------------|-------|
| **Tool timeout** | get_historical_data (1000 days) | Hangs indefinitely | Request never completes |
| **API rate limit** | VnStock API limit exceeded | Exception caught, error message | Agent doesn't retry/backoff |
| **Invalid parameters** | SMA with window > data points | Tool throws ValueError | Error message in history |
| **Network error** | VnStock API unreachable | Socket timeout | Treated as execution error |
| **Partial results** | get_company_info returns None for some fields | Processing continues | Null fields propagate to answer |
| **Tool calls themselves** | Tool A calls Tool B (chain) | No mechanism for chaining | Must return result in single call |

**Risk Scenarios**:
1. **Infinite retry loop**: Agent sees error, decides to call tool again → same error → loop
2. **State corruption**: Tool partially modifies database, crashes, rolls back silently
3. **Resource exhaustion**: Multiple concurrent tool calls exhaust connection pool
4. **Answer quality**: Graceful error message looks like valid answer to user

---

### 🔴 4. RAG CONTEXT MISMATCH WITH TOOL RESULTS

**Scenario**: RAG returns one answer, tool returns different answer

**Example**:
- **RAG**: "TCB's latest revenue is $500M (from 2024 Q3 report)"
- **Tool**: "TCB's latest stock price is $25 (from VnStock API)"
- **Conflict**: Different data sources, different freshness
- **Agent must decide**: Which to trust?

**Current Behavior**:
```python
# From node_agent in langgraph_workflow.py
if rag_context:
    system_text += "\n📚 Tài liệu liên quan:\n"
    for i, doc in enumerate(rag_context[:5], 1):
        system_text += f"  {i}. {title} (score: {score:.1%})\n"

# Then later:
response = chain.invoke({"messages": messages})  # LLM merges both!
```

**Problems**:
1. **No conflict resolution**: LLM decides based on prompt, not data freshness
2. **Hallucination risk**: LLM might fabricate reconciliation
3. **User confusion**: Answer doesn't cite which source was primary
4. **Potential contradictions**: "Report says revenue is $X, but stock trades at $Y (contradiction!)"

---

### 🔴 5. CIRCULAR TOOL DEPENDENCIES

**Scenario**: Tool A needs output of Tool B, but Tool B needs output of Tool A

**Example Dependency Chain**:
```
User: "Calculate RSI for TCB over last 30 days"
     │
     ├─ Tool: calculate_rsi()
     │  └─ Needs: historical_prices
     │     │
     │     └─ Tool: get_historical_data()
     │        └─ Needs: ticker ("TCB")
     │           ✓ From user prompt
     │
     └─ Success: RSI calculated
```

**Problem Scenario** (Hypothetical):
```
User: "Calculate SMA for the stock mentioned in the document"
     │
     ├─ Tool: calculate_sma()
     │  └─ Needs: ticker
     │     │
     │     └─ Extract from doc (but which doc? multiple uploaded)
     │        │
     │        └─ Tool: search_documents()  
     │           └─ Needs: search_query
     │              │
     │              └─ Generated from user prompt...
     │                 ├─ "the stock" (ambiguous!)
     │                 ├─ Interpreted as FIRST ticker found
     │                 └─ If wrong ticker, SMA is wrong
     │
     └─ Silent failure: SMA calculated for wrong ticker!
```

**Current Implementation Has No**:
- ✗ Dependency resolution
- ✗ Multi-step tool chains
- ✗ Parameter validation between tools
- ✗ Circularity detection

---

## Data Flow Edge Cases

### 🟡 6. ENCODING & UNICODE ISSUES

**Scenario**: Vietnamese text with special characters

**Edge Cases**:

| Case | Input | Risk | Example |
|------|-------|------|---------|
| **Mixed encodings** | PDF with UTF-8 + ISO-8859-1 | Mojibake | "Tiếng Việt" → "TiÕ‰ng ViÕ‡t" |
| **Emoji in prompt** | "Công ty 📈 là gì?" | Tokenization breaks | LLM truncates after emoji |
| **RTL text** | Arabic/Hebrew in documents | Display issues | Search still works, UI breaks |
| **Tone marks** | "tài chính" vs "tài chính" (different marks) | Search misses | Similar meaning, exact match fails |
| **Control characters** | PDF with embedded null bytes | Parser crashes | `\x00` breaks string operations |

**Example Code**:
```python
# From document_service.py
def extract_text_from_image(self, image_path):
    # Uses Tesseract with default config
    text = pytesseract.image_to_string(image_path)  # ⚠️  No encoding specified
    
    # If image has mixed encodings:
    # - May succeed but produce garbage
    # - May fail silently
    # - May throw exception with partial results
    return text
```

**Recommended Fix**:
```python
# ✅ Explicit encoding handling
import unicodedata

def normalize_text(text: str) -> str:
    """Normalize Vietnamese text"""
    # NFD decomposition for consistent handling of tone marks
    text = unicodedata.normalize('NFD', text)
    # Remove null bytes that break vector operations
    text = text.replace('\x00', '')
    return text
```

---

### 🟡 7. LARGE CONTEXT WINDOW OVERFLOW

**Scenario**: Many documents + long conversation history = exceeds LLM token limit

**Example**:
```
Conversation History:
├── User: Query 1 (500 tokens)
├── Agent: Response 1 (2000 tokens)
├── User: Query 2 (500 tokens)
├── Agent: Response 2 (2000 tokens)
├── [... 50 turns later ...]
└── User: Query 50 (500 tokens)
    Total: ~125,000 tokens

+ RAG Documents (5 results × 2000 tokens each) = 10,000 tokens

+ System Prompt = 3,000 tokens

= 138,000 tokens > 128K Claude token limit ❌
```

**Current Behavior**:
```python
# From node_agent
prompt = ChatPromptTemplate.from_messages([
    ("system", system_text),  # 3K tokens
    MessagesPlaceholder(variable_name="messages"),  # ALL history!
])
response = chain.invoke({"messages": messages})  # Sends everything to LLM
```

**Problems**:
1. **No token counting**: Sends full history without checking
2. **No truncation**: Oldest messages never pruned
3. **No summarization**: Can't compress history
4. **Hard failure**: LLM returns 413 Payload Too Large
5. **User impact**: Chat becomes unusable after 50+ turns

---

### 🟡 8. RACE CONDITIONS IN CONCURRENT REQUESTS

**Scenario**: Two users upload files simultaneously

**Setup**:
```
User A (session_a): Uploads file_a.pdf
User B (session_b): Uploads file_b.pdf
            Both trigger workflow simultaneously
            Both use default user_id="default"  ← RACE CONDITION!
```

**Timeline**:
```
T0: User A → Upload file_a.pdf
    T0+100ms: User B → Upload file_b.pdf

T0+200ms: Workflow A starts → [EXTRACT_DATA] for file_a
T0+250ms: Workflow B starts → [EXTRACT_DATA] for file_b

T0+400ms: Workflow A → [INGEST_FILE] for file_a.pdf
          └─ rag_service.add_document(user_id="default", session_id="session_a", ...)

T0+420ms: Workflow B → [INGEST_FILE] for file_b.pdf
          └─ rag_service.add_document(user_id="default", session_id="session_b", ...)

T0+500ms: User A queries "What's in my document?"
          ├─ Searches personal RAG for user_id="default"
          ├─ Finds BOTH file_a AND file_b!  ← BUG!
          └─ Answer includes User B's data!

T0+520ms: User B queries "What's in my document?"
          ├─ Searches personal RAG for user_id="default"
          ├─ Finds BOTH file_a AND file_b!  ← BUG!
          └─ Answer includes User A's data!
```

**Why It Happens**:
```python
# From langgraph_workflow.py
user_id = state.get("user_id", "default")  # ← Fallback for both users!
session_id = state.get("session_id", "default")  # ← Same for both!

# At API level (app.py), user_id should come from JWT token
# BUT if extraction fails → defaults to "default" for BOTH
```

---

## File Handling Edge Cases

### 🟡 9. PDF SPECIAL CASES

**Scenario**: Different PDF structures require different handling

| PDF Type | How It's Created | Extraction Method | Edge Case |
|----------|-----------------|------------------|-----------|
| **Native PDF** | PDF creation software | Text layer | ✅ Works |
| **Scanned PDF** | Scan → PDF (image pages) | OCR (Tesseract) | ⚠️  Quality depends on scan |
| **Mixed PDF** | Some pages text, some scanned | Both methods combined | ❌ Inconsistent results |
| **Form PDF** | PDF with form fields | Text extraction | ❌ Misses data in fields |
| **PDF/A archival** | Long-term storage format | May have encryption | ❌ Extraction fails |

**Code Analysis**:
```python
# From document_service.py
def process_file(self, file_path, ...):
    file_ext = file_path.suffix.lower().lstrip('.')
    
    if file_ext == 'pdf':
        return self._process_pdf(file_path)
```

```python
def _process_pdf(self, file_path):
    try:
        with pdfplumber.open(file_path) as pdf:
            text = ""
            for page in pdf.pages:
                # Try to extract text from text layer
                page_text = page.extract_text()
                if not page_text:
                    # ⚠️  Falls back to OCR here
                    page_text = self._ocr_page(page)
                text += page_text
            return text
    except Exception as e:
        # Silently returns empty string or error
        logger.error(f"PDF extraction failed: {e}")
        return ""
```

**Problems**:
1. **No progress tracking**: 1000-page PDF processes silently, no feedback
2. **OCR silently degrades quality**: User doesn't know text came from OCR
3. **Performance cliff**: PDF with 1000 scanned pages takes 5+ minutes
4. **No cancellation**: Can't stop mid-extraction
5. **Memory leaks**: Large PDFs in memory, never garbage collected

---

### 🟡 10. EXCEL MULTIPLE SHEET HANDLING

**Scenario**: Excel file has 10 sheets, user uploads without specifying sheet

**Example Workbook**:
```
Sheet1: Financial Ratios (relevant)
Sheet2: Board Members (metadata)
Sheet3: Historical Prices (huge table, 10K rows)
Sheet4-10: Empty or deprecated
```

**Current Behavior**:
```python
# From excel_tools.py
def analyze_excel_to_markdown(file_path):
    # Likely reads all sheets or first sheet only
    df = pd.read_excel(file_path)
    return df.to_markdown()
```

**Problems**:
1. **Wrong sheet selected**: Reads first sheet, might be metadata, not data
2. **Combinatorial explosion**: 10K rows × 50 columns = massive text
3. **Chunking artifacts**: Cuts table mid-row, breaks structure
4. **User confusion**: "I uploaded this file but it's not finding my data"
5. **No sheet preview**: Can't guide user to correct sheet

---

## RAG & Retrieval Edge Cases

### 🔴 11. SEMANTIC SEARCH FAILURE MODES

**Scenario**: Embeddings don't capture domain semantics

| Query | Expected | What Actually Happens |
|-------|----------|----------------------|
| "tăng trưởng lợi nhuận" (growth in profit) | Financial reports with revenue increases | Returns generic business articles |
| "định giá P/E cao" (high P/E valuation) | Articles on valuation metrics | Returns articles mentioning "định giá" and "cao" (high) separately |
| "Ngân hàng nào tốt nhất?" (Which bank is best?) | Comparative analysis of banks | Returns individual bank pages in random order |
| "TCB so với VCB" (TCB vs VCB comparison) | Comparative articles | Returns TCB page + VCB page (no comparison) |

**Root Cause**:
```python
# From multi_collection_rag_service.py (hypothetically)
# Embeddings from: OpenAI, Gemini, or local model

# These models may not understand:
# - Vietnamese financial domain terminology
# - Implicit comparisons ("A so với B" = comparison)
# - Negations ("không tốt" = opposite meaning)
# - Irony ("tốt quá!" as sarcasm)
```

**Example Failure**:
```
Query: "Công ty nào có doanh thu cao nhất?"
       (Which company has highest revenue?)

Expected: Financial reports ranked by revenue

Actual: Returns documents with keywords:
        ✓ "công ty" (company)
        ✓ "cao" (high)
        ✗ But not necessarily highest revenue!
        ✗ Might include "Công ty cao su" (rubber company - mismatched)
```

---

### 🔴 12. KEYWORD SEARCH BRITTLENESS

**Scenario**: Exact keyword matching fails with variations

| Query | Document Content | Match? |
|-------|------------------|--------|
| "TCB" | "Techcombank" | ❌ No |
| "Techcombank" | "TCB (Techcombank)" | ⚠️  Partial |
| "giá cổ phiếu" (stock price) | "Giá cổ phiếu TCB" | ✓ Yes |
| "giá cổ phiếu" | "TCB's stock price" | ❌ No (English) |
| "lợi nhuận" (profit) | "laba" (Indonesian) | ❌ No (wrong language) |
| "P/E" | "P/E Ratio" | ⚠️  Maybe (depends on tokenizer) |

**Current Implementation**:
```python
# From multi_collection_rag_service.py (hypothetically)
def search_keyword(self, query: str, user_id: str):
    # Probably uses simple text search
    results = db.filter(
        filter={
            "user_id": user_id,
            "text": {"$contains": query}  # ← Exact substring match
        }
    )
    return results
```

**Problems**:
1. **No fuzzy matching**: Typos cause 0 results ("TCB" vs "TCb")
2. **No synonym support**: "stock" and "share" are different
3. **No lemmatization**: "invested", "investing", "investment" are different words
4. **Case sensitive**: "TCB" ≠ "tcb" in some databases
5. **Vietnamese-specific**: Tone marks matter ("tài" vs "tai")

---

### 🟡 13. VECTOR STORE STATE INCONSISTENCY

**Scenario**: Document added to Qdrant but metadata not saved in DB

**Timeline**:
```
T0: add_document() called
T0+100ms: Document embedded and added to Qdrant ✓
T0+200ms: Metadata saved to PostgreSQL...
T0+300ms: ❌ PostgreSQL connection drops!
          Metadata never saved
          
T0+500ms: Qdrant has vector but no metadata
          Can retrieve content but not verify source/date/user

T1 (later): 
  Vector search returns orphaned embedding
  Cannot verify user permissions (no metadata)
  Possible: Wrong user sees document from another user!
```

**Current Code**:
```python
# From file_ingestion_service.py or rag_service.py
def add_document(self, user_id, text, title, metadata):
    # Step 1: Embed and add to Qdrant
    embedding = embed_model.embed(text)
    qdrant_client.upsert(
        collection_name="documents",
        points=[Point(id=uuid4(), vector=embedding, payload=metadata)]
    )
    
    # Step 2: Save to database (⚠️  SEPARATE OPERATION)
    db_session.add(Document(
        user_id=user_id,
        title=title,
        content=text,
        metadata=json.dumps(metadata)
    ))
    db_session.commit()  # ← Can fail here!
```

**Problems**:
1. **No transaction**: Two separate operations, no atomic guarantee
2. **Orphaned vectors**: Qdrant has data, DB doesn't
3. **Privacy breach**: Search returns vectors user shouldn't see
4. **Inconsistent counts**: "5 documents in DB" but search finds 6
5. **No cleanup mechanism**: Orphaned vectors stay forever

---

## Tool Execution Edge Cases

### 🟡 14. FINANCIAL DATA STALENESS

**Scenario**: Tool returns outdated price data

**Example**:
```
Query: "What's TCB's current price?"
Time: Dec 24, 2025, 3:00 AM (before market open)

Tool: get_current_price()
├─ Calls VnStock API
├─ Returns: $23.50 (from yesterday's close)
└─ Agent response: "TCB is trading at $23.50"  ← Stale!

Reality: Market opens at 9:15 AM, price is now $23.80
User sees: Outdated price, makes bad trading decision
```

**Reasons**:
1. **API lag**: VnStock might cache data for 5-15 minutes
2. **Market hours**: Vietnam market 9:15-11:30 AM, 1:00-3:00 PM
3. **No timestamp** in tool response, user doesn't know if fresh
4. **No cache invalidation**: Cached result served all day

**Current Code**:
```python
# From vnstock_tools.py
def get_current_price(ticker):
    try:
        # VnStock API call - no freshness guarantee
        price = vnstock.stock.get_price(ticker)
        return {"price": price}
    except Exception:
        return {"error": "Price not available"}
```

**Problems**:
1. **No timestamp**: Response doesn't say when price was fetched
2. **No staleness check**: Accepts any data regardless of age
3. **Market hours unaware**: Returns yesterday's close at 3 AM
4. **No refresh on error**: Retries with same stale cache

---

### 🟡 15. MISSING OR INVALID TICKER HANDLING

**Scenario**: Tool receives invalid or delisted ticker

| Ticker | Status | Tool Behavior | Outcome |
|--------|--------|---------------|---------|
| "TCB" | Valid, active | Returns data | ✓ Works |
| "tcb" | Valid (lowercase) | May return 404 | ❌ Or normalizes to TCB |
| "TCB!" | Invalid (special char) | Returns error | ✓ Error message |
| "DEADCO" | Delisted 5 years ago | API returns 404 | ❌ Or empty history |
| "" | Empty string | Tool behavior unclear | ❌ Undefined |
| "TCB TCB" | Duplicate | API interprets as typo | ❌ Wrong interpretation |
| "A" | Too short, real ticker? | Ambiguous | ❌ Matches multiple |

**Code Issues**:
```python
# From vnstock_tools.py or technical_tools.py
def calculate_sma(ticker, window=20, days=100):
    # ⚠️  No validation of ticker format
    history = vnstock.stock.get_historical_data(
        ticker,  # Could be invalid!
        start_date=start_date,
        end_date=end_date
    )
    
    if not history:
        # What to do? Return error? Empty result?
        return {"error": "No data found"}
    
    # Calculate SMA
    return {"sma": sma_values}
```

**Problems**:
1. **No format validation**: "-TCB", "TCB.HNX", "TCB:VNM" treated as different
2. **No typo correction**: "TVC" vs "TCB" misses by 1 letter
3. **Silent failure**: Empty result looks like "no data available"
4. **User confusion**: "I asked for TCB but got no results"

---

## State Management Edge Cases

### 🔴 16. BIDIRECTIONAL STATE MUTATION

**Scenario**: Nodes modify state in unexpected ways, affecting other nodes

**Example Flow**:
```
Initial State:
├── best_search_results: [doc1, doc2, doc3]
└── conversation_history: [HumanMessage, AIMessage]

Node: Agent
├── Reads: best_search_results
├── Reads: conversation_history
├── ⚠️  MUTATES: conversation_history.append(new_message)
└── Returns: updated state

Node: Tools (next node)
├── Reads: conversation_history (has new_message!)
├── UNEXPECTED: Processes message it didn't create
└── Bug: Potential duplicate processing
```

**Code Risk**:
```python
# From langgraph_workflow.py
async def node_agent(self, state: WorkflowState) -> Dict[str, Any]:
    messages = state.get("conversation_history", [])
    
    # ⚠️  Direct mutation of list
    updated_messages = messages + [response]  # OK - creates new list
    # BUT if code later does:
    messages.append(response)  # ❌ Direct mutation of shared reference!
    
    return {"conversation_history": updated_messages}
```

**Why It's Dangerous**:
```
If messages is a reference to the original state dict's list:
- Modifying messages modifies the state directly
- Other nodes see the modified state
- Can't rollback if error occurs after mutation
- Hard to debug: "Where did this message come from?"
```

---

### 🟡 17. TYPE MISMATCHES IN STATE TRANSITIONS

**Scenario**: State expects Dict but receives List, or None when non-None expected

| Field | Type Spec | Possible Runtime Values | Issue |
|-------|-----------|------------------------|-------|
| `extracted_file_data` | `Optional[Dict]` | `None`, `{}`, `Dict[str, Dict]` | ✓ OK |
| `ingested_file_ids` | `List[str]` | `[]`, `["file1"]`, `None`! | ❌ None violates type |
| `best_search_results` | `List[Dict]` | `[]`, `[{...}]`, `"error message"`! | ❌ String instead of List |
| `conversation_history` | `List[Dict]` | `[]`, `[...messages...]`, `None` | ❌ None breaks iteration |
| `metadata` | `Dict` | `{}`, `{...}`, `None` | ⚠️  May be None |

**Example Code Bug**:
```python
# From node_ingest_file
ingested_file_ids = ingested_file_ids if isinstance(ingested_file_ids, list) else []

# Later, from node_agent
for file_id in state.get("ingested_file_ids"):  # ← Could be None!
    # Crashes here if None
    search_in_file(file_id)  # TypeError: 'NoneType' object is not iterable
```

---

## Error Handling Edge Cases

### 🔴 18. SILENT FAILURES IN TRY-EXCEPT BLOCKS

**Scenario**: Errors are caught but not properly handled

**Pattern Found Repeatedly**:
```python
# From langgraph_workflow.py, file_ingestion_service.py, etc.
try:
    result = risky_operation()
    return {"success": True, "data": result}
except Exception as e:
    logger.error(f"Operation failed: {e}")  # ← Logged but...
    return {"success": False, "error": str(e)}  # ← Graceful fallback
    # ❌ Continue processing as if success!
```

**Why It's Dangerous**:
```
User uploads file:
├─ [EXTRACT_DATA] catches exception, returns {"success": False, "error": "..."}
├─ [INGEST_FILE] checks `if not extracted_data` → skips ingestion ✓
├─ [AGENT] proceeds without extracted data ✓
└─ User gets answer without their document → No notification!

User never knows:
- File upload failed
- Why it failed
- What to do differently
```

**Better Pattern**:
```python
try:
    result = risky_operation()
    return result
except FileNotFoundError as e:
    # Specific handling for known errors
    logger.warning(f"File missing: {e}")
    return None  # Caller knows None = file missing
except ValueError as e:
    # Validation error - user's fault
    logger.warning(f"Invalid input: {e}")
    raise  # Re-raise for API to catch and return 400
except Exception as e:
    # Unexpected error - system's fault
    logger.error(f"Unexpected error: {e}", exc_info=True)
    raise  # Re-raise for API to catch and return 500
```

---

### 🟡 19. TIMEOUT & RESOURCE EXHAUSTION

**Scenario**: Long-running operations exhaust resources

| Operation | Timeout | Impact | Severity |
|-----------|---------|--------|----------|
| **PDF extraction (1000 pages)** | 5 min? | Extraction hangs | 🟡 MEDIUM |
| **OCR on 100 images** | 10 min? | Tesseract uses 100% CPU | 🔴 HIGH |
| **Vector embedding 50K chunks** | 30 min? | Database connection pool exhausted | 🔴 HIGH |
| **Semantic search (100K vectors)** | 10 sec? | Qdrant query timeout | 🟡 MEDIUM |
| **LLM inference (128K tokens)** | 2 min? | API timeout | 🔴 HIGH |

**Current State**:
- No timeouts configured ❌
- No resource limits ❌
- No async/queue system for heavy ops ❌
- No progress tracking ❌
- No cancellation mechanism ❌

**Example**:
```python
# From file_processing_pipeline.py
def process(self, file_path, file_type, file_name):
    # ⚠️  No timeout set
    if file_type == "pdf":
        return self._process_pdf(file_path)  # ← Could hang forever!

def _process_pdf(self, file_path):
    # ⚠️  No OCR timeout
    text = pytesseract.image_to_string(image)  # ← 10 min+ for complex images
```

---

### 🟡 20. EXCEPTION INFORMATION LOSS

**Scenario**: Error details lost in exception translation

**Example**:
```
VnStock API error:
├─ Original exception: "Connection timeout after 30s (service down)"
├─ Tool catches: except Exception as e
├─ Translates to: {"error": "Tool execution failed"}  ← Lost detail!
├─ Agent sees: {"error": "Tool execution failed"}  ← No retry logic!
├─ User sees: "I couldn't get that data"  ← No context!
└─ Root cause never fixed: Service still down 1 hour later

vs. Better:
├─ Tool catches specific: except requests.Timeout
├─ Translates to: {"error": "Service temporarily unavailable", "retry_after": 60}
├─ Agent sees: retry flag → decides to wait and retry
├─ User sees: "Getting fresh data..." → waits
└─ Data fetched successfully after wait
```

---

## Performance & Scalability Edge Cases

### 🟡 21. VECTOR STORE SCALING LIMITS

**Scenario**: Qdrant performance degrades with collection size

| Collection Size | Vector Search Latency | Issue |
|-----------------|----------------------|-------|
| 1K documents | 10ms | ✓ Fine |
| 100K documents | 100ms | ✓ Acceptable |
| 1M documents | 1s | 🟡 Noticeable |
| 10M documents | 10s | 🔴 Too slow |
| 100M documents | > 60s | 🔴 Timeout |

**Why It Happens**:
```
- Vector search requires distance calculation with every vector
- With 10M vectors, even simple operations slow down
- Filtering (by user_id) helps, but doesn't scale linearly
- Qdrant needs manual sharding/partitioning for multi-tenancy
```

**Current Risk**:
```
After 1 year of operation:
├── 1000 users × 100 documents each = 100K documents
├── Semantic search takes 500ms → User sees delay
├── Add more documents → 200ms search becomes 1s
├── Users complain → App seems broken
└── No built-in sharding → Can't distribute load
```

---

### 🟡 22. EMBEDDING MODEL CAPACITY

**Scenario**: Too many concurrent embedding operations

**Example**:
```
10 users upload 10 files simultaneously:
├── 100 files being processed
├── Each file → 50 chunks
├── 5000 chunks need embedding
├── Embedding API limit: 100 req/min
└── Queuing time: 5000 / 100 = 50 minutes!

Result:
├── User 1 waits 50 minutes for "upload complete"
├── UI shows spinning wheel
├── User refreshes browser (cancels upload)
├── Partial data in database
└── Orphaned vectors in Qdrant
```

**Current Implementation**:
```python
# From file_ingestion_service.py
for chunk in chunks:
    embedding = embed_model.embed(chunk)  # ← Sequential, no batching!
    qdrant_client.upsert(...)
    # Process one chunk per iteration = slow
```

---

## Recommendation Matrix

### 🔴 CRITICAL (Fix Immediately)

| #  | Issue | Impact | Effort | Priority |
|----|-------|--------|--------|----------|
| 1  | User isolation defaults to "default" | Data breach | Medium | P0 |
| 3  | Tool errors cause infinite retries | API hang | Medium | P0 |
| 4  | RAG vs Tool result conflicts unresolved | Wrong answers | High | P0 |
| 11 | Semantic search fails on domain terms | Poor search | Medium | P0 |
| 16 | State mutations affect other nodes | Race conditions | High | P0 |
| 18 | Silent failures in try-catch | Lost user data | Medium | P0 |

### 🟡 HIGH (Fix in Sprint)

| #  | Issue | Impact | Effort | Priority |
|----|-------|--------|--------|----------|
| 2  | No file validation | Data waste | Low | P1 |
| 6  | Encoding/Unicode issues | Search fails | Medium | P1 |
| 7  | Token limit overflow | Chat crash | High | P1 |
| 8  | Race conditions in concurrent requests | Data leak | High | P1 |
| 9  | PDF special cases | Poor extraction | Medium | P1 |
| 12 | Keyword search brittleness | No results | Medium | P1 |
| 13 | Vector store inconsistency | Data loss | High | P1 |
| 14 | Financial data staleness | Wrong decisions | Medium | P1 |
| 17 | Type mismatches in state | Crashes | Medium | P1 |
| 19 | Timeouts/resource exhaustion | API hang | High | P1 |
| 21 | Vector store scaling | Slowdown | High | P2 |

### 🟢 MEDIUM (Plan for Next Phase)

| #  | Issue | Impact | Effort | Priority |
|----|-------|--------|--------|----------|
| 5  | Circular tool dependencies | Wrong results | High | P2 |
| 10 | Excel multiple sheets | User confusion | Low | P2 |
| 15 | Invalid ticker handling | Wrong ticker | Low | P2 |
| 20 | Exception info loss | Poor debugging | Low | P2 |
| 22 | Embedding capacity | Slow uploads | High | P3 |

---

## Implementation Checklist

### For Each Edge Case Fix

- [ ] **Add input validation**
  ```python
  if not user_id:
      raise ValueError("user_id required")
  ```

- [ ] **Add type checking**
  ```python
  from typing import List
  assert isinstance(results, list), f"Expected list, got {type(results)}"
  ```

- [ ] **Add logging**
  ```python
  logger.warning(f"Edge case detected: {condition}")
  ```

- [ ] **Add unit test**
  ```python
  def test_edge_case_empty_input():
      result = function_under_test("")
      assert result == expected_behavior
  ```

- [ ] **Add integration test**
  ```python
  def test_end_to_end_with_edge_case():
      response = api_call(edge_case_input)
      assert response.status_code == expected_code
  ```

- [ ] **Update documentation**
  - Known limitations
  - Workarounds
  - Future improvements

---

## Summary

This Financial Agent has a sophisticated architecture but exhibits common edge case vulnerabilities:

**Most Critical**: User isolation, state mutations, and tool error handling need immediate fixes.

**Most Common**: Encoding issues, file handling, and type mismatches appear throughout the codebase.

**Most Impactful**: RAG search and tool execution directly affect end-user experience.

**Next Steps**:
1. Implement user ID validation at all entry points
2. Add proper error boundaries with rollback
3. Implement token counting before LLM calls
4. Add resource limits and timeouts
5. Comprehensive test suite for edge cases

---

**Document Generated**: December 24, 2025  
**Status**: Analysis Complete  
**Recommendations**: Prioritized by severity and effort
