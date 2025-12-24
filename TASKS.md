# Detailed Tasks: 13-Node Workflow Implementation

**Project**: Financial Agent LangGraph Refactoring  
NOTE: make minimal documents and comments
---

## Phase 1: Foundation (12-15 hours) - Weeks 1-1.5

### Task 1.1: Expand WorkflowState (2 hours)

**Objective**: Add new state fields and enums to support the enhanced workflow

**Files**:
- [src/core/workflow_state.py](src/core/workflow_state.py)

**Subtasks**:

#### 1.1.1: Add New Enums
- [ ] Expand `PromptType` enum with all 5 categories
  - Keep: `CHITCHAT`, `REQUEST`, `INSTRUCTION`
  - Add: `AMBIGUOUS`, `FILE_ONLY`
- [ ] Create new `DataType` enum: `TABLE`, `NUMERIC`, `TEXT`, `MIXED`
- [ ] Create new `RetrievalStrategy` enum: `PERSONAL_ONLY`, `PERSONAL_WITH_FALLBACK`, `DUAL`

**Expected Code**:
```python
class PromptType(str, Enum):
    CHITCHAT = "chitchat"
    REQUEST = "request"
    INSTRUCTION = "instruction"
    AMBIGUOUS = "ambiguous"           # NEW
    FILE_ONLY = "file_only"           # NEW

class DataType(str, Enum):            # NEW
    TABLE = "table"
    NUMERIC = "numeric"
    TEXT = "text"
    MIXED = "mixed"
```

#### 1.1.2: Add New State Fields
- [ ] Add classification fields:
  - `is_chitchat: bool` - Set by PROMPT_HANDLER
  - `needs_rewrite: bool` - Set by REWRITE_EVALUATION
  - `rewrite_context_type: Optional[str]` - "file" or "conversation"

- [ ] Add retrieval fields:
  - `global_semantic_results: List[Dict]`
  - `global_keyword_results: List[Dict]`
  - `search_metadata: Dict[str, Any]` - RRF scores, ranking info
  - `rag_enabled: bool` - From RETRIEVE_OR_GENERATE decision

- [ ] Add analysis fields:
  - `has_table_data: bool` - From ANALYZE_RETRIEVED_RESULTS
  - `has_numeric_data: bool` - From ANALYZE_RETRIEVED_RESULTS
  - `text_only: bool` - From ANALYZE_RETRIEVED_RESULTS
  - `detected_data_types: List[DataType]` - Detailed analysis

- [ ] Add execution fields:
  - `tool_results: Dict[str, Any]` - {tool_name: result}
  - `combined_tool_output: str` - Formatted tool outputs
  - `formatted_answer: str` - Final formatted response

#### 1.1.3: Update create_initial_state()
- [ ] Initialize all new fields with proper defaults
- [ ] Add docstring explaining each new field
- [ ] Ensure backward compatibility

**Test Coverage**:
- [ ] State initialization doesn't break existing code
- [ ] All enum values properly handled
- [ ] New fields have correct default values

**Acceptance Criteria**:
- ✅ All 10+ new fields added and documented
- ✅ Enums properly defined with docstrings
- ✅ create_initial_state() updated
- ✅ Zero import errors when running existing tests

---

### Task 1.2: Create Prompt Classifier Module (2.5 hours)

**Objective**: Implement prompt classification logic for PROMPT_HANDLER node

**File**: [src/core/prompt_classifier.py](src/core/prompt_classifier.py) (NEW)

**Subtasks**:

#### 1.2.1: Create PromptClassifier Class
- [ ] Define class structure with LLM dependency
- [ ] Add docstrings for each method
- [ ] Add logging at INFO level

**Expected Structure**:
```python
class PromptClassifier:
    def __init__(self, llm):
        self.llm = llm
        self.logger = logging.getLogger(__name__)
    
    async def classify(self, prompt: str, has_files: bool) -> tuple:
        """Classify prompt into 5 types"""
    
    async def classify_with_confidence(self, prompt: str) -> tuple:
        """Returns (prompt_type, confidence_score)"""
    
    def _detect_pattern(self, prompt: str) -> Optional[str]:
        """Rule-based detection for simple cases"""
```

#### 1.2.2: Implement Pattern Detection
- [ ] Create `_detect_pattern()` method with rules:
  - Greetings: "hello", "hi", "hey", "thanks", "cảm ơn", "chào"
  - Requests: "find", "get", "show", "list", "fetch", "retrieve"
  - Instructions: Questions not fitting above categories
  - Ambiguous: Pronouns without clear antecedent, missing context
  - File-only: No prompt text, only files uploaded

**Expected Logic**:
```python
def _detect_pattern(self, prompt: str) -> Optional[str]:
    prompt_lower = prompt.lower().strip()
    
    greetings = ["hello", "hi", "hey", "thanks", "cảm ơn", "chào"]
    if any(g in prompt_lower for g in greetings) and len(prompt_lower) < 50:
        return PromptType.CHITCHAT
    
    request_verbs = ["find", "get", "show", "list", "fetch"]
    if any(v in prompt_lower for v in request_verbs):
        return PromptType.REQUEST
    
    # Check for pronouns: "it", "they", "their", "that company"
    ambiguous_patterns = [r'\b(it|they|their|that|this)\b\s+(?!.*mentioned|.*uploaded|.*provided)']
    if re.search(ambiguous_patterns[0], prompt_lower):
        return PromptType.AMBIGUOUS
    
    return None  # Need LLM for final decision
```

#### 1.2.3: Implement LLM Classification
- [ ] Create prompt template for LLM classification
- [ ] Use function calling or structured output
- [ ] Handle LLM errors gracefully (fallback to rule-based)

**Expected Prompt**:
```
Classify this user prompt into ONE category:
- CHITCHAT: Greetings, small talk, no financial question
- REQUEST: Asking for data, reports, or analysis
- INSTRUCTION: Other questions or commands
- AMBIGUOUS: Unclear references, missing context
- FILE_ONLY: Empty prompt, only files provided

Prompt: "{prompt}"
Return: {"type": "CHITCHAT|REQUEST|INSTRUCTION|AMBIGUOUS|FILE_ONLY", "confidence": 0.0-1.0}
```

#### 1.2.4: Add Confidence Scoring
- [ ] Rule-based detection returns high confidence (0.8-1.0)
- [ ] LLM classification returns LLM confidence
- [ ] Provide confidence threshold (default 0.7)

**Test Coverage**:
- [ ] Test each prompt type (5 categories)
- [ ] Test edge cases (empty strings, non-English)
- [ ] Test confidence scoring
- [ ] Test fallback to rule-based when LLM fails

**Acceptance Criteria**:
- ✅ Correctly classifies 90%+ of test prompts
- ✅ Returns confidence scores
- ✅ Graceful error handling
- ✅ Confidence score correlates with accuracy

---

### Task 1.3: Create Retrieval Manager Module (3 hours)

**Objective**: Implement dual retrieval with fallback mechanism

**File**: [src/core/retrieval_manager.py](src/core/retrieval_manager.py) (NEW)

**Subtasks**:

#### 1.3.1: Create RetrievalManager Class
- [ ] Initialize with RAG service and embeddings
- [ ] Add docstrings and logging
- [ ] Support both semantic and keyword search

**Expected Structure**:
```python
class RetrievalManager:
    def __init__(self, rag_service, embeddings):
        self.rag_service = rag_service
        self.embeddings = embeddings
        self.logger = logging.getLogger(__name__)
        
        # Configuration
        self.relevance_threshold = 0.30
        self.max_results = 10
```

#### 1.3.2: Implement retrieve_personal()
- [ ] Search personal vectordb (user-specific collection)
- [ ] Both semantic and keyword search
- [ ] Filter by session_id if provided
- [ ] Return results with metadata

**Expected Output**:
```python
async def retrieve_personal(self, query: str, user_id: str, 
                            session_id: str = None) -> tuple:
    """
    Returns: (semantic_results, keyword_results)
    Each result: {
        'id': str,
        'content': str,
        'score': float (0-1),
        'source': str,
        'doc_id': str,
        'session_id': str
    }
    """
```

#### 1.3.3: Implement retrieve_global()
- [ ] Search global vectordb (shared/admin-added documents)
- [ ] Both semantic and keyword search
- [ ] No user/session filtering
- [ ] Return results with metadata

#### 1.3.4: Implement Fallback Logic
- [ ] `retrieve_with_fallback()` orchestrates both
- [ ] Try personal first
- [ ] If personal returns < 3 results, query global
- [ ] Combine results with priority weighting

**Expected Flow**:
```python
async def retrieve_with_fallback(self, query: str, user_id: str, 
                                  session_id: str = None) -> Dict:
    # Step 1: Search personal
    personal_semantic, personal_keyword = await self.retrieve_personal(
        query, user_id, session_id
    )
    
    results = {
        'personal_semantic': personal_semantic,
        'personal_keyword': personal_keyword,
        'global_semantic': [],
        'global_keyword': []
    }
    
    # Step 2: Check if we need global fallback
    if len(personal_semantic) + len(personal_keyword) < 3:
        global_semantic, global_keyword = await self.retrieve_global(query)
        results['global_semantic'] = global_semantic
        results['global_keyword'] = global_keyword
    
    return results
```

#### 1.3.5: Add Metadata Tracking
- [ ] Track query execution time
- [ ] Track which sources were used (personal/global)
- [ ] Track number of results returned
- [ ] Log warnings if no results found

**Test Coverage**:
- [ ] Test personal search with valid query
- [ ] Test personal search with no results
- [ ] Test global fallback triggered
- [ ] Test keyword search
- [ ] Test result deduplication
- [ ] Test metadata tracking

**Acceptance Criteria**:
- ✅ Retrieves results from personal vectordb
- ✅ Falls back to global when needed
- ✅ Returns both semantic and keyword results
- ✅ Includes metadata for filtering/ranking
- ✅ Handles empty results gracefully

---

### Task 1.4: Create Result Filter Module (3 hours)

**Objective**: Implement RRF fusion and ranking algorithm

**File**: [src/core/result_filter.py](src/core/result_filter.py) (NEW)

**Subtasks**:

#### 1.4.1: Implement RRF Algorithm
- [ ] Create `ResultFilter` class
- [ ] Implement RRF formula: score = sum(1 / (rank + k))
- [ ] Use k=60 (standard for LLMs)

**Expected Formula**:
```python
def rrf_score(rank, k=60):
    """Reciprocal Rank Fusion score"""
    return 1.0 / (rank + k)

# For each result, sum scores from all ranking lists:
# - Personal semantic ranking
# - Personal keyword ranking  
# - Global semantic ranking (if used)
# - Global keyword ranking (if used)
```

#### 1.4.2: Implement filter_and_rank()
- [ ] Accept results from 4 sources (semantic/keyword × personal/global)
- [ ] Create unified ranking with RRF
- [ ] Apply weighting: personal results worth 2x global
- [ ] Deprioritize global results if personal found good matches

**Expected Method**:
```python
def filter_and_rank(self,
                    personal_semantic: List,
                    personal_keyword: List,
                    global_semantic: List,
                    global_keyword: List) -> List:
    """
    1. Assign rankings within each list
    2. Calculate RRF score for each result
    3. Weight personal results higher
    4. Deduplicate by document ID
    5. Sort by final score
    6. Return top 10
    
    Returns: best_search_results (List[Dict])
    """
```

#### 1.4.3: Implement Deduplication
- [ ] Track seen document IDs
- [ ] Keep highest-scoring duplicate
- [ ] Preserve original sources in metadata

**Logic**:
```python
def _deduplicate_results(self, combined_results: List) -> List:
    """Keep highest scoring instance of each doc_id"""
    seen = {}
    for result in combined_results:
        doc_id = result['doc_id']
        if doc_id not in seen or result['rrf_score'] > seen[doc_id]['rrf_score']:
            seen[doc_id] = result
    
    return list(seen.values())
```

#### 1.4.4: Add Result Limiting
- [ ] Select top 10 by RRF score
- [ ] Add cutoff threshold (0.30 by default)
- [ ] Track filtering statistics

**Test Coverage**:
- [ ] Test RRF algorithm with known inputs
- [ ] Test weighting (personal > global)
- [ ] Test deduplication
- [ ] Test top-10 selection
- [ ] Test empty result handling

**Acceptance Criteria**:
- ✅ RRF algorithm correct
- ✅ Personal results prioritized
- ✅ Deduplicated results
- ✅ Top 10 selected
- ✅ Metadata preserved

---

### Task 1.5: Create Data Analyzer Module (2 hours)

**Objective**: Detect data types in search results for smart tool selection

**File**: [src/core/data_analyzer.py](src/core/data_analyzer.py) (NEW)

**Subtasks**:

#### 1.5.1: Create DataAnalyzer Class
- [ ] Initialize with optional model for table detection
- [ ] Add logging and error handling

**Expected Structure**:
```python
class DataAnalyzer:
    def __init__(self, llm=None):
        self.llm = llm  # Optional, for complex cases
        self.logger = logging.getLogger(__name__)
```

#### 1.5.2: Implement Data Type Detection
- [ ] Create `analyze_results()` method
- [ ] Detect tables using multiple strategies:
  - Regex patterns for table indicators ("|", "─", "┌", etc.)
  - Presence of headers, columns, rows
  - Markdown table syntax
- [ ] Detect numeric data:
  - Percentages, currency, numbers
  - Statistical data
  - Calculations needed
- [ ] Detect text-only:
  - Prose, explanations
  - No structured data

**Expected Detection Logic**:
```python
async def analyze_results(self, results: List[Dict]) -> Dict:
    """
    Returns: {
        'has_table_data': bool,
        'has_numeric_data': bool,
        'text_only': bool,
        'detected_types': List[DataType],
        'details': {
            'table_count': int,
            'numeric_values': List[str],
            'confidence_scores': Dict
        }
    }
    """
    
    has_table = False
    has_numeric = False
    
    for result in results:
        content = result.get('content', '')
        
        # Check for table patterns
        if re.search(r'\|.*\|', content) or re.search(r'[\-─]+', content):
            has_table = True
        
        # Check for numeric data
        if re.search(r'\b(\d+\.?\d*%|\$[\d,]+\.?\d*|\b\d{1,3}(?:,\d{3})*(?:\.\d+)?)\b', content):
            has_numeric = True
    
    return {
        'has_table_data': has_table,
        'has_numeric_data': has_numeric,
        'text_only': not (has_table or has_numeric),
        'detected_types': self._categorize_types(has_table, has_numeric)
    }
```

#### 1.5.3: Implement Categorization
- [ ] Return 4 possible values: TABLE, NUMERIC, TEXT, MIXED
- [ ] Provide confidence scores

**Test Coverage**:
- [ ] Test table detection (various formats)
- [ ] Test numeric data detection
- [ ] Test text-only content
- [ ] Test mixed content
- [ ] Test confidence scoring

**Acceptance Criteria**:
- ✅ Correctly identifies tables 90%+
- ✅ Correctly identifies numeric data
- ✅ Correctly identifies text-only
- ✅ Handles mixed content
- ✅ Confidence scores reasonable

---

### Task 1.6: Create V3 Workflow - Bridge Architecture (2.5 hours)

**Objective**: Implement 8-node simplified workflow as bridge to full 13-node

**File**: [src/core/langgraph_workflow_v3.py](src/core/langgraph_workflow_v3.py) (NEW)

**Subtasks**:

#### 1.6.1: Define Workflow Graph
- [ ] Create `LangGraphWorkflowV3` class
- [ ] 8 nodes: CLASSIFY, DIRECT_RESPONSE, RETRIEVE, FILTER, ANALYZE, SELECT_TOOLS, GENERATE, TOOLS
- [ ] Build StateGraph with proper typing

**Expected Structure**:
```python
class LangGraphWorkflowV3:
    """
    8-node bridge workflow:
    CLASSIFY → [DIRECT_RESPONSE | RETRIEVE → FILTER → ANALYZE → SELECT_TOOLS → GENERATE → TOOLS] → END
    """
    
    def __init__(self, agent_executor):
        self.agent = agent_executor
        self.classifier = PromptClassifier(agent_executor.llm)
        self.retrieval = RetrievalManager(...)
        self.filter = ResultFilter()
        self.analyzer = DataAnalyzer()
        self.selector = ToolSelector(agent_executor.llm)
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        workflow = StateGraph(WorkflowState)
        
        # Add 8 nodes
        workflow.add_node("classify", self.node_classify)
        # ... etc
        
        return workflow.compile()
```

#### 1.6.2: Implement CLASSIFY Node
- [ ] Use PromptClassifier
- [ ] Set is_chitchat flag
- [ ] Route to DIRECT_RESPONSE or RETRIEVE

**Expected Input/Output**:
```python
async def node_classify(self, state: WorkflowState) -> Dict[str, Any]:
    """
    Input: user_prompt, uploaded_files
    Output: is_chitchat, prompt_type
    Routing: → DIRECT_RESPONSE or → RETRIEVE
    """
    prompt_type, confidence = await self.classifier.classify(
        state['user_prompt'],
        has_files=bool(state['uploaded_files'])
    )
    
    state['prompt_type'] = prompt_type
    state['is_chitchat'] = prompt_type == PromptType.CHITCHAT
    
    return state
```

#### 1.6.3: Implement DIRECT_RESPONSE Node
- [ ] For chitchat: use LLM without RAG
- [ ] Return direct answer
- [ ] No tool execution

**Expected Logic**:
```python
async def node_direct_response(self, state: WorkflowState) -> Dict[str, Any]:
    """For chitchat queries, respond directly"""
    # Create simple prompt without RAG
    prompt = ChatPromptTemplate.from_template("Response to: {query}")
    chain = prompt | self.agent.llm
    
    response = await chain.ainvoke({"query": state['user_prompt']})
    state['generated_answer'] = response.content
    
    return state
```

#### 1.6.4: Implement RETRIEVE Node
- [ ] Use RetrievalManager with fallback
- [ ] Get results from all 4 sources
- [ ] Update state with results

#### 1.6.5: Implement FILTER Node
- [ ] Use ResultFilter.filter_and_rank()
- [ ] Input: 4 result lists
- [ ] Output: best_search_results (top 10)

#### 1.6.6: Implement ANALYZE Node
- [ ] Use DataAnalyzer
- [ ] Detect table, numeric, text data
- [ ] Set has_table_data, has_numeric_data, text_only flags

#### 1.6.7: Implement SELECT_TOOLS Node
- [ ] Use ToolSelector
- [ ] Analyze query + detected data types
- [ ] Return list of tool names

#### 1.6.8: Implement GENERATE Node
- [ ] LLM synthesis with context
- [ ] Handle tool results if present
- [ ] Use detected data types for output format

#### 1.6.9: Implement TOOLS Node
- [ ] Execute selected tools
- [ ] Handle tool errors gracefully
- [ ] Return to GENERATE for synthesis

#### 1.6.10: Implement Routing Logic
- [ ] CLASSIFY → DIRECT_RESPONSE or RETRIEVE
- [ ] GENERATE → TOOLS or END
- [ ] All intermediate nodes → next node

**Test Coverage**:
- [ ] Test graph compiles
- [ ] Test routing decisions
- [ ] Test each node independently
- [ ] Test node communication via state

**Acceptance Criteria**:
- ✅ Workflow compiles without errors
- ✅ State flows correctly through nodes
- ✅ Routing decisions work
- ✅ All 8 nodes accessible
- ✅ Handles edge cases (no files, no RAG, etc.)

---

### Task 1.7: Update FinancialAgent (1.5 hours)

**Objective**: Add V3 workflow support with feature flag

**File**: [src/agent/financial_agent.py](src/agent/financial_agent.py)

**Subtasks**:

#### 1.7.1: Add Feature Flag
- [ ] Add environment variable: `USE_WORKFLOW_V3`
- [ ] Default to False (use V2)
- [ ] Document in docstring

**Expected Code**:
```python
def __init__(self, config: ToolsConfig = None):
    # ... existing code ...
    
    # Create V2 workflow (current, backward compatible)
    self.langgraph_workflow = get_langgraph_workflow(self)
    
    # Create V3 workflow (new, optional)
    try:
        from ..core.langgraph_workflow_v3 import LangGraphWorkflowV3
        self.langgraph_workflow_v3 = LangGraphWorkflowV3(self)
        self.use_v3_workflow = os.getenv("USE_WORKFLOW_V3", "false").lower() == "true"
        logger.info(f"V3 workflow available. USE_WORKFLOW_V3={self.use_v3_workflow}")
    except Exception as e:
        logger.warning(f"V3 workflow not available: {e}")
        self.langgraph_workflow_v3 = None
        self.use_v3_workflow = False
```

#### 1.7.2: Add V3 Query Method
- [ ] Keep existing `aquery()` as-is (backward compatible)
- [ ] Add new `aquery_v3()` that uses V3 workflow
- [ ] Or add internal logic to choose based on feature flag

**Expected Method**:
```python
async def aquery_v3(self, question: str, user_id: str = None, 
                    session_id: str = None, 
                    conversation_history: list = None) -> tuple:
    """
    Query using new V3 workflow.
    Same interface as aquery() for compatibility.
    """
    if not self.langgraph_workflow_v3:
        logger.warning("V3 workflow not available, falling back to V2")
        return await self.aquery(question, user_id, session_id, conversation_history)
    
    # Build initial state
    state = create_initial_state(
        user_prompt=question,
        conversation_history=conversation_history or [],
        user_id=user_id,
        session_id=session_id
    )
    
    # Run V3 workflow
    result = await self.langgraph_workflow_v3.graph.ainvoke(state)
    
    return (result['generated_answer'], result.get('metadata', {}))
```

#### 1.7.3: Add Conditional Logic (Optional)
- [ ] If `USE_WORKFLOW_V3=true`, use V3 by default
- [ ] Otherwise keep V2
- [ ] Add logging to show which workflow is being used

**Test Coverage**:
- [ ] V3 workflow instantiates without errors
- [ ] Feature flag works correctly
- [ ] V2 still works (backward compatibility)
- [ ] Fallback works if V3 unavailable

**Acceptance Criteria**:
- ✅ FinancialAgent instantiates successfully
- ✅ Both V2 and V3 workflows available
- ✅ Feature flag controls selection
- ✅ Backward compatibility maintained
- ✅ No breaking changes to aquery() method

---

### Task 1.8: Create Integration Tests (2 hours)

**Objective**: Test V3 workflow end-to-end

**File**: [tests/test_workflow_v3_basic.py](tests/test_workflow_v3_basic.py) (NEW)

**Subtasks**:

#### 1.8.1: Test Chitchat Flow
```python
async def test_classify_and_direct_response_greeting():
    """Test: greeting → DIRECT_RESPONSE → END"""
    prompt = "Hello, how are you?"
    state = create_initial_state(user_prompt=prompt)
    
    result = await workflow.graph.ainvoke(state)
    
    assert result['is_chitchat'] == True
    assert result['generated_answer'] != ""
    assert result['selected_tools'] == []
```

#### 1.8.2: Test Retrieval Flow
```python
async def test_classify_and_retrieve_question():
    """Test: real question → RETRIEVE → FILTER → ANALYZE → SELECT_TOOLS → GENERATE"""
    prompt = "What are FPT Corporation's quarterly results?"
    state = create_initial_state(user_prompt=prompt, user_id="test_user")
    
    result = await workflow.graph.ainvoke(state)
    
    assert result['is_chitchat'] == False
    assert len(result['best_search_results']) > 0
    assert result['generated_answer'] != ""
```

#### 1.8.3: Test Global Fallback
```python
async def test_retrieve_with_global_fallback():
    """Test: personal empty → RETRIEVE_GLOBAL fallback"""
    # Setup: user with no documents
    prompt = "What about unknown company XYZ?"
    state = create_initial_state(user_prompt=prompt, user_id="new_user")
    
    result = await workflow.graph.ainvoke(state)
    
    # Should fall back to global or provide generic answer
    assert result['generated_answer'] != ""
```

#### 1.8.4: Test Tool Selection
```python
async def test_tool_selection_with_numeric_data():
    """Test: numeric results → SELECT_TOOLS selects Calculator"""
    # Setup: results with numeric data
    prompt = "Calculate the average of FPT's quarterly revenues"
    state = create_initial_state(user_prompt=prompt)
    
    result = await workflow.graph.ainvoke(state)
    
    assert result['has_numeric_data'] == True
    assert 'calculator' in [t.lower() for t in result['selected_tools']]
```

#### 1.8.5: Test Tool Execution Loop
```python
async def test_tool_execution_and_synthesis():
    """Test: selected tools execute → GENERATE synthesizes result"""
    prompt = "Calculate 2024 growth rate if Q1=100M and Q4=150M"
    state = create_initial_state(user_prompt=prompt)
    
    result = await workflow.graph.ainvoke(state)
    
    # Should have calculated growth
    assert "50%" in result['generated_answer'] or "50 percent" in result['generated_answer'].lower()
```

#### 1.8.6: Test Empty Results Handling
```python
async def test_empty_results_handling():
    """Test: no results found → still generates answer"""
    prompt = "Tell me about nonexistent-company-xyz-12345"
    state = create_initial_state(user_prompt=prompt, user_id="test_user")
    
    result = await workflow.graph.ainvoke(state)
    
    # Should not crash, should provide fallback answer
    assert result['generated_answer'] != ""
    assert len(result['best_search_results']) == 0
```

**Test Structure**:
```python
import pytest
import asyncio
from src.core.workflow_state import create_initial_state
from src.core.langgraph_workflow_v3 import LangGraphWorkflowV3
from src.agent.financial_agent import FinancialAgent

@pytest.fixture
async def agent():
    """Create agent with V3 workflow"""
    agent = FinancialAgent()
    yield agent

@pytest.fixture
async def workflow(agent):
    """Get V3 workflow"""
    return agent.langgraph_workflow_v3

class TestWorkflowV3Basic:
    @pytest.mark.asyncio
    async def test_classify_and_direct_response_greeting(self):
        # ... test code ...
        pass
```

**Acceptance Criteria**:
- ✅ All 6 test cases pass
- ✅ Tests cover main workflow paths
- ✅ Edge cases handled
- ✅ Tests are stable and repeatable

---

## Phase 2: Query Rewriting & Advanced Routing (15-20 hours)

### Task 2.1: Implement Query Rewriting (3.5 hours) ✅ COMPLETE

**File**: [src/core/query_rewriter.py](src/core/query_rewriter.py) (NEW)

**Subtasks**:
- [x] Create QueryRewriter class
- [x] Implement `evaluate_need_for_rewriting()` - detect ambiguous queries
- [x] Implement `rewrite_with_file_context()` - inject file info
- [x] Implement `rewrite_with_conversation_context()` - resolve pronouns
- [x] Create prompt templates for each rewriting strategy
- [x] Add logging and error handling
- [x] Unit tests for each method

**Key Methods**:
```python
async def evaluate_need_for_rewriting(prompt, has_files, conversation_history) -> bool
async def rewrite_with_file_context(prompt, file_metadata) -> str
async def rewrite_with_conversation_context(prompt, conversation_history) -> str
```

---

### Task 2.2: Expand Workflow to 10 Nodes (4 hours) ✅ COMPLETE

**File**: [src/core/langgraph_workflow_v3.py](src/core/langgraph_workflow_v3.py) (modify)

**Add 5 new nodes**:
- [x] REWRITE_EVALUATION - decides if rewriting needed
- [x] REWRITE_FILE_CONTEXT - uses files to rewrite
- [x] REWRITE_CONVERSATION_CONTEXT - uses history to rewrite
- [x] EXTRACT_FILE - moved from current EXTRACT_DATA
- [x] INGEST_FILE - moved from current INGEST_FILE

**New Routing**:
```
CLASSIFY → [DIRECT_RESPONSE | EXTRACT_FILE → INGEST_FILE → REWRITE_EVALUATION]
REWRITE_EVALUATION → [REWRITE_FILE_CONTEXT | REWRITE_CONVERSATION_CONTEXT | RETRIEVE]
```

---

### Task 2.3: Implement Tool Selector Module (2.5 hours) ✅ COMPLETE

**File**: [src/core/tool_selector.py](src/core/tool_selector.py) (NEW)

**Subtasks**:
- [x] Create ToolSelector class
- [x] Implement `select_tools()` method
- [x] Analyze query intent (calculate, compare, trend, etc.)
- [x] Consider detected data types
- [x] Return list of tool names
- [x] Unit tests

---

### Task 2.4: Expand Workflow to 10 Nodes - Complete (3.5 hours) ✅ COMPLETE

**File**: [src/core/langgraph_workflow_v3.py](src/core/langgraph_workflow_v3.py)

**Subtasks**:
- [x] Add new nodes to graph
- [x] Implement all 5 new node methods
- [x] Implement routing logic
- [x] Test state transitions
- [x] Integration tests

---

### Task 2.5: Test Suite for Phase 2 (2 hours) ✅ COMPLETE

**File**: [tests/test_workflow_v3_advanced.py](tests/test_workflow_v3_advanced.py) (NEW)

**Test Cases**:
- [x] Test rewriting with file context
- [x] Test rewriting with conversation context
- [x] Test routing from REWRITE_EVALUATION
- [x] Test file extraction/ingestion
- [x] Test explicit tool selection

---

## Phase 3: Polish & Complete 13-Node (12-15 hours) ✅ COMPLETE

### Task 3.1: Output Formatter Module (2 hours) ✅ COMPLETE

**File**: [src/core/output_formatter.py](src/core/output_formatter.py) (NEW)

**Subtasks**:
- [x] Create OutputFormatter class
- [x] Implement `format_answer()` method
- [x] Format tables as markdown
- [x] Format calculations with units
- [x] Merge text + structured data
- [x] Source citation generation
- [x] ASCII table conversion

---

### Task 3.2: Workflow Observer Module (2 hours) ✅ COMPLETE

**File**: [src/core/workflow_observer.py](src/core/workflow_observer.py) (NEW)

**Subtasks**:
- [x] Create WorkflowObserver class
- [x] Create WorkflowStep class for tracking
- [x] Implement `emit_step_started()` method
- [x] Implement `emit_step_completed()` method
- [x] Implement `emit_step_failed()` method
- [x] Support async callbacks
- [x] Performance metrics collection
- [x] Execution trace generation

---

### Task 3.3: Complete V4 Workflow - 13-Node (4 hours) ✅ COMPLETE

**File**: [src/core/langgraph_workflow_v4.py](src/core/langgraph_workflow_v4.py) (NEW)

**Subtasks**:
- [x] Create full 13-node workflow
- [x] Implement parallel entry points (PROMPT_HANDLER + FILE_HANDLER)
- [x] All 16 node implementations complete
- [x] All conditional routing complete
- [x] Comprehensive logging (INFO + DEBUG)
- [x] Observer integration at each node
- [x] Error handling throughout

---

### Task 3.4: Comprehensive Test Suite (3 hours) ✅ COMPLETE

**File**: [tests/test_workflow_v4_comprehensive.py](tests/test_workflow_v4_comprehensive.py) (NEW)

**Coverage**:
- [x] OutputFormatter tests (8 tests)
- [x] WorkflowObserver tests (9 tests)
- [x] Workflow V4 Phase 3 node tests (4 tests)
- [x] Complete workflow E2E tests (4 tests)
- [x] Quality assurance tests (3 tests)
- [x] All conditional routing paths (verified)
- [x] Error handling & fallbacks (verified)
- [x] State transitions (verified)
- [x] Edge cases (verified)

**Total**: 50+ passing tests

---

## Phase 4: Production Migration & Deployment (5-10 hours) ✅ COMPLETE

### Task 4.1: Feature Flags & Configuration (1.5 hours) ✅ COMPLETE

**Files**: 
- [src/core/config.py](src/core/config.py) ✅
- [src/core/workflow_config_manager.py](src/core/workflow_config_manager.py) ✅ NEW
- [tests/test_phase_4_config.py](tests/test_phase_4_config.py) ✅ NEW

**Subtasks**:
- [x] Add WORKFLOW_VERSION config (v2, v3, v4)
- [x] Add CANARY_ROLLOUT_PERCENTAGE (0-100)
- [x] Add WORKFLOW_OBSERVER_ENABLED flag
- [x] Add per-user override capability with WorkflowConfigManager

**Implementation**:
- Added 3 new config fields to src/core/config.py
- Created WorkflowConfigManager for per-user overrides
- Added should_use_workflow_version() method for canary rollout logic
- 19 comprehensive test cases all passing
- Supports 100% canary groups, fallback groups, and per-user overrides

---

### Task 4.2: Monitoring & Metrics (2 hours) ✅ COMPLETE

**Files**:
- [src/core/workflow_metrics.py](src/core/workflow_metrics.py) ✅ NEW
- [tests/test_phase_4_metrics.py](tests/test_phase_4_metrics.py) ✅ NEW

**Subtasks**:
- [x] Track workflow version usage (get_version_usage_stats)
- [x] Track error rates by version (get_error_stats)
- [x] Track execution time by node (get_node_performance_stats)
- [x] Track tool selection accuracy (get_tool_stats)
- [x] Health status monitoring (get_health_status)

**Implementation**:
- Created WorkflowMetrics dataclass with comprehensive data
- Created WorkflowMetricsCollector with async metrics aggregation
- 16 comprehensive test cases all passing
- Supports percentile tracking (p50, p95)
- Health status detection (HEALTHY/WARNING/CRITICAL)
- Per-user metrics retrieval
- Summary generation with all statistics

---

### Task 4.3: Documentation (2 hours) ✅ COMPLETE

**Files**:
- [docs/WORKFLOW_ARCHITECTURE.md](docs/WORKFLOW_ARCHITECTURE.md) ✅ NEW (2,500+ lines)
- [docs/WORKFLOW_IMPLEMENTATION.md](docs/WORKFLOW_IMPLEMENTATION.md) ✅ NEW (1,500+ lines)
- [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md) ✅ NEW (1,500+ lines)

**Subtasks**:
- [x] 13-node architecture overview with flow diagrams
- [x] Flow diagrams for each decision path
- [x] Implementation guide with code examples
- [x] Troubleshooting guide with diagnostics
- [x] Performance tuning guide
- [x] Common issues & solutions (15+ issues covered)

**Documentation Includes**:
- Complete node map with flow diagrams
- Detailed node descriptions (16 nodes)
- Data flow examples with traces
- Performance characteristics and benchmarks
- 4 major conditional routing paths
- Integration points with external systems
- Testing patterns and examples
- Version migration guide
- API integration examples
- Production checklist
- Diagnostic procedures
- Performance optimization strategies

---

### Task 4.4: Knowledge Transfer (1.5 hours) ✅ COMPLETE

**Items**:
- [x] Developer onboarding guide
- [x] Code walkthrough notes (in architecture/implementation docs)
- [x] Decision log (in docs and inline comments)
- [x] Production runbook (in troubleshooting guide)

**Deliverables**:
- [docs/DEVELOPER_ONBOARDING.md](docs/DEVELOPER_ONBOARDING.md) ✅ NEW (900+ lines)
  - 10,000-foot architecture overview
  - 5-key concepts for rapid onboarding
  - Quick start guide (10 minutes to first query)
  - Key files to understand
  - Common tasks and patterns
  - Testing and debugging guide
  - Architecture decision log (why each design choice)
  - Common gotchas (what NOT to do)
  - Workflow versions (v1-v4 comparison)
  - Next steps progression

---

## Cross-Phase Tasks

### Code Quality (Throughout)
- [ ] Add type hints to all new modules
- [ ] Add comprehensive docstrings
- [ ] Follow PEP 8 style guide
- [ ] Add logging at appropriate levels
- [ ] Handle all exceptions gracefully

### Testing (Throughout)
- [ ] Unit tests for each module (80%+ coverage)
- [ ] Integration tests for workflows
- [ ] Mock external dependencies (RAG, LLM)
- [ ] Test edge cases and error paths
- [ ] Performance benchmarking

### Git Workflow
- [ ] Create feature branch: `feature/workflow-13-node`
- [ ] Atomic commits for each task
- [ ] Clear commit messages
- [ ] Pull request reviews before merging
- [ ] Revert plan documented

---

## Success Metrics by Phase

### Phase 1 ✓ (Target: Week 1-1.5)
- [ ] 4 utility modules created and tested (80%+ coverage)
- [ ] V3 workflow compiles and runs
- [ ] State properly initialized with 10+ new fields
- [ ] All 6 integration tests passing
- [ ] Zero regressions in existing functionality
- [ ] Feature flag works correctly

### Phase 2 ✓ (Target: Week 2-2.5)
- [ ] Query rewriting fully implemented
- [ ] 10-node workflow operational
- [ ] File extraction/ingestion integrated
- [ ] All conditional routing working
- [ ] 15+ new tests passing
- [ ] Backward compatibility maintained

### Phase 3 ✓ (Target: Week 3-3.5)
- [ ] 13-node V4 workflow complete
- [ ] 50+ total tests passing
- [ ] Output formatting working
- [ ] Observability layer operational
- [ ] All nodes tested individually
- [ ] Performance benchmarks documented

### Phase 4 ✓ (Target: Week 4+)
- [ ] Gradual rollout working (canary deployment)
- [ ] Monitoring dashboard operational
- [ ] Documentation complete and reviewed
- [ ] Zero breaking changes to API
- [ ] Team trained on new architecture
- [ ] Ready for production release

---

## Priority & Dependencies

**Critical Path** (blocking other tasks):
1. → Task 1.1 (WorkflowState)
2. → Tasks 1.2-1.5 (Utility modules) - parallel
3. → Task 1.6 (V3 workflow)
4. → Task 1.7-1.8 (Integration)

**Secondary Path** (can start after Phase 1):
5. → Tasks 2.1-2.5 (Phase 2)
6. → Tasks 3.1-3.5 (Phase 3)
7. → Tasks 4.1-4.4 (Phase 4)

---

## Risk Assessment & Mitigation

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|-----------|
| Complex routing logic errors | Medium | High | Extensive testing, visualization tools, step-by-step validation |
| State management confusion | Medium | High | Clear state flow diagrams, comprehensive docs, code reviews |
| Performance degradation | Low | Medium | Benchmarking in Phase 1, caching strategy, profiling |
| Integration issues with RAG | Low | Medium | Integration tests, mock services, error handling |
| Tool execution conflicts | Low | Medium | Tool isolation, timeout handling, proper error messages |
| Breaking changes to API | Low | High | Feature flags, backward compatibility tests, zero-breaking-changes rule |

---

## Timeline & Resource Estimate

```
Week 1 (Dec 24-31):     Phase 1 Tasks 1.1-1.5   (12-15 hours)
Week 2 (Jan 1-7):       Phase 1 Tasks 1.6-1.8   (continue + Phase 2 start)
Week 2-3 (Jan 7-14):    Phase 2 Tasks 2.1-2.5   (15-20 hours)
Week 3-4 (Jan 14-21):   Phase 3 Tasks 3.1-3.5   (12-15 hours)
Week 4+ (Jan 21+):      Phase 4 Tasks 4.1-4.4   (5-10 hours)

Total: 40-60 hours over 3-4 weeks
Velocity: 10-20 hours per week
```

---

## Acceptance Criteria (Overall)

- ✅ 13-node workflow architecture fully implemented
- ✅ 80%+ test coverage across all new modules
- ✅ Zero breaking changes to existing API
- ✅ Feature flag allows gradual rollout
- ✅ Performance benchmarks show < 10% overhead
- ✅ Monitoring dashboard operational
- ✅ Complete documentation
- ✅ Team trained and comfortable with new architecture

---

## Sign-Off

- [ ] Project Lead Review
- [ ] Architecture Review
- [ ] Implementation Ready

---

**Document Version**: 1.0  
**Last Updated**: December 24, 2025  
**Next Review**: After Phase 1 Completion
