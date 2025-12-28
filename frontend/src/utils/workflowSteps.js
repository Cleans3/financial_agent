/**
 * Workflow Steps Configuration
 * Maps LangGraph V4 workflow nodes to frontend display steps
 * 
 * V4 Workflow (13-node architecture):
 * 1. PROMPT_HANDLER → Route prompt vs files
 * 2. FILE_HANDLER → Handle file uploads
 * 3. CLASSIFY → Classify query type (greeting, chitchat, finance)
 * 4. DIRECT_RESPONSE → Response for non-financial queries
 * 5. EXTRACT_FILE → Extract text from uploaded files
 * 6. INGEST_FILE → Process & vectorize documents
 * 7. REWRITE_EVAL → Evaluate if query needs rewriting
 * 8. REWRITE_FILE → Rewrite query with file context
 * 9. REWRITE_CONVO → Rewrite query with conversation context
 * 10. RETRIEVE → RAG retrieval with advanced strategies
 * 11. FILTER → Filter & rank results with RRF
 * 12. ANALYZE → Analyze detected data types
 * 13. SELECT_TOOLS → Intelligently select tools
 * 14. EXECUTE_TOOLS → Execute selected tools
 * 15. SUMMARY_TOOLS → Apply summary techniques
 * 16. QUERY_REFORMULATION → Build context for LLM
 * 17. GENERATE → Generate final answer
 */

export const WORKFLOW_STEPS = {
  // Entry Points
  PROMPT_HANDLER: {
    id: 'prompt_handler',
    order: 1,
    title: '📥 Input Processing',
    icon: '📥',
    color: 'slate',
    description: 'Analyzing your input...',
    details: 'Determining whether this is a prompt, file upload, or conversation continuation',
  },
  
  FILE_HANDLER: {
    id: 'file_handler',
    order: 2,
    title: '📄 File Detection',
    icon: '📄',
    color: 'slate',
    description: 'Processing uploaded files...',
    details: 'Checking for files in your message',
  },

  CLASSIFY: {
    id: 'classify',
    order: 3,
    title: '🔍 Query Classification',
    icon: '🔍',
    color: 'blue',
    description: 'Classifying your query...',
    details: 'Understanding whether your query is a greeting, chitchat, or financial question',
  },

  DIRECT_RESPONSE: {
    id: 'direct_response',
    order: 4,
    title: '💬 Conversational Response',
    icon: '💬',
    color: 'cyan',
    description: 'Generating response...',
    details: 'Responding to your message without tools or document analysis',
  },

  // File Processing
  EXTRACT_FILE: {
    id: 'extract_file',
    order: 5,
    title: '🔄 File Extraction',
    icon: '🔄',
    color: 'amber',
    description: 'Extracting text from files...',
    details: 'Converting PDF/DOCX/Images to readable text',
  },

  INGEST_FILE: {
    id: 'ingest_file',
    order: 6,
    title: '⚙️ Document Processing',
    icon: '⚙️',
    color: 'purple',
    description: 'Processing documents...',
    details: 'Creating structural chunks (2) and metric-centric chunks (~9). Creating embeddings and storing in Qdrant',
  },

  // Query Enhancement
  REWRITE_EVAL: {
    id: 'rewrite_eval',
    order: 7,
    title: '✏️ Rewrite Evaluation',
    icon: '✏️',
    color: 'amber',
    description: 'Evaluating if query needs enhancement...',
    details: 'Checking if your query should be rewritten with file or conversation context',
  },

  REWRITE_FILE: {
    id: 'rewrite_file',
    order: 8,
    title: '📝 File Context Injection',
    icon: '📝',
    color: 'amber',
    description: 'Adding file context to query...',
    details: 'Enhancing your question with information about uploaded files',
  },

  REWRITE_CONVO: {
    id: 'rewrite_convo',
    order: 9,
    title: '💭 Conversation Context Injection',
    icon: '💭',
    color: 'amber',
    description: 'Adding conversation context...',
    details: 'Enriching your query with relevant information from conversation history',
  },

  // RAG Pipeline
  RETRIEVE: {
    id: 'retrieve',
    order: 10,
    title: '🗂️ RAG Retrieval',
    icon: '🗂️',
    color: 'emerald',
    description: 'Searching documents...',
    details: 'Using dual retrieval strategy: metadata-only for generic queries, RRF for specific questions',
  },

  FILTER: {
    id: 'filter',
    order: 11,
    title: '🎯 Result Filtering',
    icon: '🎯',
    color: 'emerald',
    description: 'Filtering & ranking results...',
    details: 'Deduplicating chunks and ranking with RRF (Reciprocal Rank Fusion)',
  },

  ANALYZE: {
    id: 'analyze',
    order: 12,
    title: '📊 Data Type Analysis',
    icon: '📊',
    color: 'sky',
    description: 'Analyzing data types...',
    details: 'Detecting tables, numeric data, and text content',
  },

  // Tool & Processing
  SELECT_TOOLS: {
    id: 'select_tools',
    order: 13,
    title: '🔧 Tool Selection',
    icon: '🔧',
    color: 'purple',
    description: 'Selecting appropriate tools...',
    details: 'Determining which financial tools (company info, stock prices, technical analysis) are needed',
  },

  EXECUTE_TOOLS: {
    id: 'execute_tools',
    order: 14,
    title: '⚡ Tool Execution',
    icon: '⚡',
    color: 'purple',
    description: 'Executing tools...',
    details: 'Running selected tools to gather financial data and calculations',
  },

  // Synthesis
  SUMMARY_TOOLS: {
    id: 'summary_tools',
    order: 15,
    title: '📈 Summary Synthesis',
    icon: '📈',
    color: 'cyan',
    description: 'Synthesizing summary...',
    details: 'Using specialized techniques: comparative analysis, anomaly detection, narrative arc, etc.',
  },

  QUERY_REFORMULATION: {
    id: 'query_reformulation',
    order: 16,
    title: '🧩 Context Assembly',
    icon: '🧩',
    color: 'cyan',
    description: 'Assembling context...',
    details: 'Combining RAG results, tool outputs, and summaries into structured context for LLM',
  },

  // Final Step
  GENERATE: {
    id: 'generate',
    order: 17,
    title: '✨ Answer Generation',
    icon: '✨',
    color: 'emerald',
    description: 'Generating final answer...',
    details: 'LLM synthesizing comprehensive response from all available context',
  },
};

/**
 * Get color class for a workflow step
 */
export const getStepColorClass = (stepColor) => {
  const colorMap = {
    slate: { border: 'border-slate-600/50', bg: 'bg-slate-900/20', title: 'text-slate-300' },
    blue: { border: 'border-blue-600/50', bg: 'bg-blue-900/20', title: 'text-blue-300' },
    cyan: { border: 'border-cyan-600/50', bg: 'bg-cyan-900/20', title: 'text-cyan-300' },
    amber: { border: 'border-amber-600/50', bg: 'bg-amber-900/20', title: 'text-amber-300' },
    purple: { border: 'border-purple-600/50', bg: 'bg-purple-900/20', title: 'text-purple-300' },
    emerald: { border: 'border-emerald-600/50', bg: 'bg-emerald-900/20', title: 'text-emerald-300' },
    sky: { border: 'border-sky-600/50', bg: 'bg-sky-900/20', title: 'text-sky-300' },
  };
  return colorMap[stepColor] || colorMap.slate;
};

/**
 * Transform backend step to frontend display step
 */
export const transformBackendStep = (backendStep) => {
  if (!backendStep) return null;

  const stepId = backendStep.id || backendStep.step_id || backendStep.title?.toLowerCase().replace(/\s+/g, '_');
  const workflowStep = WORKFLOW_STEPS[Object.keys(WORKFLOW_STEPS).find(key => 
    WORKFLOW_STEPS[key].id === stepId || WORKFLOW_STEPS[key].title === backendStep.title
  )];

  return {
    ...workflowStep,
    ...backendStep,
    status: backendStep.status || 'in-progress', // 'pending', 'in-progress', 'completed', 'error'
    result: backendStep.result || backendStep.metadata?.result,
    error: backendStep.error,
    duration: backendStep.duration,
    metadata: backendStep.metadata || {},
  };
};

/**
 * Get summary text for completed workflow
 */
export const getWorkflowSummary = (steps) => {
  const stepsByPhase = {
    input: steps.filter(s => ['PROMPT_HANDLER', 'FILE_HANDLER'].includes(s.id)),
    classification: steps.filter(s => ['CLASSIFY', 'DIRECT_RESPONSE'].includes(s.id)),
    fileProcessing: steps.filter(s => ['EXTRACT_FILE', 'INGEST_FILE'].includes(s.id)),
    queryEnhancement: steps.filter(s => ['REWRITE_EVAL', 'REWRITE_FILE', 'REWRITE_CONVO'].includes(s.id)),
    retrieval: steps.filter(s => ['RETRIEVE', 'FILTER', 'ANALYZE'].includes(s.id)),
    tools: steps.filter(s => ['SELECT_TOOLS', 'EXECUTE_TOOLS'].includes(s.id)),
    synthesis: steps.filter(s => ['SUMMARY_TOOLS', 'QUERY_REFORMULATION'].includes(s.id)),
    generation: steps.filter(s => ['GENERATE'].includes(s.id)),
  };

  return {
    totalSteps: steps.length,
    phases: stepsByPhase,
    completedSteps: steps.filter(s => s.status === 'completed').length,
    errorCount: steps.filter(s => s.status === 'error').length,
  };
};
