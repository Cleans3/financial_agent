/**
 * Enhanced Thinking Steps Component
 * Displays detailed workflow execution steps in real-time
 * Shows all 17 nodes from V4 workflow with detailed descriptions
 * Features: Auto-scrolling, scrollable list (3 steps visible), live progress
 */

import { useState, useRef, useEffect } from "react";
import { ChevronDown, ChevronUp, Clock, AlertCircle, CheckCircle2, Loader2 } from "lucide-react";
import { getStepColorClass } from "../utils/workflowSteps";

const ThinkingSteps = ({ steps = [], isCollapsed = false }) => {
  const [showDetails, setShowDetails] = useState(true);
  const [expandedStepId, setExpandedStepId] = useState(null);
  const scrollContainerRef = useRef(null);

  // Auto-scroll to latest step when new step is added
  useEffect(() => {
    if (scrollContainerRef.current) {
      // Scroll to bottom after a small delay to ensure DOM is updated
      setTimeout(() => {
        scrollContainerRef.current.scrollTop = scrollContainerRef.current.scrollHeight;
      }, 50);
    }
  }, [steps.length]);

  if (!steps || steps.length === 0) {
    return null;
  }

  // Count completed vs total steps
  const completedSteps = steps.filter(s => s.status === 'completed').length;
  const totalSteps = steps.length;
  const hasErrors = steps.some(s => s.status === 'error');

  // Group steps by phase
  const phases = {
    'Input Processing': ['PROMPT_HANDLER', 'FILE_HANDLER'],
    'Classification': ['CLASSIFY', 'DIRECT_RESPONSE'],
    'File Processing': ['EXTRACT_FILE', 'INGEST_FILE'],
    'Query Enhancement': ['REWRITE_EVAL', 'REWRITE_FILE', 'REWRITE_CONVO'],
    'Document Retrieval': ['RETRIEVE', 'FILTER', 'ANALYZE'],
    'Tool Processing': ['SELECT_TOOLS', 'EXECUTE_TOOLS'],
    'Synthesis': ['SUMMARY_TOOLS', 'QUERY_REFORMULATION'],
    'Final Output': ['GENERATE'],
  };

  const getStepIcon = (step) => {
    if (step.status === 'completed') {
      return <CheckCircle2 className="w-4 h-4 text-emerald-400" />;
    } else if (step.status === 'error') {
      return <AlertCircle className="w-4 h-4 text-red-400" />;
    } else if (step.status === 'in-progress') {
      return <Loader2 className="w-4 h-4 text-cyan-400 animate-spin" />;
    }
    return <div className="w-4 h-4 rounded-full border border-slate-600" />;
  };

  const getStepStatusText = (status) => {
    switch (status) {
      case 'completed': return 'Completed';
      case 'in-progress': return 'Processing...';
      case 'error': return 'Error';
      case 'pending': return 'Pending';
      default: return 'Unknown';
    }
  };

  const getStatusColor = (status) => {
    switch (status) {
      case 'completed': return 'text-emerald-400';
      case 'in-progress': return 'text-cyan-400';
      case 'error': return 'text-red-400';
      case 'pending': return 'text-slate-400';
      default: return 'text-slate-400';
    }
  };

  return (
    <div className="border-l-4 border-blue-500 bg-slate-900/60 rounded-r-lg p-3">
      {/* Header */}
      <button
        onClick={() => setShowDetails(!showDetails)}
        className="flex items-center gap-2 text-blue-400 hover:text-blue-300 transition-colors text-sm font-semibold p-0 mb-3 w-full"
      >
        {showDetails ? (
          <ChevronUp className="w-5 h-5" />
        ) : (
          <ChevronDown className="w-5 h-5" />
        )}
        <span>🧠 Agent Thinking Process</span>
        <span className={`text-xs px-2 py-1 rounded-full ${
          hasErrors ? 'bg-red-500/20 text-red-300' : 'bg-emerald-500/20 text-emerald-300'
        }`}>
          {completedSteps}/{totalSteps} steps
        </span>
        {showDetails && totalSteps > 3 && (
          <span className="text-xs px-2 py-1 rounded-full bg-slate-700/50 text-slate-300 ml-auto">
            Scroll for more ↓
          </span>
        )}
        {completedSteps === totalSteps && (
          <span className="text-xs px-2 py-1 rounded-full bg-emerald-500/20 text-emerald-300 ml-auto">
            ✓ Complete
          </span>
        )}
      </button>

      {/* Progress bar */}
      <div className="w-full h-1.5 bg-slate-800/50 rounded-full mb-3 overflow-hidden">
        <div
          className={`h-full rounded-full transition-all duration-500 ${
            hasErrors ? 'bg-red-500' : 'bg-emerald-500'
          }`}
          style={{ width: `${(completedSteps / totalSteps) * 100}%` }}
        />
      </div>

      {/* Steps list - Scrollable container */}
      {showDetails && (
        <div
          ref={scrollContainerRef}
          className="space-y-2 overflow-y-auto pr-2 mb-2"
          style={{
            scrollBehavior: 'smooth',
            maxHeight: 'calc(3 * 85px + 8px)', // Approximately 3 steps with some spacing
            scrollbarWidth: 'thin',
            scrollbarColor: '#475569 #1e293b'
          }}
        >
          {steps.map((step, idx) => {
            const stepId = step.id || step.step_id || idx;
            const isExpanded = expandedStepId === stepId;
            const colors = getStepColorClass(step.color || 'slate');

            return (
              <div
                key={stepId}
                className={`rounded-lg border-l-2 p-3 transition-all ${colors.border} ${colors.bg}`}
              >
                {/* Step header */}
                <button
                  onClick={() => setExpandedStepId(isExpanded ? null : stepId)}
                  className="w-full flex items-start justify-between gap-2 text-left hover:opacity-80 transition-opacity"
                >
                  <div className="flex items-start gap-2 flex-1 min-w-0">
                    {getStepIcon(step)}
                    <div className="flex-1 min-w-0">
                      <div className={`font-semibold text-sm ${colors.title}`}>
                        {step.title || step.name || `Step ${idx + 1}`}
                      </div>
                      <div className="text-slate-400 text-xs mt-1">
                        {step.description || 'Processing...'}
                      </div>
                    </div>
                  </div>
                  {(step.duration || step.details || step.metadata) && (
                    <ChevronDown
                      className={`w-4 h-4 text-slate-400 flex-shrink-0 transition-transform ${
                        isExpanded ? 'rotate-180' : ''
                      }`}
                    />
                  )}
                </button>

                {/* Step result/status */}
                {step.result && (
                  <div className="text-xs mt-2 inline-block px-2 py-1 rounded-full bg-emerald-900/30">
                    <span className="mr-1">✓</span>
                    <span className="text-emerald-300">{step.result}</span>
                  </div>
                )}

                {step.error && (
                  <div className="text-xs mt-2 inline-block px-2 py-1 rounded-full bg-red-900/30">
                    <span className="mr-1">✗</span>
                    <span className="text-red-300">{step.error}</span>
                  </div>
                )}

                {/* Status badge */}
                <div className={`text-xs mt-2 inline-block px-2 py-1 rounded-full ml-auto ${getStatusColor(step.status)} opacity-75`}>
                  {getStepStatusText(step.status)}
                  {step.duration && <span className="ml-1">({step.duration}ms)</span>}
                </div>

                {/* Expanded details */}
                {isExpanded && (step.details || step.metadata) && (
                  <div className="mt-3 pt-3 border-t border-slate-700 space-y-2">
                    {step.details && (
                      <div className="text-slate-300 text-xs leading-relaxed">
                        <span className="text-slate-400 block mb-1">Details:</span>
                        {step.details}
                      </div>
                    )}

                    {step.metadata && Object.keys(step.metadata).length > 0 && (
                      <div className="bg-slate-800/50 rounded p-2">
                        <div className="text-slate-400 text-xs mb-1.5 font-semibold">Metadata:</div>
                        <div className="space-y-1">
                          {Object.entries(step.metadata).map(([key, value]) => (
                            <div key={key} className="text-xs text-slate-400 flex justify-between">
                              <span className="text-slate-500">{key}:</span>
                              <span className="text-cyan-300 font-mono">
                                {typeof value === 'object' ? JSON.stringify(value) : String(value)}
                              </span>
                            </div>
                          ))}
                        </div>
                      </div>
                    )}
                  </div>
                )}
              </div>
            );
          })}
        </div>
      )}

      {/* Phase Summary (when collapsed) */}
      {!showDetails && (
        <div className="text-xs text-slate-400 space-y-1 px-1">
          {Object.entries(phases).map(([phaseName, stepIds]) => {
            const phaseSteps = steps.filter(s => stepIds.includes(s.id));
            if (phaseSteps.length === 0) return null;

            const completed = phaseSteps.filter(s => s.status === 'completed').length;
            const hasError = phaseSteps.some(s => s.status === 'error');

            return (
              <div key={phaseName} className="flex items-center justify-between py-1">
                <span>{phaseName}</span>
                <span className={`${hasError ? 'text-red-400' : 'text-slate-500'}`}>
                  {completed}/{phaseSteps.length}
                </span>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
};

// Add webkit scrollbar styling for better UX
const styles = `
  /* Webkit scrollbar styling (Chrome, Safari, Edge) */
  div[ref*="scroll"]::-webkit-scrollbar {
    width: 6px;
  }
  
  div[ref*="scroll"]::-webkit-scrollbar-track {
    background: #1e293b;
    border-radius: 3px;
  }
  
  div[ref*="scroll"]::-webkit-scrollbar-thumb {
    background: #475569;
    border-radius: 3px;
  }
  
  div[ref*="scroll"]::-webkit-scrollbar-thumb:hover {
    background: #64748b;
  }
`;

export default ThinkingSteps;
