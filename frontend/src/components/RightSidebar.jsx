import { ChevronDown, Lightbulb } from "lucide-react";
import React, { useState, memo } from "react";

const RightSidebar = memo(({ isOpen = true, onToggle }) => {
  const [examplesExpanded, setExamplesExpanded] = useState(true);

  if (!isOpen) return null;

  const examples = [
    { text: "Thông tin về công ty VNM", category: "Thông tin công ty" },
    { text: "Cổ đông lớn của VCB", category: "Cổ đông" },
    { text: "Ban lãnh đạo HPG", category: "Ban lãnh đạo" },
    { text: "Công ty con của VNM", category: "Công ty con" },
    { text: "Giá VCB 3 tháng gần nhất", category: "Dữ liệu giá" },
    { text: "Tính SMA-20 cho HPG", category: "Phân tích kỹ thuật" },
    { text: "RSI của VIC hiện tại", category: "Phân tích kỹ thuật" },
    { text: "Sự kiện gần đây của FPT", category: "Sự kiện" },
  ];

  const handleExampleClick = (text) => {
    const textarea = document.querySelector("textarea");
    if (textarea) {
      textarea.value = text;
      textarea.focus();
      // Trigger input event for React state update
      textarea.dispatchEvent(new Event('input', { bubbles: true }));
    }
  };

  return (
    <aside className="hidden 2xl:flex flex-col w-80 bg-gradient-to-b from-slate-800 to-slate-900 backdrop-blur-sm border-l border-slate-700 flex-shrink-0">
      {/* Header */}
      <div className="px-4 py-4 border-b border-slate-700">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Lightbulb className="w-5 h-5 text-amber-400" />
            <h2 className="text-lg font-semibold text-white">Gợi ý</h2>
          </div>
          {onToggle && (
            <button
              onClick={onToggle}
              className="p-2 hover:bg-slate-700 rounded-lg transition-colors"
              title="Collapse sidebar"
            >
              <ChevronDown className="w-5 h-5 text-slate-400 rotate-90" />
            </button>
          )}
        </div>
      </div>

      {/* Content */}
      <div className="flex-1 overflow-y-auto px-4 py-4">
        {/* Example Questions */}
        <div>
          <button
            onClick={() => setExamplesExpanded(!examplesExpanded)}
            className="flex items-center gap-2 px-3 py-2 w-full text-sm font-semibold text-slate-300 hover:text-white transition-colors"
          >
            <ChevronDown className={`w-4 h-4 transition-transform ${examplesExpanded ? "" : "-rotate-90"}`} />
            Câu hỏi mẫu
          </button>
          {examplesExpanded && (
            <div className="space-y-2 mt-2">
              {examples.map((example, idx) => (
                <button
                  key={idx}
                  className="w-full text-left p-3 rounded-lg bg-slate-700/50 hover:bg-slate-700 
                           transition-all duration-200 border border-slate-600/50 hover:border-cyan-500/50 group"
                  onClick={() => handleExampleClick(example.text)}
                >
                  <div className="text-xs text-slate-400 mb-1">{example.category}</div>
                  <div className="text-sm text-slate-200 group-hover:text-cyan-300 transition-colors">
                    {example.text}
                  </div>
                </button>
              ))}
            </div>
          )}
        </div>
      </div>
    </aside>
  );
});

RightSidebar.displayName = "RightSidebar";
export default RightSidebar;
