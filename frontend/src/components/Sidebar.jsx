import { X, BookOpen, Github, FileText } from "lucide-react";

const Sidebar = ({ isOpen, onClose }) => {
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

  return (
    <>
      {/* Overlay */}
      {isOpen && (
        <div
          className="fixed inset-0 bg-black/50 z-40 lg:hidden"
          onClick={onClose}
        />
      )}

      {/* Sidebar */}
      <aside
        className={`
          fixed lg:static inset-y-0 left-0 z-50
          w-80 bg-slate-800/50 backdrop-blur-sm border-r border-slate-700
          transform transition-transform duration-300 ease-in-out
          ${isOpen ? "translate-x-0" : "-translate-x-full lg:translate-x-0"}
          flex flex-col
        `}
      >
        {/* Header */}
        <div className="p-4 border-b border-slate-700 flex items-center justify-between">
          <h2 className="text-lg font-semibold text-white flex items-center gap-2">
            <BookOpen className="w-5 h-5" />
            Câu hỏi mẫu
          </h2>
          <button
            onClick={onClose}
            className="lg:hidden p-2 hover:bg-slate-700 rounded-lg transition-colors"
          >
            <X className="w-5 h-5 text-slate-400" />
          </button>
        </div>

        {/* Examples */}
        <div className="flex-1 overflow-y-auto p-4 space-y-3">
          {examples.map((example, idx) => (
            <button
              key={idx}
              className="w-full text-left p-3 rounded-lg bg-slate-700/50 hover:bg-slate-700 
                       transition-all duration-200 border border-slate-600/50 hover:border-cyan-500/50
                       group"
              onClick={() => {
                // This will be handled by parent component
                const input = document.querySelector("textarea");
                if (input) {
                  input.value = example.text;
                  input.focus();
                }
                onClose();
              }}
            >
              <div className="text-xs text-slate-400 mb-1">
                {example.category}
              </div>
              <div className="text-sm text-slate-200 group-hover:text-cyan-300 transition-colors">
                {example.text}
              </div>
            </button>
          ))}
        </div>

        {/* Footer */}
        <div className="p-4 border-t border-slate-700 space-y-2">
          <a
            href="http://localhost:8000/docs"
            target="_blank"
            rel="noopener noreferrer"
            className="flex items-center gap-2 text-sm text-slate-400 hover:text-cyan-400 transition-colors"
          >
            <FileText className="w-4 h-4" />
            API Documentation
          </a>
          <a
            href="https://vnstocks.com/docs/vnstock"
            target="_blank"
            rel="noopener noreferrer"
            className="flex items-center gap-2 text-sm text-slate-400 hover:text-cyan-400 transition-colors"
          >
            <BookOpen className="w-4 h-4" />
            VnStock Docs
          </a>
        </div>
      </aside>
    </>
  );
};

export default Sidebar;
