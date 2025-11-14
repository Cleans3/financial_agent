import { useState, useRef, useEffect } from "react";
import { Send, Loader2, User, Bot } from "lucide-react";
import axios from "axios";
import MessageBubble from "./MessageBubble";

const ChatInterface = () => {
  const [messages, setMessages] = useState([
    {
      role: "assistant",
      content:
        "Xin chào! Tôi là Financial Agent, trợ lý AI chuyên về thị trường chứng khoán Việt Nam. Tôi có thể giúp bạn:\n\n• Tra cứu thông tin công ty\n• Xem dữ liệu giá lịch sử (OHLCV)\n• Phân tích cổ đông và ban lãnh đạo\n• Tính toán chỉ số kỹ thuật (SMA, RSI)\n• Xem sự kiện công ty\n\nHãy thử hỏi tôi về bất kỳ mã chứng khoán nào!",
    },
  ]);
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const messagesEndRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!input.trim() || isLoading) return;

    const userMessage = input.trim();
    setInput("");

    // Add user message
    setMessages((prev) => [...prev, { role: "user", content: userMessage }]);
    setIsLoading(true);

    try {
      const response = await axios.post("/api/chat", {
        question: userMessage,
      });

      // Add assistant response
      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          content: response.data.answer,
        },
      ]);
    } catch (error) {
      console.error("Error:", error);
      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          content: `Xin lỗi, đã có lỗi xảy ra: ${
            error.response?.data?.detail || error.message
          }`,
        },
      ]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSubmit(e);
    }
  };

  return (
    <div className="flex-1 flex flex-col overflow-hidden">
      {/* Messages */}
      <div className="flex-1 overflow-y-auto px-4 py-6 space-y-6">
        <div className="max-w-4xl mx-auto space-y-6">
          {messages.map((message, idx) => (
            <MessageBubble key={idx} message={message} />
          ))}

          {isLoading && (
            <div className="flex items-start gap-3 animate-slide-in">
              <div
                className="w-8 h-8 rounded-full bg-gradient-to-br from-emerald-400 to-cyan-500 
                            flex items-center justify-center flex-shrink-0"
              >
                <Bot className="w-5 h-5 text-white" />
              </div>
              <div className="flex-1 bg-slate-800/50 rounded-2xl rounded-tl-none p-4 border border-slate-700">
                <div className="flex items-center gap-2 text-slate-400">
                  <Loader2 className="w-4 h-4 animate-spin" />
                  <span className="text-sm">Đang suy nghĩ...</span>
                </div>
              </div>
            </div>
          )}

          <div ref={messagesEndRef} />
        </div>
      </div>

      {/* Input */}
      <div className="border-t border-slate-700 bg-slate-800/30 backdrop-blur-sm p-4">
        <form onSubmit={handleSubmit} className="max-w-4xl mx-auto">
          <div className="relative flex items-end gap-2">
            <div className="flex-1 relative">
              <textarea
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={handleKeyDown}
                placeholder="Hỏi về bất kỳ mã chứng khoán nào... (VD: Thông tin VNM, Giá VCB 3 tháng)"
                className="w-full bg-slate-800 text-white placeholder-slate-500 
                         rounded-2xl px-4 py-3 pr-12
                         border border-slate-600 focus:border-cyan-500 focus:ring-2 focus:ring-cyan-500/20
                         outline-none resize-none transition-all
                         max-h-32"
                rows={1}
                disabled={isLoading}
                style={{
                  minHeight: "52px",
                  height: "auto",
                }}
                onInput={(e) => {
                  e.target.style.height = "auto";
                  e.target.style.height = e.target.scrollHeight + "px";
                }}
              />
            </div>

            <button
              type="submit"
              disabled={!input.trim() || isLoading}
              className="bg-gradient-to-r from-emerald-500 to-cyan-500 hover:from-emerald-600 hover:to-cyan-600
                       text-white rounded-2xl p-3
                       disabled:opacity-50 disabled:cursor-not-allowed
                       transition-all duration-200 transform hover:scale-105 active:scale-95
                       shadow-lg hover:shadow-cyan-500/50"
            >
              {isLoading ? (
                <Loader2 className="w-5 h-5 animate-spin" />
              ) : (
                <Send className="w-5 h-5" />
              )}
            </button>
          </div>

          <p className="text-xs text-slate-500 mt-2 text-center">
            Nhấn Enter để gửi, Shift+Enter để xuống dòng
          </p>
        </form>
      </div>
    </div>
  );
};

export default ChatInterface;
