import { useState, useRef, useEffect } from "react";
import { Send, Loader2, User, Bot, Plus, X, File, Image, FileText, FileSpreadsheet, FileArchive, BookOpen } from "lucide-react";
import axios from "axios";
import MessageBubble from "./MessageBubble";
import DocumentPanel from "./DocumentPanel";
import { conversationService } from "../services/conversationService";

const ChatInterface = ({ conversationId, onConversationChange, onSidebarRefresh }) => {
  const [messages, setMessages] = useState([
    {
      role: "assistant",
      content:
        "Xin chào! Tôi là Financial Agent, trợ lý AI chuyên về thị trường chứng khoán Việt Nam. Tôi có thể giúp bạn:\n\n• Tra cứu thông tin công ty\n• Xem dữ liệu giá lịch sử (OHLCV)\n• Phân tích cổ đông và ban lãnh đạo\n• Tính toán chỉ số kỹ thuật (SMA, RSI)\n• Xem sự kiện công ty\n\nHãy thử hỏi tôi về bất kỳ mã chứng khoán nào!",
    },
  ]);
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [uploadedFiles, setUploadedFiles] = useState([]);
  const [fileContext, setFileContext] = useState(null);
  const [isDragOver, setIsDragOver] = useState(false);
  const [activeConversationId, setActiveConversationId] = useState(null);
  const [pendingUserMessage, setPendingUserMessage] = useState(null);
  const [useRAG, setUseRAG] = useState(true);
  const [showDocumentPanel, setShowDocumentPanel] = useState(false);
  const messagesEndRef = useRef(null);
  const fileInputRef = useRef(null);
  const textareaRef = useRef(null);

  // Reset messages when conversation changes
  useEffect(() => {
    const loadConversation = async () => {
      // Don't switch conversations if agent is thinking in another conversation
      if (isLoading && activeConversationId && activeConversationId !== conversationId) {
        return; // Stay in the active conversation
      }
      
      if (conversationId === null) {
        setMessages([
          {
            role: "assistant",
            content:
              "Xin chào! Tôi là Financial Agent, trợ lý AI chuyên về thị trường chứng khoán Việt Nam. Tôi có thể giúp bạn:\n\n• Tra cứu thông tin công ty\n• Xem dữ liệu giá lịch sử (OHLCV)\n• Phân tích cổ đông và ban lãnh đạo\n• Tính toán chỉ số kỹ thuật (SMA, RSI)\n• Xem sự kiện công ty\n\nHãy thử hỏi tôi về bất kỳ mã chứng khoán nào!",
          },
        ]);
        setInput("");
        setUploadedFiles([]);
        setFileContext(null);
        setPendingUserMessage(null);
      } else {
        // Load conversation history from API
        const conversation = await conversationService.getConversation(conversationId);
        let loadedMessages = [
          {
            role: "assistant",
            content:
              "Xin chào! Tôi là Financial Agent, trợ lý AI chuyên về thị trường chứng khoán Việt Nam. Tôi có thể giúp bạn:\n\n• Tra cứu thông tin công ty\n• Xem dữ liệu giá lịch sử (OHLCV)\n• Phân tích cổ đông và ban lãnh đạo\n• Tính toán chỉ số kỹ thuật (SMA, RSI)\n• Xem sự kiện công ty\n\nHãy thử hỏi tôi về bất kỳ mã chứng khoán nào!",
          },
        ];

        if (conversation && conversation.messages && conversation.messages.length > 0) {
          loadedMessages = conversation.messages.map(msg => ({
            role: msg.role,
            content: msg.content
          }));
        }

        // If there's a pending user message, add it to the loaded messages
        if (pendingUserMessage) {
          loadedMessages.push({
            role: "user",
            content: pendingUserMessage
          });
          setPendingUserMessage(null);
        }

        setMessages(loadedMessages);
      }
    };
    loadConversation();
  }, [conversationId]);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  // Save messages to conversation storage
  useEffect(() => {
    if (conversationId && messages.length > 0) {
      conversationService.saveMessages(conversationId, messages);
      // Don't refresh sidebar on every message save - only on conversation creation/deletion
    }
  }, [messages, conversationId]);

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!input.trim() && uploadedFiles.length === 0 || isLoading) return;

    const userMessage = input.trim();
    setInput("");
    const filesToProcess = [...uploadedFiles];
    setUploadedFiles([]);

    // Create a new conversation if none exists
    let currentConvId = conversationId;
    if (!currentConvId) {
      try {
        const newConv = await conversationService.createConversation(userMessage || "Phân tích file");
        currentConvId = newConv.id;
        // Set pending message before switching conversation
        setPendingUserMessage(userMessage || "Phân tích file");
        onConversationChange?.(currentConvId);
        return; // Exit early - the conversation change will trigger the message to be added via useEffect
      } catch (error) {
        console.error('Error creating conversation:', error);
        setMessages((prev) => [...prev, { 
          role: "assistant", 
          content: `Lỗi: Không thể tạo cuộc trò chuyện. ${error.message}` 
        }]);
        return;
      }
    }

    // Mark this conversation as the active one (agent is thinking here)
    setActiveConversationId(currentConvId);

    // Add user message (only for existing conversations)
    setMessages((prev) => [...prev, { role: "user", content: userMessage || "Phân tích file" }]);
    setIsLoading(true);

    try {
      // If files are uploaded, process them
      if (filesToProcess.length > 0) {
        for (const file of filesToProcess) {
          try {
            // Convert data URL to blob
            const response = await fetch(file.data);
            const blob = await response.blob();

            // Create FormData
            const formData = new FormData();
            formData.append("file", blob, file.name);
            if (userMessage) {
              formData.append("question", userMessage);
            }

            // Upload and analyze
            const analysisResponse = await axios.post("/api/upload", formData, {
              headers: {
                "Content-Type": "multipart/form-data",
              },
            });

            // Display analysis result
            const analysisMessage = analysisResponse.data.analysis || analysisResponse.data.message;
            setMessages((prev) => [
              ...prev,
              {
                role: "assistant",
                content: `📄 **${file.name}**\n\n${analysisMessage}`,
              },
            ]);
            
            // Lưu file context để dùng cho câu hỏi tiếp theo
            setFileContext({
              fileName: file.name,
              analysis: analysisMessage
            });
          } catch (error) {
            console.error("Error analyzing file:", error);
            setMessages((prev) => [
              ...prev,
              {
                role: "assistant",
                content: `❌ Lỗi phân tích file "${file.name}": ${
                  error.response?.data?.detail || error.message
                }`,
              },
            ]);
          }
        }
      } 
      // Nếu CHỈ có text message (không upload file), gửi qua chat API
      // Hoặc nếu user có file context từ lần upload trước, gửi với context đó
      else if (userMessage) {
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
      }
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
      setActiveConversationId(null); // Clear active conversation when done thinking
      // Only refresh sidebar on conversation creation or cleanup
      // Don't refresh on every message to avoid excessive API calls
      try {
        const deletedCount = await conversationService.deleteEmptyConversations();
        if (deletedCount > 0) {
          onSidebarRefresh?.();
        }
      } catch (error) {
        console.warn('Error deleting empty conversations:', error);
      }
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSubmit(e);
    }
  };

  const handleFileSelect = (e) => {
    const files = Array.from(e.target.files || []);
    files.forEach((file) => {
      const reader = new FileReader();
      reader.onload = (event) => {
        setUploadedFiles((prev) => [
          ...prev,
          {
            id: Date.now() + Math.random(),
            name: file.name,
            size: file.size,
            type: file.type,
            data: event.target.result,
          },
        ]);
      };
      reader.readAsDataURL(file);
    });
    // Reset input
    if (fileInputRef.current) {
      fileInputRef.current.value = "";
    }
  };

  const removeFile = (fileId) => {
    setUploadedFiles((prev) => prev.filter((f) => f.id !== fileId));
  };

  const getFileIcon = (type) => {
    if (type.startsWith("image/")) {
      return (
        <div className="text-emerald-400">
          <Image className="w-6 h-6" />
        </div>
      );
    }
    if (type.includes("pdf")) {
      return (
        <div className="text-red-400">
          <FileText className="w-6 h-6" />
        </div>
      );
    }
    if (type.includes("word") || type.includes("document")) {
      return (
        <div className="text-blue-400">
          <FileText className="w-6 h-6" />
        </div>
      );
    }
    if (
      type.includes("sheet") ||
      type.includes("excel") ||
      type.includes("spreadsheet")
    ) {
      return (
        <div className="text-green-400">
          <FileSpreadsheet className="w-6 h-6" />
        </div>
      );
    }
    if (type.includes("zip") || type.includes("rar") || type.includes("7z")) {
      return (
        <div className="text-yellow-400">
          <FileArchive className="w-6 h-6" />
        </div>
      );
    }
    return (
      <div className="text-slate-400">
        <File className="w-6 h-6" />
      </div>
    );
  };

  const formatFileSize = (bytes) => {
    if (bytes === 0) return "0 Bytes";
    const k = 1024;
    const sizes = ["Bytes", "KB", "MB"];
    const i = Math.floor(Math.log(bytes, k));
    return Math.round((bytes / Math.pow(k, i)) * 100) / 100 + " " + sizes[i];
  };

  const truncateFileName = (name, maxLength = 20) => {
    if (name.length <= maxLength) return name;
    const ext = name.split(".").pop();
    const nameWithoutExt = name.substring(0, name.lastIndexOf("."));
    const truncated = nameWithoutExt.substring(0, maxLength - ext.length - 4);
    return truncated + "..." + ext;
  };

  const handlePaste = (e) => {
    const items = e.clipboardData?.items;
    if (!items) return;

    let hasFiles = false;

    for (let i = 0; i < items.length; i++) {
      if (items[i].kind === "file") {
        const file = items[i].getAsFile();
        // Only treat as file if it's an actual image or document
        if (file && (file.type.startsWith("image/") || file.type.includes("pdf") || file.type.includes("sheet") || file.type.includes("document"))) {
          hasFiles = true;
          const reader = new FileReader();
          reader.onload = (event) => {
            setUploadedFiles((prev) => [
              ...prev,
              {
                id: Date.now() + Math.random(),
                name: file.name,
                size: file.size,
                type: file.type,
                data: event.target.result,
              },
            ]);
          };
          reader.readAsDataURL(file);
        }
      }
    }

    // If we processed files, prevent default paste behavior
    if (hasFiles) {
      e.preventDefault();
    }
    // Otherwise allow normal text paste
  };

  const handleDragOver = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragOver(true);
  };

  const handleDragLeave = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragOver(false);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragOver(false);
    
    const files = e.dataTransfer?.files;
    if (!files) return;

    // Process dropped files
    for (let i = 0; i < files.length; i++) {
      const file = files[i];
      const reader = new FileReader();
      reader.onload = (event) => {
        setUploadedFiles((prev) => [
          ...prev,
          {
            id: Date.now() + Math.random() + i,
            name: file.name,
            size: file.size,
            type: file.type,
            data: event.target.result,
          },
        ]);
      };
      reader.readAsDataURL(file);
    }
  };

  return (
    <div 
      className="flex-1 flex flex-col overflow-hidden"
      onDragOver={handleDragOver}
      onDragLeave={handleDragLeave}
      onDrop={handleDrop}
    >
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
        <form 
          onSubmit={handleSubmit}
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onDrop={handleDrop}
          className={`max-w-4xl mx-auto space-y-3 transition-all duration-200 rounded-lg p-3 ${
            isDragOver 
              ? "bg-slate-700/80 border-2 border-cyan-500/50" 
              : "bg-transparent"
          }`}
        >
          {/* File Preview Grid */}
          {uploadedFiles.length > 0 && (
            <div className="space-y-2 pointer-events-none">
              <div className="flex items-center justify-between">
                <label className="text-sm font-medium text-slate-300">
                  Tệp đã chọn ({uploadedFiles.length})
                </label>
              </div>
              <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 gap-3 pointer-events-auto">
              {uploadedFiles.map((file) => (
                <div
                  key={file.id}
                  className="relative group bg-slate-700/50 rounded-lg p-2 border border-slate-600 hover:border-cyan-500/50 transition-all"
                >
                  {/* File Preview Thumbnail */}
                  {file.type.startsWith("image/") ? (
                    <div className="w-full h-20 rounded mb-2 overflow-hidden bg-slate-800">
                      <img
                        src={file.data}
                        alt={file.name}
                        className="w-full h-full object-cover"
                      />
                    </div>
                  ) : (
                    <div className="w-full h-20 rounded mb-2 bg-slate-800 flex items-center justify-center text-slate-400">
                      {getFileIcon(file.type)}
                    </div>
                  )}

                  {/* File Info */}
                  <div className="space-y-1">
                    <p
                      title={file.name}
                      className="text-xs text-slate-300 truncate font-medium"
                    >
                      {truncateFileName(file.name, 15)}
                    </p>
                    <p className="text-xs text-slate-500">
                      {formatFileSize(file.size)}
                    </p>
                  </div>

                  {/* Remove Button */}
                  <button
                    type="button"
                    onClick={() => removeFile(file.id)}
                    className="absolute top-1 right-1 opacity-0 group-hover:opacity-100 transition-opacity
                             bg-red-500/80 hover:bg-red-600 rounded-full p-1"
                  >
                    <X className="w-3 h-3 text-white" />
                  </button>
                </div>
              ))}
              </div>
            </div>
          )}

          {/* Input Container - Flex with no wrap */}
          <div className="flex items-stretch gap-3">
            {/* File Upload Button - Circle */}
            <button
              type="button"
              onClick={() => fileInputRef.current?.click()}
              className="bg-slate-700 hover:bg-slate-600 text-white rounded-full
                       transition-all duration-200 transform hover:scale-110 active:scale-95
                       flex-shrink-0 flex items-center justify-center
                       w-12 h-12 min-w-12 min-h-12"
              title="Thêm file (hoặc Ctrl+V để paste)"
            >
              <Plus className="w-6 h-6" />
            </button>

            {/* Hidden File Input */}
            <input
              ref={fileInputRef}
              type="file"
              onChange={handleFileSelect}
              multiple
              accept="image/*,.pdf,.xlsx,.xls,.doc,.docx,.txt,.csv"
              className="hidden"
              title="Hỗ trợ: PNG, JPG, PDF, Excel, Word, CSV"
            />

            {/* Text Input */}
            <div className="flex-1 relative flex items-stretch">
              <textarea
                ref={textareaRef}
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={handleKeyDown}
                onPaste={handlePaste}
                placeholder="Hỏi về bất kỳ mã chứng khoán nào... (VD: Thông tin VNM, Giá VCB 3 tháng)"
                className="flex-1 bg-slate-800 text-white placeholder-slate-500 
                         rounded-2xl px-4 py-3 pr-12
                         border border-slate-600 focus:border-cyan-500 focus:ring-2 focus:ring-cyan-500/20
                         outline-none resize-none transition-all
                         max-h-32 overflow-hidden"
                rows={1}
                disabled={isLoading}
                style={{
                  minHeight: "48px",
                  height: "auto",
                }}
                onInput={(e) => {
                  e.target.style.height = "auto";
                  const maxHeight = 128; // max-h-32 = 8rem = 128px
                  if (e.target.scrollHeight > maxHeight) {
                    e.target.style.height = maxHeight + "px";
                    e.target.style.overflowY = "auto";
                  } else {
                    e.target.style.height = e.target.scrollHeight + "px";
                    e.target.style.overflowY = "hidden";
                  }
                }}
              />
            </div>

            {/* Send Button - Circle */}
            <button
              type="submit"
              disabled={(!input.trim() && uploadedFiles.length === 0) || isLoading}
              className="bg-gradient-to-r from-emerald-500 to-cyan-500 hover:from-emerald-600 hover:to-cyan-600
                       text-white rounded-full
                       disabled:opacity-50 disabled:cursor-not-allowed
                       transition-all duration-200 transform hover:scale-105 active:scale-95
                       shadow-lg hover:shadow-cyan-500/50 flex-shrink-0 
                       w-12 h-12 min-w-12 min-h-12 flex items-center justify-center"
            >
              {isLoading ? (
                <Loader2 className="w-5 h-5 animate-spin" />
              ) : (
                <Send className="w-5 h-5" />
              )}
            </button>
          </div>

          <p className="text-xs text-slate-500 text-center">
            Nhấn Enter để gửi, Shift+Enter để xuống dòng • Ctrl+V để paste file
          </p>
        </form>

        {/* Document Panel Modal */}
        <DocumentPanel
          sessionId={conversationId}
          isOpen={showDocumentPanel}
          onClose={() => setShowDocumentPanel(false)}
          useRAG={useRAG}
          onToggleRAG={() => setUseRAG(!useRAG)}
        />
      </div>
    </div>
  );
};

export default ChatInterface;
