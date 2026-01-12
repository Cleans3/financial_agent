import { X, Plus, Trash2, FileText, ChevronDown, MessageCircle, ChevronLeft } from "lucide-react";
import { useState, useEffect, memo } from "react";
import { conversationService } from "../services/conversationService";
import { COLORS, STYLES } from "../theme/colors";

const Sidebar = memo(({ isOpen, onClose, onCollapseClick, user, onNewChat, onSelectConversation, onConversationDeleted, currentConversationId, refreshTrigger, isAgentThinking }) => {
  const [conversations, setConversations] = useState([]);
  const [recentExpanded, setRecentExpanded] = useState(true);

  useEffect(() => {
    loadConversations();
  }, [refreshTrigger]);

  const loadConversations = async () => {
    const allConversations = await conversationService.getConversations();
    setConversations(allConversations);
  };

  const handleNewChat = async () => {
    try {
      // Create a new empty conversation
      const newConv = await conversationService.createConversation('New Conversation');
      onSelectConversation?.(newConv.id);
      await loadConversations();
      onClose?.();
    } catch (error) {
      console.error('Error creating new conversation:', error);
    }
  };

  const handleSelectConversation = (id) => {
    onSelectConversation?.(id);
    onClose?.();
  };

  const handleDeleteConversation = async (id, e) => {
    e.stopPropagation();
    try {
      const deletedIndex = recentConversations.findIndex(c => c.id === id);
      let nearestId = null;
      
      // Find nearest conversation after deletion
      if (deletedIndex !== -1) {
        if (deletedIndex < recentConversations.length - 1) {
          // If there's a conversation after this one, use it
          nearestId = recentConversations[deletedIndex + 1].id;
        } else if (deletedIndex > 0) {
          // Otherwise use the one before
          nearestId = recentConversations[deletedIndex - 1].id;
        }
      }
      
      // Remove from UI immediately (optimistic update)
      const updatedConversations = conversations.filter(c => c.id !== id);
      setConversations(updatedConversations);
      
      // Delete from server
      await conversationService.deleteConversation(id);
      
      // If we're in the deleted conversation, switch to another
      if (currentConversationId === id) {
        if (nearestId) {
          onSelectConversation?.(nearestId);
        } else {
          onConversationDeleted?.();
        }
      }
    } catch (error) {
      console.error('Error deleting conversation:', error);
      // Reload conversations on error to restore UI state
      await loadConversations();
    }
  };

  const handleToggleStar = async (id, e) => {
    // Star feature disabled - not implemented in backend
  };

  const starredConversations = [];
  const recentConversations = conversations.filter(c => !c.starred).slice(0, 10);

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
          w-80 bg-gradient-to-b from-slate-800 to-slate-900 backdrop-blur-sm border-r border-slate-700
          transform transition-transform duration-300 ease-in-out
          ${isOpen ? "translate-x-0" : "-translate-x-full lg:translate-x-0"}
          flex flex-col
        `}
      >
        {/* Close Button for Mobile */}
        <div className="p-4 flex justify-between items-center">
          <h2 className="text-lg font-semibold text-white">Conversations</h2>
          <div className="flex gap-2">
            <button
              onClick={onClose}
              className="lg:hidden p-2 hover:bg-slate-700 rounded-lg transition-colors"
              title="Close sidebar"
            >
              <X className="w-5 h-5 text-slate-400" />
            </button>
            <button
              onClick={onCollapseClick}
              className="hidden lg:block p-2 hover:bg-slate-700 rounded-lg transition-colors"
              title="Collapse sidebar"
            >
              <ChevronLeft className="w-5 h-5 text-slate-400" />
            </button>
          </div>
        </div>

        {/* New Chat Button */}
        <div className="px-4 py-2">
          <button
            onClick={handleNewChat}
            disabled={isAgentThinking}
            className={`w-full flex items-center justify-center gap-2 px-4 py-2 rounded-lg font-medium transition-all ${STYLES.button.primary} ${
              isAgentThinking ? 'opacity-50 cursor-not-allowed' : ''
            }`}
            title={isAgentThinking ? "Please wait for the current response to complete" : ""}
          >
            <Plus className="w-4 h-4" />
            Cuộc trò chuyện mới
          </button>
        </div>

        {/* Scrollable Content */}
        <div className="flex-1 overflow-y-auto px-4 py-4 space-y-6">
          
          {/* Recent Conversations */}
          {recentConversations.length > 0 && (
            <div>
              <button
                onClick={() => setRecentExpanded(!recentExpanded)}
                className="flex items-center gap-2 px-3 py-2 w-full text-sm font-semibold text-slate-300 hover:text-white transition-colors"
              >
                <ChevronDown className={`w-4 h-4 transition-transform ${recentExpanded ? "" : "-rotate-90"}`} />
                Gần đây
              </button>
              {recentExpanded && (
                <div className="space-y-2 mt-2">
                  {recentConversations.map(conv => (
                    <div
                      key={conv.id}
                      onClick={() => !isAgentThinking && handleSelectConversation(conv.id)}
                      className={`p-3 rounded-lg cursor-pointer transition-all ${
                        currentConversationId === conv.id
                          ? `bg-cyan-500/20 text-white`
                          : `bg-slate-700/30 text-slate-300 hover:bg-slate-700/50`
                      } ${isAgentThinking && currentConversationId !== conv.id ? 'opacity-50 cursor-not-allowed' : ''}`}
                      title={isAgentThinking && currentConversationId !== conv.id ? "Read-only: Agent is thinking in another conversation" : ""}
                    >
                      <div className="flex items-start gap-2">
                        <MessageCircle className="w-4 h-4 mt-1 flex-shrink-0" />
                        <div className="flex-1 min-w-0">
                          <div className="font-sm truncate">{conv.title}</div>
                          <div className="text-xs text-slate-500">{new Date(conv.createdAt).toLocaleDateString()}</div>
                        </div>
                        <div className="flex gap-1 flex-shrink-0">
                          <button
                            onClick={(e) => !isAgentThinking && handleDeleteConversation(conv.id, e)}
                            disabled={isAgentThinking}
                            className={`p-1 hover:bg-red-500/20 rounded transition-colors ${isAgentThinking ? 'opacity-50 cursor-not-allowed' : ''}`}
                          >
                            <Trash2 className="w-3 h-3 text-red-400" />
                          </button>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="p-4 space-y-3">
          {/* User Info Card */}
          {user && (
            <div className="bg-slate-700/30 rounded-lg p-3">
              <div className="text-xs text-slate-400 mb-1">Đăng nhập với tư cách</div>
              <div className="text-sm font-semibold text-slate-200">{user.username}</div>
            </div>
          )}

          {/* Links */}
          <div className="space-y-2">
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
              href="https://vnstock.site/"
              target="_blank"
              rel="noopener noreferrer"
              className="flex items-center gap-2 text-sm text-slate-400 hover:text-cyan-400 transition-colors"
            >
              <FileText className="w-4 h-4" />
              VNStock API Docs
            </a>
          </div>
        </div>
      </aside>
    </>
  );
});

Sidebar.displayName = "Sidebar";
export default Sidebar;
