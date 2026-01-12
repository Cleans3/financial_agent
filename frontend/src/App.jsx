import { useState, useEffect } from "react";
import ChatInterface from "./components/ChatInterface";
import Header from "./components/Header";
import Sidebar from "./components/Sidebar";
import RightSidebar from "./components/RightSidebar";
import LoginPage from "./components/LoginPage";
import AdminDashboard from "./components/AdminDashboard";
import axios from "axios";
import { conversationService } from "./services/conversationService";

function App() {
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [rightSidebarOpen, setRightSidebarOpen] = useState(true);
  const [isAuthenticated, setIsAuthenticated] = useState(false);
  const [user, setUser] = useState(null);
  const [currentConversationId, setCurrentConversationId] = useState(null);
  const [isLoading, setIsLoading] = useState(true);
  const [refreshSidebar, setRefreshSidebar] = useState(0);
  const [showAdminDashboard, setShowAdminDashboard] = useState(false);
  const [isAgentThinking, setIsAgentThinking] = useState(false); // Global agent thinking state

  // Check if user is already logged in on app load
  useEffect(() => {
    const token = localStorage.getItem('token');
    const userData = localStorage.getItem('user');
    
    if (token && userData) {
      try {
        const parsedUser = JSON.parse(userData);
        setUser(parsedUser);
        setIsAuthenticated(true);
        // Set default axios header
        axios.defaults.headers.common['Authorization'] = `Bearer ${token}`;
      } catch (error) {
        console.error('Error parsing user data:', error);
        localStorage.removeItem('token');
        localStorage.removeItem('user');
      }
    }
    setIsLoading(false);
  }, []);

  const handleLoginSuccess = (userData) => {
    const token = localStorage.getItem('token');
    setUser(userData);
    setIsAuthenticated(true);
    if (token) {
      axios.defaults.headers.common['Authorization'] = `Bearer ${token}`;
    }
    // Create initial empty conversation on login
    setCurrentConversationId(null);
    setRefreshSidebar(prev => prev + 1);
  };

  const handleLogout = () => {
    localStorage.removeItem('token');
    localStorage.removeItem('user');
    delete axios.defaults.headers.common['Authorization'];
    setIsAuthenticated(false);
    setUser(null);
    setCurrentConversationId(null);
  };

  const handleConversationDeleted = () => {
    setCurrentConversationId(null);
    setRefreshSidebar(prev => prev + 1);
  };

  const handleConversationChange = (id) => {
    setCurrentConversationId(id);
    setRefreshSidebar(prev => prev + 1);
  };

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-screen bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900">
        <div className="text-white text-center">
          <div className="w-12 h-12 border-4 border-slate-700 border-t-sky-500 rounded-full animate-spin mx-auto mb-4"></div>
          <p>Loading Financial Agent...</p>
        </div>
      </div>
    );
  }

  if (!isAuthenticated) {
    return <LoginPage onLoginSuccess={handleLoginSuccess} />;
  }

  // Show admin dashboard if admin mode is active
  if (showAdminDashboard && user?.is_admin) {
    return (
      <AdminDashboard 
        user={user} 
        onLogout={handleLogout}
        onClose={() => setShowAdminDashboard(false)}
      />
    );
  }

  return (
    <div className="flex h-screen bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900">
      {/* Sidebar - Conversation History */}
      <Sidebar 
        isOpen={sidebarOpen} 
        onClose={() => setSidebarOpen(false)}
        onCollapseClick={() => setSidebarOpen(false)}
        user={user}
        onNewChat={() => setCurrentConversationId(null)}
        onSelectConversation={handleConversationChange}
        onConversationDeleted={handleConversationDeleted}
        currentConversationId={currentConversationId}
        refreshTrigger={refreshSidebar}
        isAgentThinking={isAgentThinking}
      />

      {/* Main Content */}
      <div className="flex-1 flex flex-col overflow-hidden">
        <Header 
          onMenuClick={() => setSidebarOpen(!sidebarOpen)}
          user={user}
          onLogout={handleLogout}
          onAdminClick={() => setShowAdminDashboard(true)}
        />
        <ChatInterface 
          conversationId={currentConversationId}
          onConversationChange={handleConversationChange}
          onSidebarRefresh={() => setRefreshSidebar(prev => prev + 1)}
          onAgentThinkingChange={setIsAgentThinking}
        />
      </div>

      {/* Right Sidebar - Example Questions */}
      <RightSidebar isOpen={rightSidebarOpen} onToggle={() => setRightSidebarOpen(!rightSidebarOpen)} />
    </div>
  );
}

export default App;
