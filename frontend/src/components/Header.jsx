import { Menu, TrendingUp, LogOut, User, ChevronDown, Settings } from "lucide-react";
import { useState, useEffect } from "react";

const Header = ({ onMenuClick, user, onLogout, onAdminClick }) => {
  const [showUserMenu, setShowUserMenu] = useState(false);

  useEffect(() => {
    if (user) {
      console.log('Header user object:', user)
      console.log('Is admin?', user.is_admin)
    }
  }, [user])

  return (
    <header className="bg-slate-800/50 backdrop-blur-sm px-4 py-3">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <button
            onClick={onMenuClick}
            className="lg:hidden p-2 hover:bg-slate-700 rounded-lg transition-colors"
          >
            <Menu className="w-5 h-5 text-slate-300" />
          </button>

          <div className="flex items-center gap-3">
            <div className="bg-gradient-to-br from-emerald-400 to-cyan-500 p-2 rounded-lg">
              <TrendingUp className="w-6 h-6 text-white" />
            </div>
            <div>
              <h1 className="text-xl font-bold text-white">Financial Agent</h1>
              <p className="text-xs text-slate-400">
                Vietnamese Stock Market Assistant
              </p>
            </div>
          </div>
        </div>

        <div className="flex items-center gap-4">
          <div className="hidden sm:flex items-center gap-2 bg-emerald-500/10 text-emerald-400 px-3 py-1.5 rounded-full text-sm">
            <div className="w-2 h-2 bg-emerald-400 rounded-full animate-pulse"></div>
            <span>Online</span>
          </div>

          {/* User Menu */}
          {user && (
            <div className="relative">
              <button
                onClick={() => setShowUserMenu(!showUserMenu)}
                className="flex items-center gap-2 px-3 py-1.5 rounded-lg hover:bg-slate-700 transition-colors"
              >
                <div className="w-8 h-8 bg-sky-600 rounded-full flex items-center justify-center text-white text-sm font-semibold">
                  {user.username ? user.username.charAt(0).toUpperCase() : 'U'}
                </div>
                <div className="hidden sm:block text-left">
                  <p className="text-sm font-medium text-white">{user.username || user.email}</p>
                </div>
                <ChevronDown className="w-4 h-4 text-slate-400" />
              </button>

              {/* Dropdown Menu */}
              {showUserMenu && (
                <div className="absolute right-0 mt-2 w-48 bg-slate-800 border border-slate-700 rounded-lg shadow-lg z-50">
                  <div className="px-4 py-3 border-b border-slate-700">
                    <p className="text-sm font-medium text-white">{user.username}</p>
                    <p className="text-xs text-slate-400">{user.email}</p>
                  </div>
                  {user.is_admin && (
                    <button
                      onClick={() => {
                        onAdminClick && onAdminClick();
                        setShowUserMenu(false);
                      }}
                      className="w-full flex items-center gap-2 px-4 py-2 text-sm text-sky-400 hover:bg-sky-500/10 transition-colors border-b border-slate-700"
                    >
                      <Settings className="w-4 h-4" />
                      Admin Dashboard
                    </button>
                  )}
                  <button
                    onClick={() => {
                      onLogout();
                      setShowUserMenu(false);
                    }}
                    className="w-full flex items-center gap-2 px-4 py-2 text-sm text-red-400 hover:bg-red-500/10 transition-colors"
                  >
                    <LogOut className="w-4 h-4" />
                    Logout
                  </button>
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </header>
  );
};

export default Header;
