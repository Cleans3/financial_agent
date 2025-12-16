import React, { useState } from 'react'
import axios from 'axios'
import { Lock, Mail, Loader } from 'lucide-react'
import { COLORS, STYLES } from '../theme/colors'

const LoginPage = ({ onLoginSuccess }) => {
  const [username, setUsername] = useState('')
  const [password, setPassword] = useState('')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  const handleSubmit = async (e) => {
    e.preventDefault()
    setLoading(true)
    setError(null)

    try {
      const response = await axios.post('/api/auth/login', {
        username,
        password
      })

      if (response.data.access_token) {
        localStorage.setItem('token', response.data.access_token)
        axios.defaults.headers.common['Authorization'] = `Bearer ${response.data.access_token}`
        
        // Fetch full user info including is_admin flag
        try {
          console.log('Fetching user info from /api/auth/me')
          const userInfoResponse = await axios.get('/api/auth/me')
          console.log('User info response:', userInfoResponse.data)
          const userData = userInfoResponse.data
          console.log('User info fetched successfully:', userData)
          localStorage.setItem('user', JSON.stringify(userData))
          onLoginSuccess(userData)
        } catch (infoErr) {
          console.error('Error fetching user info:', infoErr.message, infoErr.response?.data)
          // Fallback to basic user data with default is_admin
          const userData = { username, email: username, is_admin: false, id: 'temp' }
          console.log('Using fallback user data:', userData)
          localStorage.setItem('user', JSON.stringify(userData))
          onLoginSuccess(userData)
        }
      }
    } catch (err) {
      setError(err.response?.data?.detail || 'Login failed. Please check your credentials.')
      console.error('Login error:', err)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900 flex items-center justify-center px-4">
      {/* Background decoration */}
      <div className="absolute inset-0 overflow-hidden">
        <div className="absolute top-0 right-0 w-96 h-96 bg-sky-500/10 rounded-full blur-3xl opacity-50"></div>
        <div className="absolute bottom-0 left-0 w-96 h-96 bg-cyan-500/10 rounded-full blur-3xl opacity-50"></div>
      </div>

      <div className="w-full max-w-md relative z-10">
        <div className="bg-slate-800/80 backdrop-blur-sm border border-slate-700 rounded-2xl shadow-2xl p-8">
          {/* Logo & Title */}
          <div className="text-center mb-8">
            <div className="inline-flex items-center justify-center w-12 h-12 bg-gradient-to-br from-sky-500 to-cyan-500 rounded-lg mb-4">
              <Lock className="w-6 h-6 text-white" />
            </div>
            <h1 className="text-3xl font-bold text-white mb-2">Financial Agent</h1>
            <p className="text-slate-400">AI-Powered Stock Market Assistant</p>
          </div>

          {/* Error Alert */}
          {error && (
            <div className="mb-6 p-4 bg-red-500/10 border border-red-500/50 rounded-lg">
              <p className="text-red-400 text-sm">{error}</p>
            </div>
          )}

          {/* Login Form */}
          <form onSubmit={handleSubmit} className="space-y-4">
            {/* Username Field */}
            <div>
              <label className="block text-sm font-medium text-slate-300 mb-2">
                Username
              </label>
              <div className="relative">
                <Mail className="absolute left-3 top-1/2 -translate-y-1/2 w-5 h-5 text-slate-500" />
                <input
                  type="text"
                  value={username}
                  onChange={(e) => setUsername(e.target.value)}
                  placeholder="admin"
                  className={`${STYLES.input} pl-10`}
                  required
                  disabled={loading}
                />
              </div>
            </div>

            {/* Password Field */}
            <div>
              <label className="block text-sm font-medium text-slate-300 mb-2">
                Password
              </label>
              <div className="relative">
                <Lock className="absolute left-3 top-1/2 -translate-y-1/2 w-5 h-5 text-slate-500" />
                <input
                  type="password"
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  placeholder="••••••••"
                  className={`${STYLES.input} pl-10`}
                  required
                  disabled={loading}
                />
              </div>
            </div>

            {/* Submit Button */}
            <button
              type="submit"
              disabled={loading}
              className={`${STYLES.button.primary} w-full mt-6 flex items-center justify-center gap-2 disabled:opacity-50 disabled:cursor-not-allowed`}
            >
              {loading ? (
                <>
                  <Loader className="w-4 h-4 animate-spin" />
                  Logging in...
                </>
              ) : (
                'Log In'
              )}
            </button>
          </form>

          {/* Demo Credentials */}
          <div className="mt-8 pt-8 border-t border-slate-700">
            <p className="text-slate-400 text-sm text-center mb-4">Demo Credentials:</p>
            <div className="bg-slate-700/50 p-4 rounded-lg space-y-2 border border-slate-600">
              <div>
                <p className="text-xs text-slate-500 uppercase tracking-wider">Username</p>
                <p className="text-slate-300 font-mono">admin</p>
              </div>
              <div>
                <p className="text-xs text-slate-500 uppercase tracking-wider">Password</p>
                <p className="text-slate-300 font-mono">admin123</p>
              </div>
            </div>
          </div>

          {/* Features List */}
          <div className="mt-8 pt-8 border-t border-slate-700">
            <p className="text-slate-400 text-xs text-center mb-4 uppercase tracking-wider">Features</p>
            <div className="grid grid-cols-3 gap-3 text-center text-xs">
              <div className="p-3 bg-sky-500/10 rounded-lg border border-sky-500/30">
                <p className="text-sky-400 font-semibold">📊 Analytics</p>
              </div>
              <div className="p-3 bg-cyan-500/10 rounded-lg border border-cyan-500/30">
                <p className="text-cyan-400 font-semibold">🤖 AI Chat</p>
              </div>
              <div className="p-3 bg-emerald-500/10 rounded-lg border border-emerald-500/30">
                <p className="text-emerald-400 font-semibold">📈 Real-time</p>
              </div>
            </div>
          </div>
        </div>

        {/* Footer */}
        <p className="text-center text-slate-500 text-xs mt-6">
          © 2025 Financial Agent. All rights reserved.
        </p>
      </div>
    </div>
  )
}

export default LoginPage
