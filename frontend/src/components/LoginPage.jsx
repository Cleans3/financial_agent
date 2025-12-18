import React, { useState } from 'react'
import axios from 'axios'
import { Lock, Mail, Loader, AlertCircle, CheckCircle } from 'lucide-react'
import { COLORS, STYLES } from '../theme/colors'

const LoginPage = ({ onLoginSuccess }) => {
  const [isRegistering, setIsRegistering] = useState(false)
  const [username, setUsername] = useState('')
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [confirmPassword, setConfirmPassword] = useState('')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [success, setSuccess] = useState(null)
  const [passwordStrength, setPasswordStrength] = useState(0)

  // Calculate password strength
  const calculatePasswordStrength = (pwd) => {
    let strength = 0
    if (pwd.length >= 6) strength += 25
    if (/[A-Z]/.test(pwd)) strength += 25
    if (/[a-z]/.test(pwd)) strength += 25
    if (/\d/.test(pwd)) strength += 25
    setPasswordStrength(strength)
    return strength
  }

  const handlePasswordChange = (e) => {
    const pwd = e.target.value
    setPassword(pwd)
    calculatePasswordStrength(pwd)
  }

  const getPasswordStrengthText = () => {
    if (passwordStrength === 0) return ''
    if (passwordStrength < 40) return 'Weak'
    if (passwordStrength < 80) return 'Fair'
    return 'Strong'
  }

  const getPasswordStrengthColor = () => {
    if (passwordStrength === 0) return 'bg-slate-600'
    if (passwordStrength < 40) return 'bg-red-500'
    if (passwordStrength < 80) return 'bg-yellow-500'
    return 'bg-green-500'
  }

  const handleRegister = async (e) => {
    e.preventDefault()
    setError(null)
    setSuccess(null)
    setLoading(true)

    try {
      // Validate inputs
      if (password !== confirmPassword) {
        setError('Passwords do not match')
        setLoading(false)
        return
      }

      if (passwordStrength < 60) {
        setError('Password is not strong enough. Please use uppercase, lowercase, and either numbers or special characters.')
        setLoading(false)
        return
      }

      const response = await axios.post('/api/auth/register', {
        username,
        email,
        password
      })

      if (response.data.user_id) {
        setSuccess('Registration successful! Please log in with your new account.')
        setUsername('')
        setEmail('')
        setPassword('')
        setConfirmPassword('')
        setPasswordStrength(0)
        
        // Switch to login tab after 2 seconds
        setTimeout(() => {
          setIsRegistering(false)
          setSuccess(null)
        }, 2000)
      }
    } catch (err) {
      setError(err.response?.data?.detail || 'Registration failed. Please try again.')
      console.error('Registration error:', err)
    } finally {
      setLoading(false)
    }
  }

  const handleLogin = async (e) => {
    e.preventDefault()
    setLoading(true)
    setError(null)
    setSuccess(null)

    try {
      const response = await axios.post('/api/auth/login', {
        username,
        password
      })

      if (response.data.access_token) {
        localStorage.setItem('token', response.data.access_token)
        localStorage.setItem('refreshToken', response.data.refresh_token)
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
          const userData = { username, email: username, is_admin: false, id: response.data.user_id }
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

          {/* Tabs */}
          <div className="flex gap-2 mb-8">
            <button
              onClick={() => setIsRegistering(false)}
              className={`flex-1 py-2 px-4 rounded-lg font-medium transition-all ${
                !isRegistering
                  ? 'bg-sky-500 text-white'
                  : 'bg-slate-700 text-slate-300 hover:bg-slate-600'
              }`}
            >
              Log In
            </button>
            <button
              onClick={() => setIsRegistering(true)}
              className={`flex-1 py-2 px-4 rounded-lg font-medium transition-all ${
                isRegistering
                  ? 'bg-sky-500 text-white'
                  : 'bg-slate-700 text-slate-300 hover:bg-slate-600'
              }`}
            >
              Sign Up
            </button>
          </div>

          {/* Error Alert */}
          {error && (
            <div className="mb-6 p-4 bg-red-500/10 border border-red-500/50 rounded-lg flex gap-3">
              <AlertCircle className="w-5 h-5 text-red-400 flex-shrink-0 mt-0.5" />
              <p className="text-red-400 text-sm">{error}</p>
            </div>
          )}

          {/* Success Alert */}
          {success && (
            <div className="mb-6 p-4 bg-green-500/10 border border-green-500/50 rounded-lg flex gap-3">
              <CheckCircle className="w-5 h-5 text-green-400 flex-shrink-0 mt-0.5" />
              <p className="text-green-400 text-sm">{success}</p>
            </div>
          )}

          {/* Login Form */}
          {!isRegistering ? (
            <form onSubmit={handleLogin} className="space-y-4">
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
                    placeholder="Enter your username"
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
          ) : (
            /* Registration Form */
            <form onSubmit={handleRegister} className="space-y-4">
              {/* Username Field */}
              <div>
                <label className="block text-sm font-medium text-slate-300 mb-2">
                  Username
                </label>
                <input
                  type="text"
                  value={username}
                  onChange={(e) => setUsername(e.target.value)}
                  placeholder="Choose a username (3-50 chars)"
                  className={STYLES.input}
                  required
                  minLength={3}
                  maxLength={50}
                  disabled={loading}
                />
              </div>

              {/* Email Field */}
              <div>
                <label className="block text-sm font-medium text-slate-300 mb-2">
                  Email
                </label>
                <input
                  type="email"
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  placeholder="Enter your email (optional)"
                  className={STYLES.input}
                  disabled={loading}
                />
              </div>

              {/* Password Field */}
              <div>
                <label className="block text-sm font-medium text-slate-300 mb-2">
                  Password
                </label>
                <input
                  type="password"
                  value={password}
                  onChange={handlePasswordChange}
                  placeholder="Min 6 chars: uppercase, lowercase, and number"
                  className={STYLES.input}
                  required
                  disabled={loading}
                />
                {password && (
                  <div className="mt-2 space-y-2">
                    <div className="flex justify-between items-center text-xs">
                      <span className="text-slate-400">Strength:</span>
                      <span className={`font-semibold ${
                        passwordStrength < 40 ? 'text-red-400' :
                        passwordStrength < 80 ? 'text-yellow-400' :
                        'text-green-400'
                      }`}>
                        {getPasswordStrengthText()} ({passwordStrength}%)
                      </span>
                    </div>
                    <div className="w-full bg-slate-700 rounded-full h-2">
                      <div
                        className={`h-2 rounded-full transition-all ${getPasswordStrengthColor()}`}
                        style={{ width: `${passwordStrength}%` }}
                      ></div>
                    </div>
                    {passwordStrength < 80 && (
                      <div className="text-xs text-slate-400 space-y-1 mt-2">
                        <p>To enable Sign Up button, your password needs:</p>
                        <ul className="list-disc list-inside ml-1">
                          <li className={/[A-Z]/.test(password) ? 'text-green-400' : 'text-red-400'}>
                            {/[A-Z]/.test(password) ? '✓' : '✗'} One uppercase letter (A-Z)
                          </li>
                          <li className={/[a-z]/.test(password) ? 'text-green-400' : 'text-red-400'}>
                            {/[a-z]/.test(password) ? '✓' : '✗'} One lowercase letter (a-z)
                          </li>
                          <li className={password.length >= 6 ? 'text-green-400' : 'text-red-400'}>
                            {password.length >= 6 ? '✓' : '✗'} 6+ characters
                          </li>
                          <li className={/\d/.test(password) ? 'text-green-400' : 'text-red-400'}>
                            {/\d/.test(password) ? '✓' : '✗'} One number (0-9)
                          </li>
                        </ul>
                      </div>
                    )}
                  </div>
                )}
              </div>

              {/* Confirm Password Field */}
              <div>
                <label className="block text-sm font-medium text-slate-300 mb-2">
                  Confirm Password
                </label>
                <input
                  type="password"
                  value={confirmPassword}
                  onChange={(e) => setConfirmPassword(e.target.value)}
                  placeholder="Confirm your password"
                  className={STYLES.input}
                  required
                  disabled={loading}
                />
                {confirmPassword && password !== confirmPassword && (
                  <p className="text-red-400 text-xs mt-2">Passwords do not match</p>
                )}
              </div>

              {/* Submit Button */}
              <button
                type="submit"
                disabled={loading || password !== confirmPassword || passwordStrength < 60}
                className={`${STYLES.button.primary} w-full mt-6 flex items-center justify-center gap-2 disabled:opacity-50 disabled:cursor-not-allowed`}
                title={passwordStrength < 60 ? `Password strength: ${passwordStrength}% (Need 60%: uppercase, lowercase, and number or special char)` : 'Click to sign up'}
              >
                {loading ? (
                  <>
                    <Loader className="w-4 h-4 animate-spin" />
                    Creating account...
                  </>
                ) : (
                  'Sign Up'
                )}
              </button>
            </form>
          )}

          {/* Demo Credentials */}
          {!isRegistering && (
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
          )}

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
