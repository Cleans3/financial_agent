import React, { useState, useEffect } from 'react'
import { LineChart, Line, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, PieChart, Pie, Cell } from 'recharts'
import { ArrowLeft } from 'lucide-react'
import axios from 'axios'

const AdminDashboard = ({ user, onLogout, onClose }) => {
  const [activeTab, setActiveTab] = useState('dashboard')
  const [stats, setStats] = useState(null)
  const [ragStats, setRagStats] = useState(null)
  const [users, setUsers] = useState([])
  const [logs, setLogs] = useState([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  const api = axios.create({
    baseURL: '/api',
    headers: {
      'Authorization': `Bearer ${localStorage.getItem('token')}`
    }
  })

  const handleLogout = () => {
    if (onLogout) {
      onLogout()
    }
  }

  const fetchStats = async () => {
    try {
      setLoading(true)
      const [statsRes, ragRes] = await Promise.all([
        api.get('/admin/stats'),
        api.get('/admin/rag-stats')
      ])
      setStats(statsRes.data)
      setRagStats(ragRes.data)
      setError(null)
    } catch (err) {
      setError('Failed to fetch statistics')
      console.error(err)
    } finally {
      setLoading(false)
    }
  }

  const fetchUsers = async () => {
    try {
      setLoading(true)
      const res = await api.get('/admin/users')
      setUsers(res.data)
      setError(null)
    } catch (err) {
      setError('Failed to fetch users')
      console.error(err)
    } finally {
      setLoading(false)
    }
  }

  const fetchLogs = async () => {
    try {
      setLoading(true)
      const res = await api.get('/admin/audit-logs')
      setLogs(res.data)
      setError(null)
    } catch (err) {
      setError('Failed to fetch logs')
      console.error(err)
    } finally {
      setLoading(false)
    }
  }

  const toggleUserActive = async (userId, currentStatus) => {
    try {
      await api.post(`/admin/users/${userId}/toggle-active`)
      fetchUsers()
    } catch (err) {
      setError('Failed to toggle user status')
      console.error(err)
    }
  }

  const deleteUserData = async (userId) => {
    if (!confirm('Are you sure you want to delete this user\'s data?')) return
    try {
      await api.delete(`/admin/users/${userId}/data`)
      fetchUsers()
    } catch (err) {
      setError('Failed to delete user data')
      console.error(err)
    }
  }

  useEffect(() => {
    if (activeTab === 'dashboard') {
      fetchStats()
    } else if (activeTab === 'users') {
      fetchUsers()
    } else if (activeTab === 'logs') {
      fetchLogs()
    }
  }, [activeTab])

  return (
    <div className="min-h-screen bg-gray-900 text-white">
      {/* Header */}
      <header className="bg-gray-800 border-b border-gray-700">
        <div className="max-w-7xl mx-auto px-4 py-6">
          <div className="flex justify-between items-center">
            <div className="flex items-center gap-4">
              {onClose && (
                <button
                  onClick={onClose}
                  className="flex items-center gap-2 px-4 py-2 rounded text-sm font-medium hover:bg-gray-700 transition"
                >
                  <ArrowLeft className="w-4 h-4" />
                  Back
                </button>
              )}
              <h1 className="text-3xl font-bold">Admin Dashboard</h1>
            </div>
            <div className="flex items-center space-x-4">
              <div className="text-sm text-gray-400">
                Welcome, <span className="text-blue-400">{user.username}</span>
              </div>
              <button
                onClick={handleLogout}
                className="bg-red-600 hover:bg-red-700 px-4 py-2 rounded text-sm font-medium transition"
              >
                Log Out
              </button>
            </div>
          </div>
        </div>
      </header>

      {/* Tab Navigation */}
      <div className="bg-gray-800 border-b border-gray-700">
        <div className="max-w-7xl mx-auto px-4">
          <nav className="flex space-x-8">
            {['dashboard', 'users', 'logs'].map(tab => (
              <button
                key={tab}
                onClick={() => setActiveTab(tab)}
                className={`py-4 px-1 border-b-2 font-medium text-sm capitalize ${
                  activeTab === tab
                    ? 'border-blue-500 text-blue-400'
                    : 'border-transparent text-gray-400 hover:text-white hover:border-gray-300'
                }`}
              >
                {tab}
              </button>
            ))}
          </nav>
        </div>
      </div>

      {/* Content */}
      <main className="max-w-7xl mx-auto px-4 py-8">
        {error && (
          <div className="mb-4 p-4 bg-red-900 border border-red-700 rounded text-red-200">
            {error}
          </div>
        )}

        {/* Dashboard Tab */}
        {activeTab === 'dashboard' && (
          <div className="space-y-6">
            {loading ? (
              <div className="text-center py-12">Loading...</div>
            ) : stats ? (
              <>
                {/* Stats Cards */}
                <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
                  <StatCard
                    title="Total Users"
                    value={stats.total_users}
                    icon="👥"
                  />
                  <StatCard
                    title="Active Sessions"
                    value={stats.active_sessions}
                    icon="🔄"
                  />
                  <StatCard
                    title="Total Documents"
                    value={stats.total_documents}
                    icon="📄"
                  />
                  <StatCard
                    title="RAG Adoption Rate"
                    value={`${(stats.rag_adoption_rate * 100).toFixed(1)}%`}
                    icon="📊"
                  />
                </div>

                {/* Charts */}
                {ragStats && (
                  <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                    <div className="bg-gray-800 p-6 rounded-lg border border-gray-700">
                      <h3 className="text-lg font-semibold mb-4">Queries by Type</h3>
                      <ResponsiveContainer width="100%" height={300}>
                        <BarChart data={Object.entries(ragStats.queries_by_type || {}).map(([key, val]) => ({
                          name: key,
                          count: val
                        }))}>
                          <CartesianGrid strokeDasharray="3 3" stroke="#4b5563" />
                          <XAxis dataKey="name" stroke="#9ca3af" />
                          <YAxis stroke="#9ca3af" />
                          <Tooltip
                            contentStyle={{ backgroundColor: '#1f2937', border: '1px solid #4b5563' }}
                            cursor={{ fill: '#374151' }}
                          />
                          <Bar dataKey="count" fill="#3b82f6" />
                        </BarChart>
                      </ResponsiveContainer>
                    </div>

                    <div className="bg-gray-800 p-6 rounded-lg border border-gray-700">
                      <h3 className="text-lg font-semibold mb-4">Usage Distribution</h3>
                      <ResponsiveContainer width="100%" height={300}>
                        <PieChart>
                          <Pie
                            data={Object.entries(ragStats.queries_by_type || {}).map(([key, val]) => ({
                              name: key,
                              value: val
                            }))}
                            cx="50%"
                            cy="50%"
                            labelLine={false}
                            label={({ name, percent }) => `${name}: ${(percent * 100).toFixed(0)}%`}
                            outerRadius={80}
                            fill="#8884d8"
                            dataKey="value"
                          >
                            {['#3b82f6', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6'].map((color, idx) => (
                              <Cell key={`cell-${idx}`} fill={color} />
                            ))}
                          </Pie>
                          <Tooltip
                            contentStyle={{ backgroundColor: '#1f2937', border: '1px solid #4b5563' }}
                          />
                        </PieChart>
                      </ResponsiveContainer>
                    </div>
                  </div>
                )}
              </>
            ) : null}
          </div>
        )}

        {/* Users Tab */}
        {activeTab === 'users' && (
          <div className="bg-gray-800 rounded-lg border border-gray-700 overflow-hidden">
            {loading ? (
              <div className="text-center py-12">Loading...</div>
            ) : (
              <div className="overflow-x-auto">
                <table className="w-full">
                  <thead className="bg-gray-700 border-b border-gray-600">
                    <tr>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-300 uppercase tracking-wider">User</th>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-300 uppercase tracking-wider">Email</th>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-300 uppercase tracking-wider">Role</th>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-300 uppercase tracking-wider">Status</th>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-300 uppercase tracking-wider">Actions</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-gray-700">
                    {users.map(user => (
                      <tr key={user.id} className="hover:bg-gray-700 transition">
                        <td className="px-6 py-4 whitespace-nowrap text-sm">{user.username}</td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-400">{user.email}</td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm">
                          <span className={`px-3 py-1 rounded-full text-xs font-medium ${
                            user.is_admin
                              ? 'bg-red-900 text-red-200'
                              : 'bg-blue-900 text-blue-200'
                          }`}>
                            {user.is_admin ? 'admin' : 'user'}
                          </span>
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm">
                          <span className={`px-3 py-1 rounded-full text-xs font-medium ${
                            user.is_active
                              ? 'bg-green-900 text-green-200'
                              : 'bg-gray-700 text-gray-400'
                          }`}>
                            {user.is_active ? 'Active' : 'Inactive'}
                          </span>
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm space-x-2">
                          <button
                            onClick={() => toggleUserActive(user.id, user.is_active)}
                            className="bg-blue-600 hover:bg-blue-700 px-3 py-1 rounded transition text-xs"
                          >
                            {user.is_active ? 'Deactivate' : 'Activate'}
                          </button>
                          <button
                            onClick={() => deleteUserData(user.id)}
                            className="bg-red-600 hover:bg-red-700 px-3 py-1 rounded transition text-xs"
                          >
                            Delete Data
                          </button>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </div>
        )}

        {/* Logs Tab */}
        {activeTab === 'logs' && (
          <div className="bg-gray-800 rounded-lg border border-gray-700 overflow-hidden">
            {loading ? (
              <div className="text-center py-12">Loading...</div>
            ) : (
              <div className="overflow-x-auto">
                <table className="w-full">
                  <thead className="bg-gray-700 border-b border-gray-600">
                    <tr>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-300 uppercase tracking-wider">Timestamp</th>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-300 uppercase tracking-wider">Action</th>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-300 uppercase tracking-wider">User</th>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-300 uppercase tracking-wider">Details</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-gray-700">
                    {logs.map((log, idx) => (
                      <tr key={idx} className="hover:bg-gray-700 transition">
                        <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-400">
                          {new Date(log.timestamp).toLocaleString()}
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm">
                          <span className="px-3 py-1 rounded-full text-xs font-medium bg-purple-900 text-purple-200">
                            {log.action}
                          </span>
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm">{log.user_id}</td>
                        <td className="px-6 py-4 text-sm text-gray-400 max-w-xs truncate">{log.details}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </div>
        )}
      </main>
    </div>
  )
}

const StatCard = ({ title, value, icon }) => (
  <div className="bg-gray-800 border border-gray-700 rounded-lg p-6 hover:border-gray-600 transition">
    <div className="flex items-center justify-between">
      <div>
        <p className="text-gray-400 text-sm font-medium">{title}</p>
        <p className="text-3xl font-bold mt-2">{value}</p>
      </div>
      <div className="text-4xl">{icon}</div>
    </div>
  </div>
)

export default AdminDashboard
