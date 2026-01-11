import React, { useState, useEffect } from 'react'
import { Activity, Users, MessageSquare, BookOpen, Zap, TrendingUp, RefreshCw } from 'lucide-react'
import axios from 'axios'

const DashboardStats = () => {
  const [stats, setStats] = useState(null)
  const [loading, setLoading] = useState(false)
  const [lastUpdated, setLastUpdated] = useState(null)
  const [error, setError] = useState(null)

  const api = axios.create({
    baseURL: '/api',
    headers: {
      'Authorization': `Bearer ${localStorage.getItem('token')}`
    }
  })

  const fetchStats = async () => {
    try {
      setLoading(true)
      const res = await api.get('/admin/stats')
      setStats(res.data)
      setLastUpdated(new Date())
      setError(null)
    } catch (err) {
      console.error('Failed to fetch stats:', err)
      setError('Unable to load statistics')
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    // Fetch immediately on mount
    fetchStats()

    // Set up auto-refresh every 10 seconds
    const interval = setInterval(fetchStats, 10000)
    return () => clearInterval(interval)
  }, [])

  if (error && !stats) {
    return null // Don't show error state if we have some data
  }

  if (!stats) {
    return null // Don't show anything while loading initially
  }

  const StatCard = ({ icon: Icon, title, value, unit, color = 'blue', subtext }) => {
    const colorClasses = {
      blue: 'border-l-blue-500 bg-blue-500/5',
      green: 'border-l-green-500 bg-green-500/5',
      purple: 'border-l-purple-500 bg-purple-500/5',
      amber: 'border-l-amber-500 bg-amber-500/5'
    }

    const iconClasses = {
      blue: 'text-blue-400',
      green: 'text-green-400',
      purple: 'text-purple-400',
      amber: 'text-amber-400'
    }

    return (
      <div className={`border-l-4 ${colorClasses[color]} rounded-lg p-4 backdrop-blur-sm`}>
        <div className="flex items-start justify-between">
          <div>
            <p className="text-sm text-slate-400 mb-2">{title}</p>
            <div className="flex items-baseline gap-2">
              <span className="text-2xl font-bold text-white">{value}</span>
              {unit && <span className="text-sm text-slate-500">{unit}</span>}
            </div>
            {subtext && <p className="text-xs text-slate-500 mt-1">{subtext}</p>}
          </div>
          <Icon className={`w-6 h-6 ${iconClasses[color]}`} />
        </div>
      </div>
    )
  }

  return (
    <div className="space-y-4">
      {/* Header with refresh button */}
      <div className="flex items-center justify-between px-2">
        <h2 className="text-lg font-semibold text-white">📊 Quick Stats</h2>
        <button
          onClick={fetchStats}
          disabled={loading}
          className="p-1.5 hover:bg-slate-700 rounded-lg transition-colors disabled:opacity-50"
          title="Refresh stats"
        >
          <RefreshCw className={`w-4 h-4 text-slate-400 ${loading ? 'animate-spin' : ''}`} />
        </button>
      </div>

      {/* Stats Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
        <StatCard
          icon={Users}
          title="Active Users"
          value={stats.users?.active || 0}
          unit="users"
          color="blue"
          subtext={`${stats.users?.total || 0} total`}
        />
        <StatCard
          icon={MessageSquare}
          title="Total Messages"
          value={stats.messages?.total || 0}
          unit="messages"
          color="green"
          subtext={`${stats.messages?.last_24h || 0} today`}
        />
        <StatCard
          icon={BookOpen}
          title="Sessions"
          value={stats.sessions?.total || 0}
          unit="sessions"
          color="purple"
          subtext={`${stats.sessions?.rag_adoption_percent || 0}% using RAG`}
        />
        <StatCard
          icon={Zap}
          title="Avg Messages"
          value={(stats.messages?.avg_per_session || 0).toFixed(1)}
          unit="per session"
          color="amber"
        />
      </div>

      {/* Last updated info */}
      <div className="text-xs text-slate-500 text-center pt-2 border-t border-slate-700">
        {lastUpdated && `Last updated: ${lastUpdated.toLocaleTimeString('vi-VN')}`}
      </div>
    </div>
  )
}

export default DashboardStats
