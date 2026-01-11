import React, { useState, useEffect } from 'react'
import { Activity, Zap, MessageCircle, Database } from 'lucide-react'

const LiveActivity = ({ currentStep, isProcessing }) => {
  const [recentActivities, setRecentActivities] = useState([])

  useEffect(() => {
    if (currentStep && isProcessing) {
      const activity = {
        id: Date.now(),
        type: currentStep.step || 'unknown',
        status: currentStep.status || 'in-progress',
        timestamp: new Date()
      }
      
      setRecentActivities(prev => {
        const updated = [activity, ...prev].slice(0, 5) // Keep last 5
        return updated
      })
    }
  }, [currentStep, isProcessing])

  if (!isProcessing && recentActivities.length === 0) {
    return null
  }

  const getActivityIcon = (type) => {
    if (type.includes('RETRIEVE') || type.includes('FILTER')) {
      return <Database className="w-4 h-4 text-cyan-400" />
    } else if (type.includes('TOOL')) {
      return <Zap className="w-4 h-4 text-amber-400" />
    } else if (type.includes('GENERATE')) {
      return <MessageCircle className="w-4 h-4 text-emerald-400" />
    }
    return <Activity className="w-4 h-4 text-blue-400" />
  }

  const getActivityLabel = (type) => {
    const labels = {
      'RETRIEVE': 'Searching knowledge base',
      'FILTER': 'Ranking results',
      'EXECUTE_TOOLS': 'Running analysis tools',
      'GENERATE': 'Generating response',
      'SELECT_TOOLS': 'Selecting tools',
      'CLASSIFY': 'Analyzing query type'
    }
    return labels[type] || type
  }

  return (
    <div className="border-l-4 border-green-500 bg-slate-900/60 rounded-r-lg p-3 mt-3">
      <div className="flex items-center gap-2 text-green-400 text-sm font-semibold mb-2">
        <div className="w-2 h-2 bg-green-400 rounded-full animate-pulse" />
        Live Activity
      </div>
      
      <div className="space-y-2 max-h-40 overflow-y-auto">
        {recentActivities.map(activity => (
          <div key={activity.id} className="flex items-start gap-2 text-xs">
            {getActivityIcon(activity.type)}
            <div className="flex-1">
              <p className="text-slate-300">{getActivityLabel(activity.type)}</p>
              <p className="text-slate-500">
                {activity.timestamp.toLocaleTimeString('vi-VN', { 
                  hour: '2-digit', 
                  minute: '2-digit', 
                  second: '2-digit' 
                })}
              </p>
            </div>
            {activity.status === 'in-progress' && (
              <div className="animate-spin">
                <div className="w-3 h-3 border-2 border-slate-600 border-t-cyan-400 rounded-full" />
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  )
}

export default LiveActivity
