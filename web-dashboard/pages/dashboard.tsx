import React, { useState, useEffect } from 'react'
import {
    TrendingUp,
    AlertCircle,
    Clock, MessageSquare, Download,
    RefreshCw,
    Eye
} from 'lucide-react'
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, PieChart, Pie, Cell } from 'recharts'

interface AnalyticsSummary {
    total_threads: number
    intents: Record<string, number>
    sentiment: Record<string, number>
    urgency: Record<string, number>
    processing_times: number[]
    avg_processing_time: number
}

interface ProcessingResult {
    thread_id: string
    summary: string
    intent: string
    confidence: number
    duration: string
    message_count: number
    user_count: number
    sentiment?: string
    urgency?: string
    entities?: string[]
    topics?: string[]
}

const DashboardPage: React.FC = () => {
    const [analytics, setAnalytics] = useState<AnalyticsSummary | null>(null)
    const [results, setResults] = useState<ProcessingResult[]>([])
    const [recentActivity, setRecentActivity] = useState<ProcessingResult[]>([])
    const [loading, setLoading] = useState(true)
    const [filter, setFilter] = useState('all')
    const [timeRange, setTimeRange] = useState('7d')

    const COLORS = ['#0ea5e9', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6', '#06b6d4', '#84cc16', '#f97316']

    useEffect(() => {
        fetchAnalytics()
        fetchResults()
        fetchRecentActivity()
        const interval = setInterval(() => {
            fetchAnalytics()
            fetchRecentActivity()
        }, 30000) // Update every 30 seconds
        return () => clearInterval(interval)
    }, [])

    const fetchAnalytics = async () => {
        try {
            const response = await fetch('/api/python/api/analytics/summary')
            const data = await response.json()
            setAnalytics(data)
        } catch (error) {
            console.error('Error fetching analytics:', error)
        } finally {
            setLoading(false)
        }
    }

    const fetchResults = async () => {
        try {
            const response = await fetch('/api/python/api/results?per_page=50')
            const data = await response.json()
            setResults(data.results)
        } catch (error) {
            console.error('Error fetching results:', error)
        }
    }

    const fetchRecentActivity = async () => {
        try {
            const response = await fetch('/api/python/api/analytics/recent?limit=10')
            const data = await response.json()
            setRecentActivity(data.recent_results)
        } catch (error) {
            console.error('Error fetching recent activity:', error)
        }
    }

    const prepareIntentData = () => {
        if (!analytics?.intents) return []
        const totalThreads = analytics.total_threads || 1 // Avoid division by zero
        return Object.entries(analytics.intents).map(([name, value]) => ({
            name: name.replace('_', ' ').replace(/\b\w/g, l => l.toUpperCase()),
            value,
            percentage: totalThreads > 0 ? ((value / totalThreads) * 100).toFixed(1) : '0.0'
        }))
    }

    const prepareSentimentData = () => {
        if (!analytics?.sentiment) return []
        const totalThreads = analytics.total_threads || 1 // Avoid division by zero
        return Object.entries(analytics.sentiment).map(([name, value]) => ({
            name: name.charAt(0).toUpperCase() + name.slice(1),
            value,
            percentage: totalThreads > 0 ? ((value / totalThreads) * 100).toFixed(1) : '0.0'
        }))
    }

    const prepareUrgencyData = () => {
        if (!analytics?.urgency) return []
        const totalThreads = analytics.total_threads || 1 // Avoid division by zero
        return Object.entries(analytics.urgency).map(([name, value]) => ({
            name: name.charAt(0).toUpperCase() + name.slice(1),
            value,
            percentage: totalThreads > 0 ? ((value / totalThreads) * 100).toFixed(1) : '0.0'
        }))
    }

    const getFilteredResults = () => {
        if (filter === 'all') return results
        return results.filter(result => result.intent === filter)
    }

    const getTopUsers = () => {
        const userCounts: Record<string, number> = {}
        results.forEach(result => {
            userCounts[result.thread_id] = (userCounts[result.thread_id] || 0) + result.user_count
        })
        return Object.entries(userCounts)
            .sort(([, a], [, b]) => b - a)
            .slice(0, 10)
            .map(([id, count]) => ({ id, count }))
    }

    const getHighestActivity = () => {
        return results
            .sort((a, b) => b.message_count - a.message_count)
            .slice(0, 10)
    }

    const getUrgentIssues = () => {
        return results
            .filter(r => r.urgency === 'urgent' || r.intent === 'bug_report')
            .sort((a, b) => b.confidence - a.confidence)
            .slice(0, 10)
    }

    if (loading) {
        return (
            <div className="flex items-center justify-center h-64">
                <div className="loading-spinner"></div>
                <span className="ml-3 text-gray-600">Loading analytics...</span>
            </div>
        )
    }

    return (
        <div className="space-y-6">
            {/* Page Header */}
            <div className="flex justify-between items-center">
                <div>
                    <h1 className="text-3xl font-bold text-gray-900">Analytics Dashboard</h1>
                    <p className="text-gray-600 mt-2">
                        Insights and metrics from processed Slack threads
                    </p>
                </div>
                <div className="flex space-x-3">
                    <button
                        onClick={() => window.location.reload()}
                        className="flex items-center px-4 py-2 bg-gray-200 text-gray-700 rounded-lg hover:bg-gray-300 transition-colors"
                    >
                        <RefreshCw className="w-4 h-4 mr-2" />
                        Refresh
                    </button>
                    <button className="flex items-center px-4 py-2 bg-primary-600 text-white rounded-lg hover:bg-primary-700 transition-colors">
                        <Download className="w-4 h-4 mr-2" />
                        Export Data
                    </button>
                </div>
            </div>

            {/* Key Metrics */}
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
                <div className="bg-white rounded-lg shadow-sm p-6 border">
                    <div className="flex items-center justify-between">
                        <div>
                            <p className="text-sm font-medium text-gray-600">Total Threads</p>
                            <p className="text-2xl font-bold text-gray-900">
                                {analytics?.total_threads || 0}
                            </p>
                        </div>
                        <div className="p-2 rounded-full bg-blue-100 text-blue-600">
                            <MessageSquare className="w-6 h-6" />
                        </div>
                    </div>
                </div>

                <div className="bg-white rounded-lg shadow-sm p-6 border">
                    <div className="flex items-center justify-between">
                        <div>
                            <p className="text-sm font-medium text-gray-600">Avg Processing Time</p>
                            <p className="text-2xl font-bold text-gray-900">
                                {analytics?.avg_processing_time ? analytics.avg_processing_time.toFixed(2) : '0.00'}s
                            </p>
                        </div>
                        <div className="p-2 rounded-full bg-green-100 text-green-600">
                            <Clock className="w-6 h-6" />
                        </div>
                    </div>
                </div>

                <div className="bg-white rounded-lg shadow-sm p-6 border">
                    <div className="flex items-center justify-between">
                        <div>
                            <p className="text-sm font-medium text-gray-600">Bug Reports</p>
                            <p className="text-2xl font-bold text-gray-900">
                                {analytics?.intents?.bug_report || 0}
                            </p>
                        </div>
                        <div className="p-2 rounded-full bg-red-100 text-red-600">
                            <AlertCircle className="w-6 h-6" />
                        </div>
                    </div>
                </div>

                <div className="bg-white rounded-lg shadow-sm p-6 border">
                    <div className="flex items-center justify-between">
                        <div>
                            <p className="text-sm font-medium text-gray-600">Feature Requests</p>
                            <p className="text-2xl font-bold text-gray-900">
                                {analytics?.intents?.feature_request || 0}
                            </p>
                        </div>
                        <div className="p-2 rounded-full bg-purple-100 text-purple-600">
                            <TrendingUp className="w-6 h-6" />
                        </div>
                    </div>
                </div>
            </div>

            {/* Charts Row */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                {/* Intent Distribution */}
                <div className="bg-white rounded-lg shadow-sm p-6 border">
                    <h2 className="text-xl font-semibold text-gray-900 mb-4">Intent Distribution</h2>
                    <div className="h-80">
                        <ResponsiveContainer width="100%" height="100%">
                            <BarChart data={prepareIntentData()}>
                                <CartesianGrid strokeDasharray="3 3" />
                                <XAxis dataKey="name" angle={-45} textAnchor="end" height={60} />
                                <YAxis />
                                <Tooltip />
                                <Bar dataKey="value" fill="#0ea5e9" />
                            </BarChart>
                        </ResponsiveContainer>
                    </div>
                </div>

                {/* Sentiment Analysis */}
                <div className="bg-white rounded-lg shadow-sm p-6 border">
                    <h2 className="text-xl font-semibold text-gray-900 mb-4">Sentiment Analysis</h2>
                    <div className="h-80">
                        <ResponsiveContainer width="100%" height="100%">
                            <PieChart>
                                <Pie
                                    data={prepareSentimentData()}
                                    cx="50%"
                                    cy="50%"
                                    labelLine={false}
                                    label={({ name, percentage }) => `${name} (${percentage}%)`}
                                    outerRadius={80}
                                    fill="#8884d8"
                                    dataKey="value"
                                >
                                    {prepareSentimentData().map((entry, index) => (
                                        <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                                    ))}
                                </Pie>
                                <Tooltip />
                            </PieChart>
                        </ResponsiveContainer>
                    </div>
                </div>
            </div>

            {/* Activity Tables */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                {/* Most Active Threads */}
                <div className="bg-white rounded-lg shadow-sm p-6 border">
                    <h2 className="text-xl font-semibold text-gray-900 mb-4">Most Active Threads</h2>
                    <div className="space-y-3">
                        {getHighestActivity().map((result, index) => (
                            <div key={index} className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
                                <div className="flex-1">
                                    <p className="font-medium text-gray-900 truncate">{result.thread_id}</p>
                                    <p className="text-sm text-gray-600 truncate">{result.summary}</p>
                                </div>
                                <div className="text-right ml-4">
                                    <p className="text-sm font-medium text-gray-900">{result.message_count} messages</p>
                                    <p className="text-xs text-gray-500">{result.user_count} users</p>
                                </div>
                            </div>
                        ))}
                    </div>
                </div>

                {/* Urgent Issues */}
                <div className="bg-white rounded-lg shadow-sm p-6 border">
                    <h2 className="text-xl font-semibold text-gray-900 mb-4">Urgent Issues</h2>
                    <div className="space-y-3">
                        {getUrgentIssues().map((result, index) => (
                            <div key={index} className="flex items-center justify-between p-3 bg-red-50 rounded-lg border border-red-200">
                                <div className="flex-1">
                                    <p className="font-medium text-gray-900 truncate">{result.thread_id}</p>
                                    <p className="text-sm text-gray-600 truncate">{result.summary}</p>
                                </div>
                                <div className="text-right ml-4">
                                    <span className={`px-2 py-1 rounded text-xs ${result.intent === 'bug_report' ? 'bg-red-100 text-red-800' : 'bg-orange-100 text-orange-800'
                                        }`}>
                                        {result.intent.replace('_', ' ')}
                                    </span>
                                    <p className="text-xs text-gray-500 mt-1">{(result.confidence * 100).toFixed(1)}% confidence</p>
                                </div>
                            </div>
                        ))}
                    </div>
                </div>
            </div>

            {/* Recent Activity */}
            <div className="bg-white rounded-lg shadow-sm p-6 border">
                <div className="flex items-center justify-between mb-4">
                    <h2 className="text-xl font-semibold text-gray-900">Recent Activity</h2>
                    <select
                        value={filter}
                        onChange={(e) => setFilter(e.target.value)}
                        className="px-3 py-2 border rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-primary-500"
                    >
                        <option value="all">All Intents</option>
                        <option value="bug_report">Bug Reports</option>
                        <option value="feature_request">Feature Requests</option>
                        <option value="how_to_question">How-to Questions</option>
                        <option value="troubleshooting">Troubleshooting</option>
                    </select>
                </div>

                <div className="space-y-3">
                    {recentActivity.map((result, index) => (
                        <div key={index} className="flex items-center justify-between p-4 border rounded-lg hover:bg-gray-50 transition-colors">
                            <div className="flex-1">
                                <div className="flex items-center space-x-3 mb-2">
                                    <h3 className="font-medium text-gray-900">{result.thread_id}</h3>
                                    <span className={`px-2 py-1 rounded text-xs ${result.intent === 'bug_report' ? 'bg-red-100 text-red-800' :
                                        result.intent === 'feature_request' ? 'bg-blue-100 text-blue-800' :
                                            result.intent === 'how_to_question' ? 'bg-green-100 text-green-800' :
                                                'bg-gray-100 text-gray-800'
                                        }`}>
                                        {result.intent.replace('_', ' ')}
                                    </span>
                                    {result.sentiment && (
                                        <span className={`px-2 py-1 rounded text-xs ${result.sentiment === 'positive' ? 'bg-green-100 text-green-800' :
                                            result.sentiment === 'negative' ? 'bg-red-100 text-red-800' :
                                                'bg-gray-100 text-gray-800'
                                            }`}>
                                            {result.sentiment}
                                        </span>
                                    )}
                                </div>
                                <p className="text-sm text-gray-600 mb-2">{result.summary}</p>
                                <div className="flex items-center space-x-4 text-xs text-gray-500">
                                    <span>Confidence: {(result.confidence * 100).toFixed(1)}%</span>
                                    <span>Duration: {result.duration}</span>
                                    <span>Messages: {result.message_count}</span>
                                    <span>Users: {result.user_count}</span>
                                </div>
                            </div>
                            <div className="ml-4">
                                <button className="flex items-center px-3 py-1 text-sm text-primary-600 hover:text-primary-700">
                                    <Eye className="w-4 h-4 mr-1" />
                                    View
                                </button>
                            </div>
                        </div>
                    ))}
                </div>
            </div>
        </div>
    )
}

export default DashboardPage 