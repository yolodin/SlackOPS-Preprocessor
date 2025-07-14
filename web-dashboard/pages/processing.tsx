import React, { useState, useEffect } from 'react'
import {
    Play,
    Square,
    RefreshCw,
    CheckCircle, Clock,
    Database,
    Settings,
    FileText,
    Brain,
    Activity
} from 'lucide-react'

interface ProcessingStatus {
    is_processing: boolean
    current_thread: string | null
    progress: number
    total_threads: number
    results: any[]
    errors: string[]
    start_time: string | null
    end_time: string | null
    processing_method: string
}

interface ProcessingConfig {
    data_file: string
    use_ml: boolean
    use_lightweight: boolean
    max_threads: number | null
}

const ProcessingPage: React.FC = () => {
    const [status, setStatus] = useState<ProcessingStatus | null>(null)
    const [config, setConfig] = useState<ProcessingConfig>({
        data_file: 'data/standardized_slack_data.json',
        use_ml: true,
        use_lightweight: true,
        max_threads: null
    })
    const [dataFiles, setDataFiles] = useState<any[]>([])
    const [logs, setLogs] = useState<string[]>([])
    const [loading, setLoading] = useState(false)

    useEffect(() => {
        fetchProcessingStatus()
        fetchDataFiles()
        const interval = setInterval(fetchProcessingStatus, 1000) // Update every second
        return () => clearInterval(interval)
    }, [])

    const fetchProcessingStatus = async () => {
        try {
            const response = await fetch('/api/python/api/processing/status')
            const data = await response.json()
            setStatus(data)

            // Add to logs if processing
            if (data.is_processing && data.current_thread) {
                setLogs(prev => [
                    ...prev.slice(-50), // Keep last 50 logs
                    `[${new Date().toLocaleTimeString()}] Processing ${data.current_thread} (${data.progress}/${data.total_threads})`
                ])
            }
        } catch (error) {
            console.error('Error fetching processing status:', error)
        }
    }

    const fetchDataFiles = async () => {
        try {
            const response = await fetch('/api/python/api/data/files')
            const data = await response.json()
            setDataFiles(data.files)
        } catch (error) {
            console.error('Error fetching data files:', error)
        }
    }

    const startProcessing = async () => {
        setLoading(true)
        setLogs([])
        try {
            const response = await fetch('/api/python/api/processing/start', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(config)
            })

            if (response.ok) {
                setLogs(prev => [...prev, `[${new Date().toLocaleTimeString()}] Processing started`])
            } else {
                const error = await response.json()
                setLogs(prev => [...prev, `[${new Date().toLocaleTimeString()}] Error: ${error.error}`])
            }
        } catch (error) {
            console.error('Error starting processing:', error)
            setLogs(prev => [...prev, `[${new Date().toLocaleTimeString()}] Error: ${error}`])
        } finally {
            setLoading(false)
        }
    }

    const stopProcessing = async () => {
        try {
            const response = await fetch('/api/python/api/processing/stop', {
                method: 'POST'
            })

            if (response.ok) {
                setLogs(prev => [...prev, `[${new Date().toLocaleTimeString()}] Processing stopped`])
            }
        } catch (error) {
            console.error('Error stopping processing:', error)
        }
    }

    const getProgressPercentage = () => {
        if (!status || status.total_threads === 0) return 0
        return Math.round((status.progress / status.total_threads) * 100)
    }

    const getProcessingDuration = () => {
        if (!status?.start_time) return null
        const start = new Date(status.start_time)
        const end = status.end_time ? new Date(status.end_time) : new Date()
        return Math.round((end.getTime() - start.getTime()) / 1000)
    }

    return (
        <div className="space-y-6">
            {/* Page Header */}
            <div className="flex justify-between items-center">
                <div>
                    <h1 className="text-3xl font-bold text-gray-900">Data Processing</h1>
                    <p className="text-gray-600 mt-2">
                        Process Slack threads with AI/ML analysis
                    </p>
                </div>
                <div className="flex space-x-3">
                    {status?.is_processing ? (
                        <button
                            onClick={stopProcessing}
                            className="flex items-center px-4 py-2 bg-red-600 text-white rounded-lg hover:bg-red-700 transition-colors"
                        >
                            <Square className="w-4 h-4 mr-2" />
                            Stop Processing
                        </button>
                    ) : (
                        <button
                            onClick={startProcessing}
                            disabled={loading}
                            className="flex items-center px-4 py-2 bg-primary-600 text-white rounded-lg hover:bg-primary-700 transition-colors disabled:opacity-50"
                        >
                            {loading ? (
                                <RefreshCw className="w-4 h-4 mr-2 animate-spin" />
                            ) : (
                                <Play className="w-4 h-4 mr-2" />
                            )}
                            Start Processing
                        </button>
                    )}
                </div>
            </div>

            {/* Configuration Panel */}
            <div className="bg-white rounded-lg shadow-sm p-6 border">
                <h2 className="text-xl font-semibold text-gray-900 mb-4 flex items-center">
                    <Settings className="w-5 h-5 mr-2" />
                    Processing Configuration
                </h2>

                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
                    <div>
                        <label className="block text-sm font-medium text-gray-700 mb-2">
                            Data File
                        </label>
                        <select
                            value={config.data_file}
                            onChange={(e) => setConfig({ ...config, data_file: e.target.value })}
                            className="w-full p-2 border rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-primary-500"
                        >
                            {dataFiles.map(file => (
                                <option key={file.name} value={`data/${file.name}`}>
                                    {file.name}
                                </option>
                            ))}
                        </select>
                    </div>

                    <div>
                        <label className="block text-sm font-medium text-gray-700 mb-2">
                            Processing Method
                        </label>
                        <select
                            value={config.use_ml ? 'ml' : 'rule-based'}
                            onChange={(e) => setConfig({ ...config, use_ml: e.target.value === 'ml' })}
                            className="w-full p-2 border rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-primary-500"
                        >
                            <option value="ml">Machine Learning</option>
                            <option value="rule-based">Rule-based</option>
                        </select>
                    </div>

                    <div>
                        <label className="block text-sm font-medium text-gray-700 mb-2">
                            ML Model Type
                        </label>
                        <select
                            value={config.use_lightweight ? 'lightweight' : 'full'}
                            onChange={(e) => setConfig({ ...config, use_lightweight: e.target.value === 'lightweight' })}
                            disabled={!config.use_ml}
                            className="w-full p-2 border rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-primary-500 disabled:opacity-50"
                        >
                            <option value="lightweight">Lightweight (Fast)</option>
                            <option value="full">Full Transformer</option>
                        </select>
                    </div>

                    <div>
                        <label className="block text-sm font-medium text-gray-700 mb-2">
                            Max Threads
                        </label>
                        <input
                            type="number"
                            value={config.max_threads || ''}
                            onChange={(e) => setConfig({ ...config, max_threads: e.target.value ? parseInt(e.target.value) : null })}
                            placeholder="All threads"
                            className="w-full p-2 border rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-primary-500"
                        />
                    </div>
                </div>
            </div>

            {/* Processing Status */}
            {status && (
                <div className="bg-white rounded-lg shadow-sm p-6 border">
                    <h2 className="text-xl font-semibold text-gray-900 mb-4 flex items-center">
                        <Activity className="w-5 h-5 mr-2" />
                        Processing Status
                    </h2>

                    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
                        <div className="flex items-center space-x-3">
                            <div className={`p-2 rounded-full ${status.is_processing ? 'bg-green-100 text-green-600' : 'bg-gray-100 text-gray-600'
                                }`}>
                                <Activity className="w-5 h-5" />
                            </div>
                            <div>
                                <p className="text-sm text-gray-600">Status</p>
                                <p className="font-medium">
                                    {status.is_processing ? 'Processing' : 'Idle'}
                                </p>
                            </div>
                        </div>

                        <div className="flex items-center space-x-3">
                            <div className="p-2 rounded-full bg-blue-100 text-blue-600">
                                <Database className="w-5 h-5" />
                            </div>
                            <div>
                                <p className="text-sm text-gray-600">Progress</p>
                                <p className="font-medium">
                                    {status.progress} / {status.total_threads}
                                </p>
                            </div>
                        </div>

                        <div className="flex items-center space-x-3">
                            <div className="p-2 rounded-full bg-purple-100 text-purple-600">
                                <Brain className="w-5 h-5" />
                            </div>
                            <div>
                                <p className="text-sm text-gray-600">Method</p>
                                <p className="font-medium">{status.processing_method}</p>
                            </div>
                        </div>

                        <div className="flex items-center space-x-3">
                            <div className="p-2 rounded-full bg-yellow-100 text-yellow-600">
                                <Clock className="w-5 h-5" />
                            </div>
                            <div>
                                <p className="text-sm text-gray-600">Duration</p>
                                <p className="font-medium">
                                    {getProcessingDuration() ? `${getProcessingDuration()}s` : 'N/A'}
                                </p>
                            </div>
                        </div>
                    </div>

                    {/* Progress Bar */}
                    <div className="mb-4">
                        <div className="flex justify-between text-sm text-gray-600 mb-2">
                            <span>Progress</span>
                            <span>{getProgressPercentage()}%</span>
                        </div>
                        <div className="w-full bg-gray-200 rounded-full h-2">
                            <div
                                className="bg-primary-600 h-2 rounded-full transition-all duration-300"
                                style={{ width: `${getProgressPercentage()}%` }}
                            ></div>
                        </div>
                    </div>

                    {/* Current Thread */}
                    {status.current_thread && (
                        <div className="bg-blue-50 p-3 rounded-lg">
                            <p className="text-sm text-blue-800">
                                <span className="font-medium">Currently processing:</span> {status.current_thread}
                            </p>
                        </div>
                    )}

                    {/* Errors */}
                    {status.errors.length > 0 && (
                        <div className="mt-4 bg-red-50 p-3 rounded-lg">
                            <p className="text-sm text-red-800 font-medium mb-2">Errors:</p>
                            <ul className="text-sm text-red-700 space-y-1">
                                {status.errors.slice(-5).map((error, index) => (
                                    <li key={index}>• {error}</li>
                                ))}
                            </ul>
                        </div>
                    )}
                </div>
            )}

            {/* Processing Logs */}
            <div className="bg-white rounded-lg shadow-sm p-6 border">
                <h2 className="text-xl font-semibold text-gray-900 mb-4 flex items-center">
                    <FileText className="w-5 h-5 mr-2" />
                    Processing Logs
                </h2>

                <div className="bg-gray-900 text-green-400 p-4 rounded-lg font-mono text-sm h-96 overflow-y-auto">
                    {logs.length > 0 ? (
                        logs.map((log, index) => (
                            <div key={index} className="mb-1">{log}</div>
                        ))
                    ) : (
                        <div className="text-gray-500">No logs yet. Start processing to see logs here.</div>
                    )}
                </div>
            </div>

            {/* Recent Results */}
            {status?.results && status.results.length > 0 && (
                <div className="bg-white rounded-lg shadow-sm p-6 border">
                    <h2 className="text-xl font-semibold text-gray-900 mb-4 flex items-center">
                        <CheckCircle className="w-5 h-5 mr-2" />
                        Recent Results
                    </h2>

                    <div className="space-y-3">
                        {status.results.slice(-5).map((result, index) => (
                            <div key={index} className="border rounded-lg p-4">
                                <div className="flex items-center justify-between mb-2">
                                    <h3 className="font-medium text-gray-900">{result.thread_id}</h3>
                                    <span className={`px-2 py-1 rounded text-xs ${result.intent === 'bug_report' ? 'bg-red-100 text-red-800' :
                                        result.intent === 'feature_request' ? 'bg-blue-100 text-blue-800' :
                                            result.intent === 'how_to_question' ? 'bg-green-100 text-green-800' :
                                                'bg-gray-100 text-gray-800'
                                        }`}>
                                        {result.intent}
                                    </span>
                                </div>
                                <p className="text-sm text-gray-600 mb-2">{result.summary}</p>
                                <div className="flex items-center space-x-4 text-xs text-gray-500">
                                    <span>Confidence: {(result.confidence * 100).toFixed(1)}%</span>
                                    <span>Duration: {result.duration}</span>
                                    <span>Messages: {result.message_count}</span>
                                    {result.sentiment && <span>Sentiment: {result.sentiment}</span>}
                                </div>
                            </div>
                        ))}
                    </div>
                </div>
            )}
        </div>
    )
}

export default ProcessingPage 