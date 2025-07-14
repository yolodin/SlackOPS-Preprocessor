import React, { useState, useEffect } from 'react'
import {
    Brain,
    CheckCircle,
    AlertCircle,
    BarChart3,
    Clock,
    Cpu,
    Database,
    RefreshCw,
    Play
} from 'lucide-react'

interface ModelInfo {
    available: boolean
    models: {
        classification: {
            transformer: string
            lightweight: string
        }
        summarization: {
            abstractive: string
            extractive: string
        }
    }
}

const MLModelsPage: React.FC = () => {
    const [modelInfo, setModelInfo] = useState<ModelInfo | null>(null)
    const [loading, setLoading] = useState(true)

    useEffect(() => {
        fetchModelInfo()
    }, [])

    const fetchModelInfo = async () => {
        try {
            const response = await fetch('/api/python/api/ml/models')
            const data = await response.json()
            setModelInfo(data)
        } catch (error) {
            console.error('Error fetching model info:', error)
        } finally {
            setLoading(false)
        }
    }

    if (loading) {
        return (
            <div className="flex items-center justify-center h-64">
                <div className="loading-spinner"></div>
                <span className="ml-3 text-gray-600">Loading ML models...</span>
            </div>
        )
    }

    return (
        <div className="space-y-6">
            {/* Page Header */}
            <div className="flex justify-between items-center">
                <div>
                    <h1 className="text-3xl font-bold text-gray-900">ML Models</h1>
                    <p className="text-gray-600 mt-2">
                        Machine learning model status and performance
                    </p>
                </div>
                <div className="flex space-x-3">
                    <button className="flex items-center px-4 py-2 bg-primary-600 text-white rounded-lg hover:bg-primary-700 transition-colors">
                        <Play className="w-4 h-4 mr-2" />
                        Train Models
                    </button>
                    <button
                        onClick={() => window.location.reload()}
                        className="flex items-center px-4 py-2 bg-gray-200 text-gray-700 rounded-lg hover:bg-gray-300 transition-colors"
                    >
                        <RefreshCw className="w-4 h-4 mr-2" />
                        Refresh
                    </button>
                </div>
            </div>

            {/* ML Status Overview */}
            <div className="bg-white rounded-lg shadow-sm p-6 border">
                <div className="flex items-center justify-between mb-4">
                    <h2 className="text-xl font-semibold text-gray-900 flex items-center">
                        <Brain className="w-5 h-5 mr-2" />
                        System Status
                    </h2>
                    <div className={`flex items-center px-3 py-1 rounded-full text-sm ${modelInfo?.available
                            ? 'bg-green-100 text-green-800'
                            : 'bg-red-100 text-red-800'
                        }`}>
                        {modelInfo?.available ? (
                            <CheckCircle className="w-4 h-4 mr-1" />
                        ) : (
                            <AlertCircle className="w-4 h-4 mr-1" />
                        )}
                        {modelInfo?.available ? 'All Models Available' : 'Models Unavailable'}
                    </div>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                    <div className="text-center p-4 bg-green-50 rounded-lg">
                        <Cpu className="w-8 h-8 mx-auto text-green-600 mb-2" />
                        <p className="text-2xl font-bold text-green-900">Active</p>
                        <p className="text-sm text-green-700">Classification</p>
                    </div>
                    <div className="text-center p-4 bg-blue-50 rounded-lg">
                        <Database className="w-8 h-8 mx-auto text-blue-600 mb-2" />
                        <p className="text-2xl font-bold text-blue-900">Active</p>
                        <p className="text-sm text-blue-700">Summarization</p>
                    </div>
                    <div className="text-center p-4 bg-purple-50 rounded-lg">
                        <BarChart3 className="w-8 h-8 mx-auto text-purple-600 mb-2" />
                        <p className="text-2xl font-bold text-purple-900">Active</p>
                        <p className="text-sm text-purple-700">Analytics</p>
                    </div>
                </div>
            </div>

            {/* Model Details */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                {/* Classification Models */}
                <div className="bg-white rounded-lg shadow-sm p-6 border">
                    <h2 className="text-xl font-semibold text-gray-900 mb-4">Classification Models</h2>

                    <div className="space-y-4">
                        <div className="p-4 border rounded-lg">
                            <div className="flex items-center justify-between mb-2">
                                <h3 className="font-medium text-gray-900">DistilBERT (Transformer)</h3>
                                <span className="px-2 py-1 bg-green-100 text-green-800 rounded text-sm">Active</span>
                            </div>
                            <p className="text-sm text-gray-600 mb-3">
                                High-accuracy transformer model for intent classification
                            </p>
                            <div className="grid grid-cols-2 gap-4 text-sm">
                                <div>
                                    <span className="text-gray-500">Accuracy:</span>
                                    <span className="font-medium ml-2">90-95%</span>
                                </div>
                                <div>
                                    <span className="text-gray-500">Speed:</span>
                                    <span className="font-medium ml-2">Moderate</span>
                                </div>
                                <div>
                                    <span className="text-gray-500">Memory:</span>
                                    <span className="font-medium ml-2">High</span>
                                </div>
                                <div>
                                    <span className="text-gray-500">Parameters:</span>
                                    <span className="font-medium ml-2">66M</span>
                                </div>
                            </div>
                        </div>

                        <div className="p-4 border rounded-lg">
                            <div className="flex items-center justify-between mb-2">
                                <h3 className="font-medium text-gray-900">MiniLM + Random Forest</h3>
                                <span className="px-2 py-1 bg-green-100 text-green-800 rounded text-sm">Active</span>
                            </div>
                            <p className="text-sm text-gray-600 mb-3">
                                Lightweight model optimized for speed and efficiency
                            </p>
                            <div className="grid grid-cols-2 gap-4 text-sm">
                                <div>
                                    <span className="text-gray-500">Accuracy:</span>
                                    <span className="font-medium ml-2">85-90%</span>
                                </div>
                                <div>
                                    <span className="text-gray-500">Speed:</span>
                                    <span className="font-medium ml-2">Very Fast</span>
                                </div>
                                <div>
                                    <span className="text-gray-500">Memory:</span>
                                    <span className="font-medium ml-2">Low</span>
                                </div>
                                <div>
                                    <span className="text-gray-500">Parameters:</span>
                                    <span className="font-medium ml-2">22M</span>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>

                {/* Summarization Models */}
                <div className="bg-white rounded-lg shadow-sm p-6 border">
                    <h2 className="text-xl font-semibold text-gray-900 mb-4">Summarization Models</h2>

                    <div className="space-y-4">
                        <div className="p-4 border rounded-lg">
                            <div className="flex items-center justify-between mb-2">
                                <h3 className="font-medium text-gray-900">BART-large-CNN</h3>
                                <span className="px-2 py-1 bg-green-100 text-green-800 rounded text-sm">Active</span>
                            </div>
                            <p className="text-sm text-gray-600 mb-3">
                                Abstractive summarization generating human-like summaries
                            </p>
                            <div className="grid grid-cols-2 gap-4 text-sm">
                                <div>
                                    <span className="text-gray-500">Type:</span>
                                    <span className="font-medium ml-2">Abstractive</span>
                                </div>
                                <div>
                                    <span className="text-gray-500">Quality:</span>
                                    <span className="font-medium ml-2">High</span>
                                </div>
                                <div>
                                    <span className="text-gray-500">Speed:</span>
                                    <span className="font-medium ml-2">Moderate</span>
                                </div>
                                <div>
                                    <span className="text-gray-500">Parameters:</span>
                                    <span className="font-medium ml-2">400M</span>
                                </div>
                            </div>
                        </div>

                        <div className="p-4 border rounded-lg">
                            <div className="flex items-center justify-between mb-2">
                                <h3 className="font-medium text-gray-900">Sentence Embeddings</h3>
                                <span className="px-2 py-1 bg-green-100 text-green-800 rounded text-sm">Active</span>
                            </div>
                            <p className="text-sm text-gray-600 mb-3">
                                Extractive summarization selecting key sentences
                            </p>
                            <div className="grid grid-cols-2 gap-4 text-sm">
                                <div>
                                    <span className="text-gray-500">Type:</span>
                                    <span className="font-medium ml-2">Extractive</span>
                                </div>
                                <div>
                                    <span className="text-gray-500">Quality:</span>
                                    <span className="font-medium ml-2">Good</span>
                                </div>
                                <div>
                                    <span className="text-gray-500">Speed:</span>
                                    <span className="font-medium ml-2">Fast</span>
                                </div>
                                <div>
                                    <span className="text-gray-500">Parameters:</span>
                                    <span className="font-medium ml-2">22M</span>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>

            {/* Performance Metrics */}
            <div className="bg-white rounded-lg shadow-sm p-6 border">
                <h2 className="text-xl font-semibold text-gray-900 mb-4">Performance Metrics</h2>

                <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                    <div className="text-center p-4 bg-gray-50 rounded-lg">
                        <Clock className="w-6 h-6 mx-auto text-gray-600 mb-2" />
                        <p className="text-2xl font-bold text-gray-900">0.5s</p>
                        <p className="text-sm text-gray-600">Avg Processing Time</p>
                    </div>
                    <div className="text-center p-4 bg-gray-50 rounded-lg">
                        <BarChart3 className="w-6 h-6 mx-auto text-gray-600 mb-2" />
                        <p className="text-2xl font-bold text-gray-900">92%</p>
                        <p className="text-sm text-gray-600">Overall Accuracy</p>
                    </div>
                    <div className="text-center p-4 bg-gray-50 rounded-lg">
                        <Cpu className="w-6 h-6 mx-auto text-gray-600 mb-2" />
                        <p className="text-2xl font-bold text-gray-900">45%</p>
                        <p className="text-sm text-gray-600">GPU Utilization</p>
                    </div>
                </div>
            </div>

            {/* Model Comparison */}
            <div className="bg-white rounded-lg shadow-sm p-6 border">
                <h2 className="text-xl font-semibold text-gray-900 mb-4">Model Comparison</h2>

                <div className="overflow-x-auto">
                    <table className="w-full text-sm">
                        <thead>
                            <tr className="border-b">
                                <th className="text-left py-3 px-4">Model</th>
                                <th className="text-left py-3 px-4">Type</th>
                                <th className="text-left py-3 px-4">Accuracy</th>
                                <th className="text-left py-3 px-4">Speed</th>
                                <th className="text-left py-3 px-4">Memory</th>
                                <th className="text-left py-3 px-4">Best For</th>
                            </tr>
                        </thead>
                        <tbody className="divide-y">
                            <tr>
                                <td className="py-3 px-4 font-medium">DistilBERT</td>
                                <td className="py-3 px-4">Classification</td>
                                <td className="py-3 px-4">
                                    <span className="px-2 py-1 bg-green-100 text-green-800 rounded text-xs">95%</span>
                                </td>
                                <td className="py-3 px-4">Moderate</td>
                                <td className="py-3 px-4">High</td>
                                <td className="py-3 px-4">High accuracy needs</td>
                            </tr>
                            <tr>
                                <td className="py-3 px-4 font-medium">MiniLM + RF</td>
                                <td className="py-3 px-4">Classification</td>
                                <td className="py-3 px-4">
                                    <span className="px-2 py-1 bg-yellow-100 text-yellow-800 rounded text-xs">88%</span>
                                </td>
                                <td className="py-3 px-4">Fast</td>
                                <td className="py-3 px-4">Low</td>
                                <td className="py-3 px-4">Real-time processing</td>
                            </tr>
                            <tr>
                                <td className="py-3 px-4 font-medium">BART</td>
                                <td className="py-3 px-4">Summarization</td>
                                <td className="py-3 px-4">
                                    <span className="px-2 py-1 bg-green-100 text-green-800 rounded text-xs">High</span>
                                </td>
                                <td className="py-3 px-4">Moderate</td>
                                <td className="py-3 px-4">High</td>
                                <td className="py-3 px-4">Quality summaries</td>
                            </tr>
                            <tr>
                                <td className="py-3 px-4 font-medium">Extractive</td>
                                <td className="py-3 px-4">Summarization</td>
                                <td className="py-3 px-4">
                                    <span className="px-2 py-1 bg-blue-100 text-blue-800 rounded text-xs">Good</span>
                                </td>
                                <td className="py-3 px-4">Fast</td>
                                <td className="py-3 px-4">Low</td>
                                <td className="py-3 px-4">Quick insights</td>
                            </tr>
                        </tbody>
                    </table>
                </div>
            </div>
        </div>
    )
}

export default MLModelsPage 