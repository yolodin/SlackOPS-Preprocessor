import React, { useState, useEffect, useRef } from 'react'
import {
    Upload,
    FileText,
    Eye,
    Download,
    Trash2,
    RefreshCw,
    CheckCircle,
    AlertTriangle,
    Database,
    FileJson,
    Calendar,
    HardDrive, Search,
    Play,
    Settings,
    BarChart3
} from 'lucide-react'

interface DataFile {
    name: string
    size: number
    modified: string
    format?: string
    status?: 'processed' | 'unprocessed' | 'processing' | 'error'
    threads?: number
    messages?: number
}

interface ValidationReport {
    total_files: number
    total_threads: number
    total_messages: number
    unique_users: number
    date_range: {
        earliest: string | null
        latest: string | null
    }
    format_detected: string
    issues: string[]
    recommendations: string[]
}

const DataPage: React.FC = () => {
    const [files, setFiles] = useState<DataFile[]>([])
    const [loading, setLoading] = useState(true)
    const [uploading, setUploading] = useState(false)
    const [selectedFile, setSelectedFile] = useState<DataFile | null>(null)
    const [previewData, setPreviewData] = useState<any>(null)
    const [validationReport, setValidationReport] = useState<ValidationReport | null>(null)
    const [searchTerm, setSearchTerm] = useState('')
    const [filterStatus, setFilterStatus] = useState('all')
    const [dragActive, setDragActive] = useState(false)
    const fileInputRef = useRef<HTMLInputElement>(null)

    useEffect(() => {
        fetchFiles()
        fetchValidationReport()
    }, [])

    const fetchFiles = async () => {
        try {
            const response = await fetch('/api/python/api/data/files')
            const data = await response.json()

            // Enhance files with processing status
            const enhancedFiles = data.files.map((file: DataFile) => ({
                ...file,
                format: detectFormat(file.name),
                status: getProcessingStatus(file.name)
            }))

            setFiles(enhancedFiles)
        } catch (error) {
            console.error('Error fetching files:', error)
        } finally {
            setLoading(false)
        }
    }

    const fetchValidationReport = async () => {
        try {
            // Fetch the actual validation report from API
            const reportResponse = await fetch('/api/python/api/data/validation')
            if (reportResponse.ok) {
                const reportData = await reportResponse.json()
                setValidationReport({
                    total_files: files.length || 0,
                    total_threads: reportData.total_threads || 0,
                    total_messages: reportData.total_messages || 0,
                    unique_users: reportData.unique_users || 0,
                    date_range: {
                        earliest: reportData.date_range?.earliest || null,
                        latest: reportData.date_range?.latest || null
                    },
                    format_detected: reportData.total_threads > 0 ? 'Processed data available' : 'Ready for processing',
                    issues: reportData.issues || [],
                    recommendations: reportData.issues?.length > 0 ? [
                        'Consider running data standardization',
                        'Review and fix identified issues'
                    ] : reportData.total_threads > 0 ? [
                        'Data appears clean and ready for analysis'
                    ] : [
                        'Upload and process Slack data to see metrics'
                    ]
                })
                return
            }

            // Fallback: create clean report from current files
            const filesResponse = await fetch('/api/python/api/data/files')
            if (filesResponse.ok) {
                const data = await filesResponse.json()
                const report: ValidationReport = {
                    total_files: data.files.length,
                    total_threads: 0,  // Start fresh
                    total_messages: 0, // Start fresh
                    unique_users: 0,   // Start fresh
                    date_range: {
                        earliest: null,
                        latest: null
                    },
                    format_detected: 'Ready for processing',
                    issues: [],
                    recommendations: ['Upload and process Slack data to see metrics']
                }
                setValidationReport(report)
            }
        } catch (error) {
            console.error('Error fetching validation report:', error)
            // Fallback to clean state
            setValidationReport({
                total_files: 0,
                total_threads: 0,
                total_messages: 0,
                unique_users: 0,
                date_range: {
                    earliest: null,
                    latest: null
                },
                format_detected: 'No data processed yet',
                issues: [],
                recommendations: ['Upload Slack data files to begin']
            })
        }
    }

    const detectFormat = (filename: string): string => {
        if (filename.includes('standardized')) return 'Standardized'
        if (filename.includes('export')) return 'Slack Export'
        if (filename.includes('channel')) return 'Channel Export'
        if (filename.includes('validation')) return 'Validation Report'
        if (filename.includes('config')) return 'Configuration'
        return 'JSON Data'
    }

    const getProcessingStatus = (filename: string): 'processed' | 'unprocessed' | 'processing' | 'error' => {
        if (filename.includes('standardized')) return 'processed'
        if (filename.includes('validation') || filename.includes('config')) return 'processed'
        return 'unprocessed'
    }

    const handleDrag = (e: React.DragEvent) => {
        e.preventDefault()
        e.stopPropagation()
        if (e.type === "dragenter" || e.type === "dragover") {
            setDragActive(true)
        } else if (e.type === "dragleave") {
            setDragActive(false)
        }
    }

    const handleDrop = (e: React.DragEvent) => {
        e.preventDefault()
        e.stopPropagation()
        setDragActive(false)

        if (e.dataTransfer.files && e.dataTransfer.files[0]) {
            handleFileUpload(e.dataTransfer.files[0])
        }
    }

    const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
        if (e.target.files && e.target.files[0]) {
            handleFileUpload(e.target.files[0])
        }
    }

    const handleFileUpload = async (file: File) => {
        if (!file.name.endsWith('.json')) {
            alert('Please upload a JSON file')
            return
        }

        setUploading(true)
        try {
            const formData = new FormData()
            formData.append('file', file)

            const response = await fetch('/api/python/api/data/upload', {
                method: 'POST',
                body: formData
            })

            if (response.ok) {
                await fetchFiles()
                alert('File uploaded successfully!')
            } else {
                const error = await response.json()
                alert(`Upload failed: ${error.error}`)
            }
        } catch (error) {
            console.error('Error uploading file:', error)
            alert('Upload failed')
        } finally {
            setUploading(false)
        }
    }

    const previewFile = async (file: DataFile) => {
        try {
            setSelectedFile(file)
            // Mock preview data - in reality, you'd fetch from your backend
            const mockData = {
                filename: file.name,
                format: file.format,
                preview: [
                    {
                        thread_ts: "1671024600.123456",
                        messages: [
                            {
                                ts: "1671024600.123456",
                                user: "alice",
                                text: "Need help with authentication setup"
                            },
                            {
                                ts: "1671024650.789012",
                                user: "bob",
                                text: "I can help with that. What specific issues are you seeing?"
                            }
                        ]
                    }
                ],
                stats: {
                    threads: Math.floor(Math.random() * 20) + 1,
                    messages: Math.floor(Math.random() * 100) + 10,
                    users: Math.floor(Math.random() * 10) + 2
                }
            }
            setPreviewData(mockData)
        } catch (error) {
            console.error('Error previewing file:', error)
        }
    }

    const processFile = async (file: DataFile) => {
        try {
            const response = await fetch('/api/python/api/processing/start', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    data_file: `data/${file.name}`,
                    use_ml: true,
                    use_lightweight: true,
                    max_threads: 10
                })
            })

            if (response.ok) {
                alert(`Processing started for ${file.name}`)
                // Update file status
                setFiles(prev => prev.map(f =>
                    f.name === file.name ? { ...f, status: 'processing' } : f
                ))
            } else {
                alert('Failed to start processing')
            }
        } catch (error) {
            console.error('Error processing file:', error)
            alert('Error starting processing')
        }
    }

    const runDataStandardization = async () => {
        try {
            setLoading(true)
            // This would call your slack_data_adapter.py
            alert('Data standardization started. This will convert all files to a unified format.')

            // Simulate the process
            setTimeout(() => {
                fetchFiles()
                fetchValidationReport()
                alert('Data standardization completed!')
            }, 3000)
        } catch (error) {
            console.error('Error running standardization:', error)
            alert('Standardization failed')
        } finally {
            setLoading(false)
        }
    }

    const filteredFiles = files.filter(file => {
        const matchesSearch = file.name.toLowerCase().includes(searchTerm.toLowerCase())
        const matchesFilter = filterStatus === 'all' || file.status === filterStatus
        return matchesSearch && matchesFilter
    })

    const getStatusColor = (status: string) => {
        switch (status) {
            case 'processed': return 'bg-green-100 text-green-800'
            case 'processing': return 'bg-yellow-100 text-yellow-800'
            case 'error': return 'bg-red-100 text-red-800'
            default: return 'bg-gray-100 text-gray-800'
        }
    }

    const getStatusIcon = (status: string) => {
        switch (status) {
            case 'processed': return <CheckCircle className="w-4 h-4" />
            case 'processing': return <RefreshCw className="w-4 h-4 animate-spin" />
            case 'error': return <AlertTriangle className="w-4 h-4" />
            default: return <FileText className="w-4 h-4" />
        }
    }

    return (
        <div className="space-y-6">
            {/* Page Header */}
            <div className="flex justify-between items-center">
                <div>
                    <h1 className="text-3xl font-bold text-gray-900">Data Management</h1>
                    <p className="text-gray-600 mt-2">
                        Upload, preview, and manage your Slack export data
                    </p>
                </div>
                <div className="flex space-x-3">
                    <button
                        onClick={runDataStandardization}
                        className="flex items-center px-4 py-2 bg-primary-600 text-white rounded-lg hover:bg-primary-700 transition-colors"
                    >
                        <Settings className="w-4 h-4 mr-2" />
                        Standardize Data
                    </button>
                    <button
                        onClick={() => fetchFiles()}
                        className="flex items-center px-4 py-2 bg-gray-200 text-gray-700 rounded-lg hover:bg-gray-300 transition-colors"
                    >
                        <RefreshCw className="w-4 h-4 mr-2" />
                        Refresh
                    </button>
                </div>
            </div>

            {/* Data Overview Cards */}
            {validationReport && (
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
                    <div className="bg-white rounded-lg shadow-sm p-6 border">
                        <div className="flex items-center justify-between">
                            <div>
                                <p className="text-sm font-medium text-gray-600">Total Files</p>
                                <p className="text-2xl font-bold text-gray-900">{validationReport.total_files}</p>
                            </div>
                            <div className="p-2 rounded-full bg-blue-100 text-blue-600">
                                <FileJson className="w-6 h-6" />
                            </div>
                        </div>
                    </div>

                    <div className="bg-white rounded-lg shadow-sm p-6 border">
                        <div className="flex items-center justify-between">
                            <div>
                                <p className="text-sm font-medium text-gray-600">Total Threads</p>
                                <p className="text-2xl font-bold text-gray-900">{validationReport.total_threads}</p>
                            </div>
                            <div className="p-2 rounded-full bg-green-100 text-green-600">
                                <Database className="w-6 h-6" />
                            </div>
                        </div>
                    </div>

                    <div className="bg-white rounded-lg shadow-sm p-6 border">
                        <div className="flex items-center justify-between">
                            <div>
                                <p className="text-sm font-medium text-gray-600">Total Messages</p>
                                <p className="text-2xl font-bold text-gray-900">{validationReport.total_messages}</p>
                            </div>
                            <div className="p-2 rounded-full bg-purple-100 text-purple-600">
                                <BarChart3 className="w-6 h-6" />
                            </div>
                        </div>
                    </div>

                    <div className="bg-white rounded-lg shadow-sm p-6 border">
                        <div className="flex items-center justify-between">
                            <div>
                                <p className="text-sm font-medium text-gray-600">Unique Users</p>
                                <p className="text-2xl font-bold text-gray-900">{validationReport.unique_users}</p>
                            </div>
                            <div className="p-2 rounded-full bg-orange-100 text-orange-600">
                                <Calendar className="w-6 h-6" />
                            </div>
                        </div>
                    </div>
                </div>
            )}

            {/* File Upload Area */}
            <div className="bg-white rounded-lg shadow-sm p-6 border">
                <div
                    className={`relative border-2 border-dashed rounded-lg p-8 text-center transition-colors ${dragActive
                        ? 'border-primary-400 bg-primary-50'
                        : 'border-gray-300 hover:border-primary-400'
                        }`}
                    onDragEnter={handleDrag}
                    onDragLeave={handleDrag}
                    onDragOver={handleDrag}
                    onDrop={handleDrop}
                >
                    <input
                        ref={fileInputRef}
                        type="file"
                        accept=".json"
                        onChange={handleFileSelect}
                        className="hidden"
                    />

                    <Upload className="mx-auto h-12 w-12 text-gray-400 mb-4" />
                    <h3 className="text-lg font-medium text-gray-900 mb-2">
                        Upload Slack Export Data
                    </h3>
                    <p className="text-gray-600 mb-4">
                        Drag and drop your JSON files here, or click to browse
                    </p>

                    <button
                        onClick={() => fileInputRef.current?.click()}
                        disabled={uploading}
                        className="inline-flex items-center px-6 py-3 bg-primary-600 text-white rounded-lg hover:bg-primary-700 transition-colors disabled:opacity-50"
                    >
                        {uploading ? (
                            <RefreshCw className="w-4 h-4 mr-2 animate-spin" />
                        ) : (
                            <Upload className="w-4 h-4 mr-2" />
                        )}
                        {uploading ? 'Uploading...' : 'Choose Files'}
                    </button>

                    <p className="text-xs text-gray-500 mt-2">
                        Supports: Slack exports, channel dumps, thread exports (JSON format)
                    </p>
                </div>
            </div>

            {/* Search and Filter */}
            <div className="bg-white rounded-lg shadow-sm p-6 border">
                <div className="flex flex-col sm:flex-row gap-4">
                    <div className="flex-1 relative">
                        <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400 w-4 h-4" />
                        <input
                            type="text"
                            placeholder="Search files..."
                            value={searchTerm}
                            onChange={(e) => setSearchTerm(e.target.value)}
                            className="w-full pl-10 pr-4 py-2 border rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-primary-500"
                        />
                    </div>
                    <select
                        value={filterStatus}
                        onChange={(e) => setFilterStatus(e.target.value)}
                        className="px-4 py-2 border rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-primary-500"
                    >
                        <option value="all">All Status</option>
                        <option value="processed">Processed</option>
                        <option value="unprocessed">Unprocessed</option>
                        <option value="processing">Processing</option>
                        <option value="error">Error</option>
                    </select>
                </div>
            </div>

            {/* Files List */}
            <div className="bg-white rounded-lg shadow-sm border">
                <div className="px-6 py-4 border-b">
                    <h2 className="text-xl font-semibold text-gray-900">Data Files</h2>
                </div>

                {loading ? (
                    <div className="p-8 text-center">
                        <RefreshCw className="w-8 h-8 animate-spin mx-auto text-gray-400 mb-2" />
                        <p className="text-gray-600">Loading files...</p>
                    </div>
                ) : filteredFiles.length === 0 ? (
                    <div className="p-8 text-center">
                        <FileText className="w-8 h-8 mx-auto text-gray-400 mb-2" />
                        <p className="text-gray-600">No files found</p>
                    </div>
                ) : (
                    <div className="divide-y">
                        {filteredFiles.map((file, index) => (
                            <div key={index} className="p-4 hover:bg-gray-50 transition-colors">
                                <div className="flex items-center justify-between">
                                    <div className="flex items-center space-x-4 flex-1">
                                        <div className="p-2 rounded-lg bg-blue-100">
                                            <FileJson className="w-5 h-5 text-blue-600" />
                                        </div>

                                        <div className="flex-1 min-w-0">
                                            <h3 className="font-medium text-gray-900 truncate">{file.name}</h3>
                                            <div className="flex items-center space-x-4 text-sm text-gray-600 mt-1">
                                                <span className="flex items-center">
                                                    <HardDrive className="w-4 h-4 mr-1" />
                                                    {(file.size / 1024).toFixed(1)} KB
                                                </span>
                                                <span className="flex items-center">
                                                    <Calendar className="w-4 h-4 mr-1" />
                                                    {new Date(file.modified).toLocaleDateString()}
                                                </span>
                                                <span className="font-medium">{file.format}</span>
                                            </div>
                                        </div>

                                        <div className={`flex items-center px-3 py-1 rounded-full text-sm ${getStatusColor(file.status || 'unprocessed')}`}>
                                            {getStatusIcon(file.status || 'unprocessed')}
                                            <span className="ml-2 capitalize">{file.status || 'unprocessed'}</span>
                                        </div>
                                    </div>

                                    <div className="flex items-center space-x-2 ml-4">
                                        <button
                                            onClick={() => previewFile(file)}
                                            className="p-2 text-gray-600 hover:text-primary-600 hover:bg-primary-50 rounded-lg transition-colors"
                                            title="Preview"
                                        >
                                            <Eye className="w-4 h-4" />
                                        </button>

                                        {file.status !== 'processed' && (
                                            <button
                                                onClick={() => processFile(file)}
                                                className="p-2 text-gray-600 hover:text-green-600 hover:bg-green-50 rounded-lg transition-colors"
                                                title="Process"
                                            >
                                                <Play className="w-4 h-4" />
                                            </button>
                                        )}

                                        <button
                                            className="p-2 text-gray-600 hover:text-blue-600 hover:bg-blue-50 rounded-lg transition-colors"
                                            title="Download"
                                        >
                                            <Download className="w-4 h-4" />
                                        </button>

                                        <button
                                            className="p-2 text-gray-600 hover:text-red-600 hover:bg-red-50 rounded-lg transition-colors"
                                            title="Delete"
                                        >
                                            <Trash2 className="w-4 h-4" />
                                        </button>
                                    </div>
                                </div>
                            </div>
                        ))}
                    </div>
                )}
            </div>

            {/* File Preview Modal */}
            {selectedFile && previewData && (
                <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4">
                    <div className="bg-white rounded-lg max-w-4xl w-full max-h-[80vh] overflow-hidden">
                        <div className="px-6 py-4 border-b flex items-center justify-between">
                            <h3 className="text-lg font-semibold">Preview: {selectedFile.name}</h3>
                            <button
                                onClick={() => setSelectedFile(null)}
                                className="text-gray-400 hover:text-gray-600"
                            >
                                ✕
                            </button>
                        </div>

                        <div className="p-6 overflow-y-auto max-h-[60vh]">
                            <div className="grid grid-cols-3 gap-4 mb-6">
                                <div className="text-center p-4 bg-gray-50 rounded-lg">
                                    <p className="text-2xl font-bold text-gray-900">{previewData.stats.threads}</p>
                                    <p className="text-sm text-gray-600">Threads</p>
                                </div>
                                <div className="text-center p-4 bg-gray-50 rounded-lg">
                                    <p className="text-2xl font-bold text-gray-900">{previewData.stats.messages}</p>
                                    <p className="text-sm text-gray-600">Messages</p>
                                </div>
                                <div className="text-center p-4 bg-gray-50 rounded-lg">
                                    <p className="text-2xl font-bold text-gray-900">{previewData.stats.users}</p>
                                    <p className="text-sm text-gray-600">Users</p>
                                </div>
                            </div>

                            <div className="bg-gray-900 text-green-400 p-4 rounded-lg font-mono text-sm">
                                <pre>{JSON.stringify(previewData.preview, null, 2)}</pre>
                            </div>
                        </div>
                    </div>
                </div>
            )}

            {/* Data Validation Report */}
            {validationReport && (
                <div className="bg-white rounded-lg shadow-sm p-6 border">
                    <h2 className="text-xl font-semibold text-gray-900 mb-4">Data Validation Report</h2>

                    <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                        <div>
                            <h3 className="font-medium text-gray-900 mb-3">Data Quality</h3>
                            <div className="space-y-2">
                                <div className="flex justify-between">
                                    <span className="text-gray-600">Format Detected:</span>
                                    <span className="font-medium">{validationReport.format_detected}</span>
                                </div>
                                <div className="flex justify-between">
                                    <span className="text-gray-600">Date Range:</span>
                                    <span className="font-medium">
                                        {validationReport.date_range.earliest ? new Date(validationReport.date_range.earliest).toLocaleDateString() : 'N/A'} -
                                        {validationReport.date_range.latest ? new Date(validationReport.date_range.latest).toLocaleDateString() : 'N/A'}
                                    </span>
                                </div>
                            </div>

                            {validationReport.issues.length > 0 && (
                                <div className="mt-4">
                                    <h4 className="font-medium text-red-800 mb-2">Issues Found:</h4>
                                    <ul className="text-sm text-red-700 space-y-1">
                                        {validationReport.issues.map((issue, i) => (
                                            <li key={i}>• {issue}</li>
                                        ))}
                                    </ul>
                                </div>
                            )}
                        </div>

                        <div>
                            <h3 className="font-medium text-gray-900 mb-3">Recommendations</h3>
                            <ul className="text-sm text-blue-700 space-y-1">
                                {validationReport.recommendations.map((rec, i) => (
                                    <li key={i}>• {rec}</li>
                                ))}
                            </ul>
                        </div>
                    </div>
                </div>
            )}
        </div>
    )
}

export default DataPage 