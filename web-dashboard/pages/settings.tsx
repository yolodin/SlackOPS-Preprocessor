import React, { useState } from 'react'
import {
    Save,
    RefreshCw,
    Bell,
    Database,
    Cpu,
    Shield,
    Monitor,
    Palette
} from 'lucide-react'

const SettingsPage: React.FC = () => {
    const [settings, setSettings] = useState({
        // Processing Settings
        defaultProcessingMethod: 'ml',
        defaultModelType: 'lightweight',
        maxThreads: 100,
        batchSize: 10,

        // Performance Settings
        enableGPU: true,
        maxMemoryUsage: 8,
        processingTimeout: 300,

        // Data Settings
        dataRetentionDays: 30,
        autoCleanup: true,
        backupEnabled: true,

        // UI Settings
        theme: 'light',
        autoRefresh: true,
        refreshInterval: 30,

        // Notifications
        emailNotifications: false,
        processingAlerts: true,
        errorAlerts: true
    })

    const handleSave = () => {
        // In a real app, this would save to the backend
        alert('Settings saved successfully!')
    }

    const handleReset = () => {
        if (confirm('Reset all settings to defaults?')) {
            // Reset to defaults
            alert('Settings reset to defaults')
        }
    }

    return (
        <div className="space-y-6">
            {/* Page Header */}
            <div className="flex justify-between items-center">
                <div>
                    <h1 className="text-3xl font-bold text-gray-900">Settings</h1>
                    <p className="text-gray-600 mt-2">
                        Configure system preferences and processing options
                    </p>
                </div>
                <div className="flex space-x-3">
                    <button
                        onClick={handleSave}
                        className="flex items-center px-4 py-2 bg-primary-600 text-white rounded-lg hover:bg-primary-700 transition-colors"
                    >
                        <Save className="w-4 h-4 mr-2" />
                        Save Changes
                    </button>
                    <button
                        onClick={handleReset}
                        className="flex items-center px-4 py-2 bg-gray-200 text-gray-700 rounded-lg hover:bg-gray-300 transition-colors"
                    >
                        <RefreshCw className="w-4 h-4 mr-2" />
                        Reset
                    </button>
                </div>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                {/* Processing Settings */}
                <div className="bg-white rounded-lg shadow-sm p-6 border">
                    <h2 className="text-xl font-semibold text-gray-900 mb-4 flex items-center">
                        <Cpu className="w-5 h-5 mr-2" />
                        Processing Settings
                    </h2>

                    <div className="space-y-4">
                        <div>
                            <label className="block text-sm font-medium text-gray-700 mb-2">
                                Default Processing Method
                            </label>
                            <select
                                value={settings.defaultProcessingMethod}
                                onChange={(e) => setSettings({ ...settings, defaultProcessingMethod: e.target.value })}
                                className="w-full p-2 border rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-primary-500"
                            >
                                <option value="ml">Machine Learning</option>
                                <option value="rule-based">Rule-based</option>
                            </select>
                        </div>

                        <div>
                            <label className="block text-sm font-medium text-gray-700 mb-2">
                                Default Model Type
                            </label>
                            <select
                                value={settings.defaultModelType}
                                onChange={(e) => setSettings({ ...settings, defaultModelType: e.target.value })}
                                className="w-full p-2 border rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-primary-500"
                            >
                                <option value="lightweight">Lightweight (Fast)</option>
                                <option value="full">Full Transformer (Accurate)</option>
                            </select>
                        </div>

                        <div>
                            <label className="block text-sm font-medium text-gray-700 mb-2">
                                Max Threads per Job
                            </label>
                            <input
                                type="number"
                                value={settings.maxThreads}
                                onChange={(e) => setSettings({ ...settings, maxThreads: parseInt(e.target.value) })}
                                className="w-full p-2 border rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-primary-500"
                            />
                        </div>

                        <div>
                            <label className="block text-sm font-medium text-gray-700 mb-2">
                                Batch Size
                            </label>
                            <input
                                type="number"
                                value={settings.batchSize}
                                onChange={(e) => setSettings({ ...settings, batchSize: parseInt(e.target.value) })}
                                className="w-full p-2 border rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-primary-500"
                            />
                        </div>
                    </div>
                </div>

                {/* Performance Settings */}
                <div className="bg-white rounded-lg shadow-sm p-6 border">
                    <h2 className="text-xl font-semibold text-gray-900 mb-4 flex items-center">
                        <Monitor className="w-5 h-5 mr-2" />
                        Performance Settings
                    </h2>

                    <div className="space-y-4">
                        <div className="flex items-center justify-between">
                            <div>
                                <label className="text-sm font-medium text-gray-700">Enable GPU Acceleration</label>
                                <p className="text-xs text-gray-500">Use GPU for faster processing</p>
                            </div>
                            <input
                                type="checkbox"
                                checked={settings.enableGPU}
                                onChange={(e) => setSettings({ ...settings, enableGPU: e.target.checked })}
                                className="h-4 w-4 text-primary-600 focus:ring-primary-500 border-gray-300 rounded"
                            />
                        </div>

                        <div>
                            <label className="block text-sm font-medium text-gray-700 mb-2">
                                Max Memory Usage (GB)
                            </label>
                            <input
                                type="number"
                                value={settings.maxMemoryUsage}
                                onChange={(e) => setSettings({ ...settings, maxMemoryUsage: parseInt(e.target.value) })}
                                className="w-full p-2 border rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-primary-500"
                            />
                        </div>

                        <div>
                            <label className="block text-sm font-medium text-gray-700 mb-2">
                                Processing Timeout (seconds)
                            </label>
                            <input
                                type="number"
                                value={settings.processingTimeout}
                                onChange={(e) => setSettings({ ...settings, processingTimeout: parseInt(e.target.value) })}
                                className="w-full p-2 border rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-primary-500"
                            />
                        </div>
                    </div>
                </div>

                {/* Data Management */}
                <div className="bg-white rounded-lg shadow-sm p-6 border">
                    <h2 className="text-xl font-semibold text-gray-900 mb-4 flex items-center">
                        <Database className="w-5 h-5 mr-2" />
                        Data Management
                    </h2>

                    <div className="space-y-4">
                        <div>
                            <label className="block text-sm font-medium text-gray-700 mb-2">
                                Data Retention (days)
                            </label>
                            <input
                                type="number"
                                value={settings.dataRetentionDays}
                                onChange={(e) => setSettings({ ...settings, dataRetentionDays: parseInt(e.target.value) })}
                                className="w-full p-2 border rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-primary-500"
                            />
                        </div>

                        <div className="flex items-center justify-between">
                            <div>
                                <label className="text-sm font-medium text-gray-700">Auto Cleanup</label>
                                <p className="text-xs text-gray-500">Automatically remove old files</p>
                            </div>
                            <input
                                type="checkbox"
                                checked={settings.autoCleanup}
                                onChange={(e) => setSettings({ ...settings, autoCleanup: e.target.checked })}
                                className="h-4 w-4 text-primary-600 focus:ring-primary-500 border-gray-300 rounded"
                            />
                        </div>

                        <div className="flex items-center justify-between">
                            <div>
                                <label className="text-sm font-medium text-gray-700">Backup Enabled</label>
                                <p className="text-xs text-gray-500">Create backups of processed data</p>
                            </div>
                            <input
                                type="checkbox"
                                checked={settings.backupEnabled}
                                onChange={(e) => setSettings({ ...settings, backupEnabled: e.target.checked })}
                                className="h-4 w-4 text-primary-600 focus:ring-primary-500 border-gray-300 rounded"
                            />
                        </div>
                    </div>
                </div>

                {/* User Interface */}
                <div className="bg-white rounded-lg shadow-sm p-6 border">
                    <h2 className="text-xl font-semibold text-gray-900 mb-4 flex items-center">
                        <Palette className="w-5 h-5 mr-2" />
                        User Interface
                    </h2>

                    <div className="space-y-4">
                        <div>
                            <label className="block text-sm font-medium text-gray-700 mb-2">
                                Theme
                            </label>
                            <select
                                value={settings.theme}
                                onChange={(e) => setSettings({ ...settings, theme: e.target.value })}
                                className="w-full p-2 border rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-primary-500"
                            >
                                <option value="light">Light</option>
                                <option value="dark">Dark</option>
                                <option value="auto">Auto</option>
                            </select>
                        </div>

                        <div className="flex items-center justify-between">
                            <div>
                                <label className="text-sm font-medium text-gray-700">Auto Refresh</label>
                                <p className="text-xs text-gray-500">Automatically refresh data</p>
                            </div>
                            <input
                                type="checkbox"
                                checked={settings.autoRefresh}
                                onChange={(e) => setSettings({ ...settings, autoRefresh: e.target.checked })}
                                className="h-4 w-4 text-primary-600 focus:ring-primary-500 border-gray-300 rounded"
                            />
                        </div>

                        <div>
                            <label className="block text-sm font-medium text-gray-700 mb-2">
                                Refresh Interval (seconds)
                            </label>
                            <input
                                type="number"
                                value={settings.refreshInterval}
                                onChange={(e) => setSettings({ ...settings, refreshInterval: parseInt(e.target.value) })}
                                disabled={!settings.autoRefresh}
                                className="w-full p-2 border rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-primary-500 disabled:opacity-50"
                            />
                        </div>
                    </div>
                </div>
            </div>

            {/* Notifications */}
            <div className="bg-white rounded-lg shadow-sm p-6 border">
                <h2 className="text-xl font-semibold text-gray-900 mb-4 flex items-center">
                    <Bell className="w-5 h-5 mr-2" />
                    Notifications
                </h2>

                <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                    <div className="flex items-center justify-between">
                        <div>
                            <label className="text-sm font-medium text-gray-700">Email Notifications</label>
                            <p className="text-xs text-gray-500">Send emails for major events</p>
                        </div>
                        <input
                            type="checkbox"
                            checked={settings.emailNotifications}
                            onChange={(e) => setSettings({ ...settings, emailNotifications: e.target.checked })}
                            className="h-4 w-4 text-primary-600 focus:ring-primary-500 border-gray-300 rounded"
                        />
                    </div>

                    <div className="flex items-center justify-between">
                        <div>
                            <label className="text-sm font-medium text-gray-700">Processing Alerts</label>
                            <p className="text-xs text-gray-500">Notify when processing completes</p>
                        </div>
                        <input
                            type="checkbox"
                            checked={settings.processingAlerts}
                            onChange={(e) => setSettings({ ...settings, processingAlerts: e.target.checked })}
                            className="h-4 w-4 text-primary-600 focus:ring-primary-500 border-gray-300 rounded"
                        />
                    </div>

                    <div className="flex items-center justify-between">
                        <div>
                            <label className="text-sm font-medium text-gray-700">Error Alerts</label>
                            <p className="text-xs text-gray-500">Notify when errors occur</p>
                        </div>
                        <input
                            type="checkbox"
                            checked={settings.errorAlerts}
                            onChange={(e) => setSettings({ ...settings, errorAlerts: e.target.checked })}
                            className="h-4 w-4 text-primary-600 focus:ring-primary-500 border-gray-300 rounded"
                        />
                    </div>
                </div>
            </div>

            {/* System Info */}
            <div className="bg-white rounded-lg shadow-sm p-6 border">
                <h2 className="text-xl font-semibold text-gray-900 mb-4 flex items-center">
                    <Shield className="w-5 h-5 mr-2" />
                    System Information
                </h2>

                <div className="grid grid-cols-1 md:grid-cols-3 gap-6 text-sm">
                    <div>
                        <span className="text-gray-500">Version:</span>
                        <span className="font-medium ml-2">1.0.0</span>
                    </div>
                    <div>
                        <span className="text-gray-500">Python Version:</span>
                        <span className="font-medium ml-2">3.11.4</span>
                    </div>
                    <div>
                        <span className="text-gray-500">Node.js Version:</span>
                        <span className="font-medium ml-2">20.11.1</span>
                    </div>
                    <div>
                        <span className="text-gray-500">API Status:</span>
                        <span className="font-medium ml-2 text-green-600">Online</span>
                    </div>
                    <div>
                        <span className="text-gray-500">ML Models:</span>
                        <span className="font-medium ml-2 text-green-600">Available</span>
                    </div>
                    <div>
                        <span className="text-gray-500">Database:</span>
                        <span className="font-medium ml-2 text-green-600">Connected</span>
                    </div>
                </div>
            </div>
        </div>
    )
}

export default SettingsPage 