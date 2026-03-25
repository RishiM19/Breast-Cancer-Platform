import { BarChart3, Bell, Lock, User, Globe } from 'lucide-react';
import { motion } from 'motion/react';
import { useState } from 'react';

interface SettingsProps {
  onNavigate: (screen: string) => void;
  userName: string;
  onUpdateUserName: (name: string) => void;
}

export function Settings({ onNavigate, userName, onUpdateUserName }: SettingsProps) {
  const [notificationsEnabled, setNotificationsEnabled] = useState(true);
  const [autoAnalysis, setAutoAnalysis] = useState(false);
  const [confidenceThreshold, setConfidenceThreshold] = useState(85);
  const [localUserName, setLocalUserName] = useState(userName);

  const getInitials = (name: string) => {
    const parts = name.split(' ');
    if (parts.length >= 2) {
      return (parts[0][0] + parts[parts.length - 1][0]).toUpperCase();
    }
    return name.substring(0, 2).toUpperCase();
  };

  const handleSave = () => {
    onUpdateUserName(localUserName);
    alert('Settings saved successfully!');
    onNavigate('dashboard');
  };

  const handleCancel = () => {
    setLocalUserName(userName);
    onNavigate('dashboard');
  };

  return (
    <div className="p-8 max-w-4xl">
          {/* Header */}
          <motion.div 
            className="mb-8"
            initial={{ y: -20, opacity: 0 }}
            animate={{ y: 0, opacity: 1 }}
            transition={{ duration: 0.5, delay: 0.1 }}
          >
            <h1 className="text-gray-900 dark:text-white mb-1">Settings</h1>
            <p className="text-gray-500 dark:text-gray-400">Manage your application preferences</p>
          </motion.div>

          {/* Settings Sections */}
          <motion.div 
            className="space-y-6"
            initial={{ y: 20, opacity: 0 }}
            animate={{ y: 0, opacity: 1 }}
            transition={{ duration: 0.5, delay: 0.2 }}
          >
            {/* Profile Settings */}
            <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm border border-gray-100 dark:border-gray-700 p-6">
              <div className="flex items-center gap-3 mb-6">
                <div className="w-10 h-10 bg-blue-100 dark:bg-blue-900/40 rounded-lg flex items-center justify-center">
                  <User className="w-5 h-5 text-[#007AFF]" />
                </div>
                <div>
                  <h3 className="text-gray-900 dark:text-white">Profile Settings</h3>
                  <p className="text-sm text-gray-500 dark:text-gray-400">Update your personal information</p>
                </div>
              </div>

              <div className="space-y-4">
                <div>
                  <label className="block text-gray-700 dark:text-gray-300 mb-2 text-sm">Full Name</label>
                  <input
                    type="text"
                    value={localUserName}
                    onChange={(e) => setLocalUserName(e.target.value)}
                    className="w-full px-4 py-2 border border-gray-200 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 dark:text-white focus:outline-none focus:ring-2 focus:ring-[#007AFF] focus:border-transparent transition-all"
                  />
                </div>
                <div>
                  <label className="block text-gray-700 dark:text-gray-300 mb-2 text-sm">Email</label>
                  <input
                    type="email"
                    defaultValue="demo@lumina.ai"
                    className="w-full px-4 py-2 border border-gray-200 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 dark:text-white focus:outline-none focus:ring-2 focus:ring-[#007AFF] focus:border-transparent transition-all"
                  />
                </div>
                <div>
                  <label className="block text-gray-700 dark:text-gray-300 mb-2 text-sm">Department</label>
                  <input
                    type="text"
                    defaultValue="Radiology"
                    className="w-full px-4 py-2 border border-gray-200 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 dark:text-white focus:outline-none focus:ring-2 focus:ring-[#007AFF] focus:border-transparent transition-all"
                  />
                </div>
              </div>
            </div>

            {/* Notification Settings */}
            <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm border border-gray-100 dark:border-gray-700 p-6">
              <div className="flex items-center gap-3 mb-6">
                <div className="w-10 h-10 bg-purple-100 dark:bg-purple-900/40 rounded-lg flex items-center justify-center">
                  <Bell className="w-5 h-5 text-purple-600 dark:text-purple-300" />
                </div>
                <div>
                  <h3 className="text-gray-900 dark:text-white">Notifications</h3>
                  <p className="text-sm text-gray-500 dark:text-gray-400">Manage notification preferences</p>
                </div>
              </div>

              <div className="space-y-4">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-gray-900 dark:text-white text-sm">Enable Notifications</p>
                    <p className="text-xs text-gray-500 dark:text-gray-400">Receive alerts for new cases</p>
                  </div>
                  <button
                    type="button"
                    onClick={() => setNotificationsEnabled(!notificationsEnabled)}
                    className={`relative w-12 h-6 rounded-full transition-colors ${
                      notificationsEnabled ? 'bg-green-500' : 'bg-gray-300 dark:bg-gray-600'
                    }`}
                  >
                    <div
                      className={`absolute top-1 left-1 w-4 h-4 bg-white rounded-full transition-transform ${
                        notificationsEnabled ? 'translate-x-6' : ''
                      }`}
                    />
                  </button>
                </div>
              </div>
            </div>

            {/* AI Model Settings */}
            <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm border border-gray-100 dark:border-gray-700 p-6">
              <div className="flex items-center gap-3 mb-6">
                <div className="w-10 h-10 bg-green-100 dark:bg-green-900/40 rounded-lg flex items-center justify-center">
                  <BarChart3 className="w-5 h-5 text-green-600 dark:text-green-400" />
                </div>
                <div>
                  <h3 className="text-gray-900 dark:text-white">AI Model Settings</h3>
                  <p className="text-sm text-gray-500 dark:text-gray-400">Configure AI analysis preferences</p>
                </div>
              </div>

              <div className="space-y-6">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-gray-900 dark:text-white text-sm">Auto-Analysis</p>
                    <p className="text-xs text-gray-500 dark:text-gray-400">Automatically analyze uploaded scans</p>
                  </div>
                  <button
                    type="button"
                    onClick={() => setAutoAnalysis(!autoAnalysis)}
                    className={`relative w-12 h-6 rounded-full transition-colors ${
                      autoAnalysis ? 'bg-green-500' : 'bg-gray-300 dark:bg-gray-600'
                    }`}
                  >
                    <div
                      className={`absolute top-1 left-1 w-4 h-4 bg-white rounded-full transition-transform ${
                        autoAnalysis ? 'translate-x-6' : ''
                      }`}
                    />
                  </button>
                </div>

                <div>
                  <label className="block text-gray-700 dark:text-gray-300 mb-3 text-sm">
                    Confidence Threshold: {confidenceThreshold}%
                  </label>
                  <input
                    type="range"
                    min="50"
                    max="99"
                    value={confidenceThreshold}
                    onChange={(e) => setConfidenceThreshold(Number(e.target.value))}
                    className="w-full accent-[#007AFF]"
                  />
                  <div className="flex justify-between text-xs text-gray-500 dark:text-gray-400 mt-1">
                    <span>50%</span>
                    <span>99%</span>
                  </div>
                </div>
              </div>
            </div>

            {/* Security Settings */}
            <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm border border-gray-100 dark:border-gray-700 p-6">
              <div className="flex items-center gap-3 mb-6">
                <div className="w-10 h-10 bg-red-100 dark:bg-red-900/40 rounded-lg flex items-center justify-center">
                  <Lock className="w-5 h-5 text-red-600 dark:text-red-400" />
                </div>
                <div>
                  <h3 className="text-gray-900 dark:text-white">Security</h3>
                  <p className="text-sm text-gray-500 dark:text-gray-400">Manage security settings</p>
                </div>
              </div>

              <div className="space-y-3">
                <button
                  type="button"
                  className="w-full text-left px-4 py-3 border border-gray-200 dark:border-gray-600 rounded-lg hover:bg-gray-50 dark:hover:bg-gray-700/50 transition-colors"
                >
                  <p className="text-gray-900 dark:text-white text-sm">Change Password</p>
                  <p className="text-xs text-gray-500 dark:text-gray-400">Update your login password</p>
                </button>
                <button
                  type="button"
                  className="w-full text-left px-4 py-3 border border-gray-200 dark:border-gray-600 rounded-lg hover:bg-gray-50 dark:hover:bg-gray-700/50 transition-colors"
                >
                  <p className="text-gray-900 dark:text-white text-sm">Two-Factor Authentication</p>
                  <p className="text-xs text-gray-500 dark:text-gray-400">Add an extra layer of security</p>
                </button>
              </div>
            </div>

            {/* Language Settings */}
            <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm border border-gray-100 dark:border-gray-700 p-6">
              <div className="flex items-center gap-3 mb-6">
                <div className="w-10 h-10 bg-yellow-100 dark:bg-yellow-900/40 rounded-lg flex items-center justify-center">
                  <Globe className="w-5 h-5 text-yellow-600 dark:text-yellow-400" />
                </div>
                <div>
                  <h3 className="text-gray-900 dark:text-white">Language & Region</h3>
                  <p className="text-sm text-gray-500 dark:text-gray-400">Set your preferred language</p>
                </div>
              </div>

              <div>
                <label className="block text-gray-700 dark:text-gray-300 mb-2 text-sm">Language</label>
                <select className="w-full px-4 py-2 border border-gray-200 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 dark:text-white focus:outline-none focus:ring-2 focus:ring-[#007AFF] focus:border-transparent transition-all">
                  <option>English (US)</option>
                  <option>English (UK)</option>
                  <option>Spanish</option>
                  <option>French</option>
                  <option>German</option>
                </select>
              </div>
            </div>

            {/* Save Button */}
            <div className="flex justify-end gap-3">
              <button
                type="button"
                className="px-6 py-3 border border-gray-200 dark:border-gray-600 rounded-lg hover:bg-gray-50 dark:hover:bg-gray-700 transition-colors text-gray-700 dark:text-gray-200"
                onClick={handleCancel}
              >
                Cancel
              </button>
              <button
                type="button"
                className="px-6 py-3 bg-[#007AFF] text-white rounded-lg hover:bg-[#0062CC] transition-colors shadow-md"
                onClick={handleSave}
              >
                Save Changes
              </button>
            </div>
          </motion.div>
    </div>
  );
}