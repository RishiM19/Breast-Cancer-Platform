import { BarChart3, Upload, TrendingUp, Bell } from 'lucide-react';
import { motion } from 'motion/react';
import { useRef, useState } from 'react';

interface CaseRecord {
  id: string;
  time: string;
  prediction: string;
  predictionColor: 'red' | 'green' | 'blue';
  confidence: string;
  status: string;
  statusColor: 'orange' | 'grey' | 'blue';
  originalImage: string;
}

interface DashboardProps {
  onUpload: (file: File, patientName: string, patientAge: string) => void;
  recentCases: CaseRecord[];
  totalScans: number;
  aiAccuracy: number;
  userName: string;
  isAnalyzing?: boolean;
}

export function Dashboard({
  onUpload,
  recentCases,
  totalScans,
  aiAccuracy,
  userName,
  isAnalyzing = false,
}: DashboardProps) {
  const [showNotifications, setShowNotifications] = useState(false);
  const [patientName, setPatientName] = useState('');
  const [patientAge, setPatientAge] = useState('');
  const fileInputRef = useRef<HTMLInputElement>(null);

  const notifications = [
    { id: 1, text: 'New scan uploaded for PT-4921', time: '2 min ago', unread: true },
    { id: 2, text: 'Analysis completed for PT-4920', time: '15 min ago', unread: true },
    { id: 3, text: 'Model accuracy updated to 96.5%', time: '1 hour ago', unread: false },
  ];

  const unreadCount = notifications.filter((n) => n.unread).length;

  const getFirstName = (name: string) => {
    const parts = name.split(' ');
    return parts[0];
  };
  const stats = [
    {
      label: 'Total Scans Today',
      value: totalScans.toString(),
      icon: BarChart3,
      color: 'text-blue-600 dark:text-blue-400',
    },
    {
      label: 'AI Accuracy',
      value: `${aiAccuracy.toFixed(1)}%`,
      icon: TrendingUp,
      color: 'text-green-600 dark:text-green-400',
      trend: 'up',
    },
  ];

  const getPredictionColor = (color: string) => {
    switch (color) {
      case 'red':
        return 'bg-red-100 text-red-800 dark:bg-red-950/60 dark:text-red-300';
      case 'green':
        return 'bg-green-100 text-green-800 dark:bg-green-950/60 dark:text-green-300';
      case 'blue':
        return 'bg-blue-100 text-blue-800 dark:bg-blue-950/60 dark:text-blue-300';
      default:
        return 'bg-gray-100 text-gray-800 dark:bg-gray-700 dark:text-gray-200';
    }
  };

  const getStatusColor = (color: string) => {
    switch (color) {
      case 'orange':
        return 'bg-orange-100 text-orange-800 dark:bg-orange-950/50 dark:text-orange-300';
      case 'grey':
        return 'bg-gray-100 text-gray-800 dark:bg-gray-700 dark:text-gray-200';
      case 'blue':
        return 'bg-blue-100 text-blue-800 dark:bg-blue-950/50 dark:text-blue-300';
      default:
        return 'bg-gray-100 text-gray-800 dark:bg-gray-700 dark:text-gray-200';
    }
  };

  const triggerFileSelect = () => {
    if (isAnalyzing) return;
    const name = patientName.trim();
    const age = patientAge.trim();
    if (!name || !age) {
      alert('Please enter the patient name and age before uploading a scan.');
      return;
    }
    fileInputRef.current?.click();
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      onUpload(file, patientName.trim(), patientAge.trim());
      e.target.value = '';
    }
  };

  return (
    <div className="p-8">
      <input
        ref={fileInputRef}
        type="file"
        accept="image/*"
        className="hidden"
        onChange={handleFileChange}
      />

      {/* Header */}
      <motion.div
        className="mb-8"
        initial={{ y: -20, opacity: 0 }}
        animate={{ y: 0, opacity: 1 }}
        transition={{ duration: 0.5, delay: 0.1 }}
      >
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-gray-900 dark:text-white mb-1">Good Morning, {getFirstName(userName)}</h1>
            <p className="text-gray-500 dark:text-gray-400">
              {new Date().toLocaleDateString('en-US', {
                weekday: 'long',
                year: 'numeric',
                month: 'long',
                day: 'numeric',
              })}
            </p>
          </div>

          <div className="relative">
            <button
              onClick={() => setShowNotifications(!showNotifications)}
              className="relative p-3 rounded-lg hover:bg-white dark:hover:bg-gray-800 transition-all border border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-800 shadow-sm"
            >
              <Bell className="w-5 h-5 text-gray-600 dark:text-gray-300" />
              {unreadCount > 0 && (
                <span className="absolute top-1 right-1 w-5 h-5 bg-red-500 text-white text-xs rounded-full flex items-center justify-center">
                  {unreadCount}
                </span>
              )}
            </button>

            {showNotifications && (
              <motion.div
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                className="absolute right-0 mt-2 w-80 bg-white dark:bg-gray-800 rounded-xl shadow-lg border border-gray-200 dark:border-gray-700 z-50"
              >
                <div className="p-4 border-b border-gray-100 dark:border-gray-700">
                  <h3 className="text-gray-900 dark:text-white">Notifications</h3>
                  <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">{unreadCount} unread messages</p>
                </div>
                <div className="max-h-96 overflow-y-auto">
                  {notifications.map((notification) => (
                    <div
                      key={notification.id}
                      className={`p-4 border-b border-gray-100 dark:border-gray-700 hover:bg-gray-50 dark:hover:bg-gray-700/50 transition-colors ${
                        notification.unread ? 'bg-blue-50 dark:bg-blue-950/40' : ''
                      }`}
                    >
                      <p className="text-gray-900 dark:text-white text-sm">{notification.text}</p>
                      <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">{notification.time}</p>
                    </div>
                  ))}
                </div>
                <div className="p-3 border-t border-gray-100 dark:border-gray-700 text-center">
                  <button className="text-[#007AFF] text-sm hover:underline">View all notifications</button>
                </div>
              </motion.div>
            )}
          </div>
        </div>
      </motion.div>

      {/* Stats Row */}
      <motion.div
        className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-8"
        initial={{ y: 20, opacity: 0 }}
        animate={{ y: 0, opacity: 1 }}
        transition={{ duration: 0.5, delay: 0.2 }}
      >
        {stats.map((stat, index) => (
          <div
            key={index}
            className="rounded-xl p-6 shadow-sm border border-gray-200/80 dark:border-gray-600 bg-white/95 dark:bg-gray-800/95 backdrop-blur-sm dark:shadow-gray-950/40"
          >
            <div className="flex items-start justify-between mb-3">
              <div className={`p-2 rounded-lg bg-gray-100/90 dark:bg-gray-700/80 ${stat.color}`}>
                <stat.icon className="w-5 h-5" />
              </div>
              {stat.trend && (
                <span className="text-green-600 dark:text-green-400 text-xs flex items-center gap-1">
                  <TrendingUp className="w-3 h-3" />
                  +2.4%
                </span>
              )}
            </div>
            <h3 className="text-gray-900 dark:text-white mb-1">{stat.value}</h3>
            <p className="text-gray-500 dark:text-gray-400 text-sm">{stat.label}</p>
          </div>
        ))}
      </motion.div>

      {/* Patient details (required before upload) */}
      <motion.div
        className="mb-6 grid grid-cols-1 md:grid-cols-2 gap-4"
        initial={{ y: 16, opacity: 0 }}
        animate={{ y: 0, opacity: 1 }}
        transition={{ duration: 0.45, delay: 0.25 }}
      >
        <div>
          <label htmlFor="patient-name" className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
            Patient Name
          </label>
          <input
            id="patient-name"
            type="text"
            autoComplete="name"
            value={patientName}
            onChange={(e) => setPatientName(e.target.value)}
            placeholder="e.g. Jane Doe"
            className="w-full px-4 py-3 rounded-xl border border-gray-200 dark:border-gray-600 bg-white/90 dark:bg-gray-700 backdrop-blur-sm shadow-sm focus:outline-none focus:ring-2 focus:ring-[#007AFF] focus:border-transparent transition-all text-gray-900 dark:text-white placeholder:text-gray-400 dark:placeholder:text-gray-500"
          />
        </div>
        <div>
          <label htmlFor="patient-age" className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
            Patient Age
          </label>
          <input
            id="patient-age"
            type="text"
            inputMode="numeric"
            value={patientAge}
            onChange={(e) => setPatientAge(e.target.value)}
            placeholder="e.g. 52"
            className="w-full px-4 py-3 rounded-xl border border-gray-200 dark:border-gray-600 bg-white/90 dark:bg-gray-700 backdrop-blur-sm shadow-sm focus:outline-none focus:ring-2 focus:ring-[#007AFF] focus:border-transparent transition-all text-gray-900 dark:text-white placeholder:text-gray-400 dark:placeholder:text-gray-500"
          />
        </div>
      </motion.div>

      {/* Upload Section */}
      <motion.div
        className="mb-8"
        initial={{ y: 20, opacity: 0 }}
        animate={{ y: 0, opacity: 1 }}
        transition={{ duration: 0.5, delay: 0.3 }}
      >
        <div
          role="button"
          tabIndex={0}
          onClick={triggerFileSelect}
          onKeyDown={(e) => {
            if (e.key === 'Enter' || e.key === ' ') {
              e.preventDefault();
              triggerFileSelect();
            }
          }}
          className={`rounded-xl p-12 border-2 border-dashed border-gray-300 dark:border-gray-500 bg-white/85 dark:bg-gray-800/90 dark:ring-1 dark:ring-gray-600/80 backdrop-blur-md hover:border-[#007AFF] dark:hover:border-[#007AFF] transition-all ${
            isAnalyzing ? 'opacity-70 pointer-events-none cursor-wait' : 'cursor-pointer'
          }`}
        >
          <div className="flex flex-col items-center justify-center text-center">
            <div className="w-16 h-16 bg-[#007AFF]/15 dark:bg-[#007AFF]/25 rounded-full flex items-center justify-center mb-4">
              <Upload className="w-8 h-8 text-[#007AFF]" />
            </div>
            <h3 className="text-gray-900 dark:text-white mb-2">
              {isAnalyzing ? 'Analyzing scan…' : 'Upload New Scan'}
            </h3>
            <p className="text-gray-500 dark:text-gray-400 text-sm">
              {isAnalyzing ? 'Please wait while the model processes your image.' : 'Drag and drop files here, or click to browse'}
            </p>
            <p className="text-gray-400 dark:text-gray-500 text-xs mt-2">Supported formats: DICOM, PNG, JPEG</p>
          </div>
        </div>
      </motion.div>

      {/* Recent Cases Table */}
      <motion.div
        initial={{ y: 20, opacity: 0 }}
        animate={{ y: 0, opacity: 1 }}
        transition={{ duration: 0.5, delay: 0.4 }}
      >
        <div className="rounded-xl shadow-sm border border-gray-200/90 dark:border-gray-600 overflow-hidden bg-white/95 dark:bg-gray-800/95 backdrop-blur-sm dark:shadow-gray-950/30">
          <div className="p-6 border-b border-gray-200 dark:border-gray-600 bg-gray-50/80 dark:bg-gray-900/50">
            <h2 className="text-gray-900 dark:text-white">Recent Patient Cases</h2>
          </div>

          <div className="overflow-x-auto">
            <table className="w-full">
              <thead className="bg-gray-100/90 dark:bg-gray-900/70 border-b border-gray-200 dark:border-gray-600">
                <tr>
                  <th className="px-6 py-3 text-left text-xs text-gray-600 dark:text-gray-300 w-20">Thumb</th>
                  <th className="px-6 py-3 text-left text-xs text-gray-600 dark:text-gray-300">Patient ID</th>
                  <th className="px-6 py-3 text-left text-xs text-gray-600 dark:text-gray-300">Scan Date</th>
                  <th className="px-6 py-3 text-left text-xs text-gray-600 dark:text-gray-300">AI Prediction</th>
                  <th className="px-6 py-3 text-left text-xs text-gray-600 dark:text-gray-300">Confidence</th>
                  <th className="px-6 py-3 text-left text-xs text-gray-600 dark:text-gray-300">Status</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-gray-200 dark:divide-gray-600 bg-white dark:bg-gray-800/80">
                {recentCases.map((case_item, index) => (
                  <tr key={index} className="hover:bg-gray-50 dark:hover:bg-gray-700/50 transition-colors">
                    <td className="px-6 py-3">
                      <div className="w-12 h-12 rounded-lg overflow-hidden border border-gray-200 dark:border-gray-500 bg-gray-100 dark:bg-gray-700 shrink-0">
                        {case_item.originalImage ? (
                          <img
                            src={case_item.originalImage}
                            alt=""
                            className="w-full h-full object-cover"
                          />
                        ) : (
                          <div className="w-full h-full bg-gray-200 dark:bg-gray-600" />
                        )}
                      </div>
                    </td>
                    <td className="px-6 py-4 text-gray-900 dark:text-white">{case_item.id}</td>
                    <td className="px-6 py-4 text-gray-600 dark:text-gray-300">{case_item.time}</td>
                    <td className="px-6 py-4">
                      <span
                        className={`inline-flex px-3 py-1 rounded-full text-xs ${getPredictionColor(case_item.predictionColor)}`}
                      >
                        {case_item.prediction}
                      </span>
                    </td>
                    <td className="px-6 py-4 text-gray-900 dark:text-white">{case_item.confidence}</td>
                    <td className="px-6 py-4">
                      <span
                        className={`inline-flex px-3 py-1 rounded-full text-xs ${getStatusColor(case_item.statusColor)}`}
                      >
                        {case_item.status}
                      </span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </motion.div>
    </div>
  );
}
