import { TrendingUp, Activity, CheckCircle, AlertCircle } from 'lucide-react';
import { motion } from 'motion/react';

interface ModelStatsProps {
  aiAccuracy: number;
}

export function ModelStats({ aiAccuracy }: ModelStatsProps) {
  const performanceMetrics = [
    {
      label: 'Overall Accuracy',
      value: `${aiAccuracy.toFixed(1)}%`,
      icon: TrendingUp,
      color: 'text-green-600 dark:text-green-400',
      bgColor: 'bg-green-100 dark:bg-green-950/55',
    },
    {
      label: 'Sensitivity',
      value: '96.8%',
      icon: Activity,
      color: 'text-blue-600 dark:text-blue-400',
      bgColor: 'bg-blue-100 dark:bg-blue-950/55',
    },
    {
      label: 'Specificity',
      value: '97.2%',
      icon: CheckCircle,
      color: 'text-purple-600 dark:text-purple-400',
      bgColor: 'bg-purple-100 dark:bg-purple-950/55',
    },
    {
      label: 'False Positive Rate',
      value: '2.1%',
      icon: AlertCircle,
      color: 'text-orange-600 dark:text-orange-400',
      bgColor: 'bg-orange-100 dark:bg-orange-950/55',
    },
  ];

  const modelDetails = [
    { label: 'Model Version', value: 'v2.4.1' },
    { label: 'Architecture', value: 'ResNet-101 + Attention' },
    { label: 'Training Dataset', value: '124,000 scans' },
    { label: 'Last Updated', value: 'Nov 15, 2025' },
    { label: 'Total Predictions', value: '48,392' },
    { label: 'Avg. Inference Time', value: '1.2s' },
  ];

  const recentPerformance = [
    { date: 'Nov 28', accuracy: 98.1 },
    { date: 'Nov 29', accuracy: 98.3 },
    { date: 'Nov 30', accuracy: 98.2 },
    { date: 'Dec 01', accuracy: aiAccuracy },
  ];

  return (
    <div className="p-8">
          {/* Header */}
          <motion.div 
            className="mb-8"
            initial={{ y: -20, opacity: 0 }}
            animate={{ y: 0, opacity: 1 }}
            transition={{ duration: 0.5, delay: 0.1 }}
          >
            <h1 className="text-gray-900 dark:text-white mb-1">Model Performance Statistics</h1>
            <p className="text-gray-500 dark:text-gray-400">Real-time AI model metrics and insights</p>
          </motion.div>

          {/* Performance Metrics */}
          <motion.div 
            className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8"
            initial={{ y: 20, opacity: 0 }}
            animate={{ y: 0, opacity: 1 }}
            transition={{ duration: 0.5, delay: 0.2 }}
          >
            {performanceMetrics.map((metric, index) => (
              <div
                key={index}
                className="rounded-xl p-6 shadow-sm border border-gray-200/80 dark:border-gray-600 bg-white/95 dark:bg-gray-800/95 backdrop-blur-sm dark:shadow-gray-950/40"
              >
                <div className="flex items-start justify-between mb-3">
                  <div className={`p-2.5 rounded-lg ring-1 ring-black/5 dark:ring-white/10 ${metric.bgColor}`}>
                    <metric.icon className={`w-5 h-5 ${metric.color}`} />
                  </div>
                </div>
                <h3 className="text-gray-900 dark:text-white mb-1">{metric.value}</h3>
                <p className="text-gray-500 dark:text-gray-400 text-sm">{metric.label}</p>
              </div>
            ))}
          </motion.div>

          {/* Model Details */}
          <motion.div 
            className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-8"
            initial={{ y: 20, opacity: 0 }}
            animate={{ y: 0, opacity: 1 }}
            transition={{ duration: 0.5, delay: 0.3 }}
          >
            {/* Model Information */}
            <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm border border-gray-100 dark:border-gray-700 p-6">
              <h2 className="text-gray-900 dark:text-white mb-6">Model Information</h2>
              <div className="space-y-4">
                {modelDetails.map((detail, index) => (
                  <div
                    key={index}
                    className="flex justify-between items-center pb-3 border-b border-gray-100 dark:border-gray-700 last:border-0"
                  >
                    <p className="text-gray-600 dark:text-gray-400 text-sm">{detail.label}</p>
                    <p className="text-gray-900 dark:text-white">{detail.value}</p>
                  </div>
                ))}
              </div>
            </div>

            {/* Recent Performance Chart */}
            <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm border border-gray-100 dark:border-gray-700 p-6">
              <h2 className="text-gray-900 dark:text-white mb-6">Recent Performance Trend</h2>
              <div className="space-y-4">
                {recentPerformance.map((day, index) => (
                  <div key={index} className="space-y-2">
                    <div className="flex justify-between items-center text-sm">
                      <span className="text-gray-600 dark:text-gray-400">{day.date}</span>
                      <span className="text-gray-900 dark:text-white">{day.accuracy.toFixed(1)}%</span>
                    </div>
                    <div className="w-full bg-gray-200 dark:bg-gray-600 rounded-full h-2">
                      <div
                        className="bg-gradient-to-r from-[#007AFF] to-[#32D74B] h-2 rounded-full transition-all duration-500"
                        style={{ width: `${day.accuracy}%` }}
                      />
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </motion.div>

          {/* Classification Breakdown */}
          <motion.div
            initial={{ y: 20, opacity: 0 }}
            animate={{ y: 0, opacity: 1 }}
            transition={{ duration: 0.5, delay: 0.4 }}
          >
            <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm border border-gray-100 dark:border-gray-700 p-6">
              <h2 className="text-gray-900 dark:text-white mb-6">Classification Breakdown (Last 30 Days)</h2>
              
              <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                <div className="text-center p-6 bg-red-50 dark:bg-red-950/40 rounded-lg border border-transparent dark:border-red-900/50">
                  <div className="text-4xl text-red-600 dark:text-red-400 mb-2">342</div>
                  <p className="text-gray-700 dark:text-gray-200">Malignant Cases</p>
                  <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">94.2% Avg. Confidence</p>
                </div>
                
                <div className="text-center p-6 bg-green-50 dark:bg-green-950/40 rounded-lg border border-transparent dark:border-green-900/50">
                  <div className="text-4xl text-green-600 dark:text-green-400 mb-2">1,248</div>
                  <p className="text-gray-700 dark:text-gray-200">Benign Cases</p>
                  <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">96.8% Avg. Confidence</p>
                </div>
                
                <div className="text-center p-6 bg-blue-50 dark:bg-blue-950/40 rounded-lg border border-transparent dark:border-blue-900/50">
                  <div className="text-4xl text-blue-600 dark:text-blue-400 mb-2">892</div>
                  <p className="text-gray-700 dark:text-gray-200">Normal Cases</p>
                  <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">98.1% Avg. Confidence</p>
                </div>
              </div>
            </div>
          </motion.div>
    </div>
  );
}
