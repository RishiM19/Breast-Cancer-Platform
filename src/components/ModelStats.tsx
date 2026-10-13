import { Activity, CheckCircle, Database } from 'lucide-react';
import { motion } from 'motion/react';
import { FeedbackStats } from './FeedbackStats';
import type { PatientRecord } from '../types/patient';

interface ModelStatsProps {
  patientDatabase: PatientRecord[];
}

export function ModelStats({ patientDatabase }: ModelStatsProps) {
  const modelCards = [
    {
      label: 'BUSI', value: 'EfficientNet-B0', detail: '3-class • BUSI dataset', icon: Activity,
    },
    {
      label: 'UCLM', value: 'EfficientNet-B0', detail: '3-class • BUS-UCLM dataset', icon: Database,
    },
    {
      label: 'BUS-BRA', value: 'EfficientNet-B0', detail: 'Binary • BUS-BRA dataset', icon: CheckCircle,
    },
  ];
  const classificationSummary = ['Malignant', 'Benign', 'Normal'].map((label) => {
    const records = patientDatabase.filter((patient) => patient.analysis.prediction === label);
    const confidenceTotal = records.reduce((total, patient) => {
      const match = patient.analysis.confidence.match(/(\d+(?:\.\d+)?)/);
      return total + (match ? Number(match[1]) / (Number(match[1]) > 1 ? 100 : 1) : 0);
    }, 0);
    return {
      label,
      count: records.length,
      averageConfidence: records.length ? (confidenceTotal / records.length) * 100 : null,
    };
  });

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
            <p className="text-gray-500 dark:text-gray-400">Real-time clinical intelligence metrics and insights</p>
          </motion.div>

          {/* Active model checkpoints */}
          <motion.div 
            className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8"
            initial={{ y: 20, opacity: 0 }}
            animate={{ y: 0, opacity: 1 }}
            transition={{ duration: 0.5, delay: 0.2 }}
          >
            {modelCards.map((metric, index) => (
              <div
                key={index}
                className="rounded-xl p-6 shadow-sm border border-gray-200/80 dark:border-gray-600 bg-white/95 dark:bg-gray-800/95 backdrop-blur-sm dark:shadow-gray-950/40"
              >
                <div className="flex items-start justify-between mb-3">
                  <div className="p-2.5 rounded-lg ring-1 ring-black/5 dark:ring-white/10 bg-blue-100 dark:bg-blue-950/55">
                    <metric.icon className="w-5 h-5 text-blue-600 dark:text-blue-400" />
                  </div>
                </div>
                <h3 className="text-gray-900 dark:text-white mb-1">{metric.label}</h3>
                <p className="text-gray-700 dark:text-gray-200 text-sm">{metric.value}</p>
                <p className="text-gray-500 dark:text-gray-400 text-xs mt-1">{metric.detail}</p>
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
            {/* Known evaluation metrics */}
            <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm border border-gray-100 dark:border-gray-700 p-6">
              <h2 className="text-gray-900 dark:text-white mb-6">Known Evaluation Metrics</h2>
              <div className="space-y-4">
                {[
                  ['BUSI', 'Validation accuracy approximately 91%'],
                  ['UCLM V2', 'Validation balanced accuracy 77.10%; test accuracy 83.81%; test balanced accuracy 76.29%'],
                  ['BUS-BRA', 'Test accuracy 82.98%; test balanced accuracy 82.35%'],
                ].map(([label, value]) => (
                  <div
                    key={label}
                    className="flex justify-between items-center pb-3 border-b border-gray-100 dark:border-gray-700 last:border-0"
                  >
                    <p className="text-gray-600 dark:text-gray-400 text-sm">{label}</p>
                    <p className="text-gray-900 dark:text-white text-sm text-right max-w-[65%]">{value}</p>
                  </div>
                ))}
              </div>
            </div>

            {/* Scope note */}
            <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm border border-gray-100 dark:border-gray-700 p-6">
              <h2 className="text-gray-900 dark:text-white mb-6">Evaluation Scope</h2>
              <p className="text-gray-600 dark:text-gray-300 text-sm leading-relaxed">Metrics are reported separately for each dataset and task. They are not combined into an overall Lumina accuracy.</p>
              <p className="text-gray-600 dark:text-gray-300 text-sm leading-relaxed mt-4">Predictions shown in analysis are checkpoint inference results, not validation metrics.</p>
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
                {classificationSummary.map((summary) => (
                  <div key={summary.label} className="text-center p-6 bg-gray-50 dark:bg-gray-900/40 rounded-lg border border-gray-200 dark:border-gray-700">
                    <div className="text-4xl text-gray-900 dark:text-white mb-2">{summary.count}</div>
                    <p className="text-gray-700 dark:text-gray-200">{summary.label} Cases</p>
                    <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">
                      {summary.averageConfidence == null ? 'No cases recorded' : `${summary.averageConfidence.toFixed(1)}% Avg. Confidence`}
                    </p>
                  </div>
                ))}
              </div>
            </div>
          </motion.div>

          {/* Feedback Statistics Section */}
          <motion.div
            className="mt-8"
            initial={{ y: 20, opacity: 0 }}
            animate={{ y: 0, opacity: 1 }}
            transition={{ duration: 0.5, delay: 0.5 }}
          >
            <h2 className="text-gray-900 dark:text-white mb-6 text-lg font-semibold">Doctor Feedback & Model Learning</h2>
            <FeedbackStats />
          </motion.div>
    </div>
  );
}
