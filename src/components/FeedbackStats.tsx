import { useState, useEffect } from 'react';
import { MessageSquare, TrendingUp, CheckCircle, AlertCircle } from 'lucide-react';
import { motion } from 'motion/react';

interface FeedbackStats {
  total_feedback: number;
  accuracy_percentage: number;
  total_predictions_reviewed: number;
  correct_predictions: number;
  breakdown: Array<{
    model_said: string;
    actually: string;
    count: number;
    avg_model_confidence: number;
    avg_actual_confidence: number;
  }>;
  recent_errors: Array<{
    case_id: string;
    model_prediction: string;
    actual_prediction: string;
    reason: string;
    date: string;
  }>;
}

export function FeedbackStats() {
  const [stats, setStats] = useState<FeedbackStats | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchStats = async () => {
      try {
        const response = await fetch('http://localhost:8000/feedback/stats');
        if (!response.ok) throw new Error('Failed to fetch feedback stats');
        const data = await response.json();
        setStats(data);
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Failed to load stats');
      } finally {
        setLoading(false);
      }
    };

    const interval = setInterval(fetchStats, 10000);
    fetchStats();
    return () => clearInterval(interval);
  }, []);

  if (loading) {
    return (
      <div className="flex items-center justify-center p-8">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600"></div>
      </div>
    );
  }

  if (error || !stats) {
    return (
      <div className="bg-red-50 dark:bg-red-900/30 border border-red-200 dark:border-red-800 rounded-lg p-4 text-red-700 dark:text-red-300">
        {error || 'Failed to load feedback statistics'}
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Header Stats */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="bg-white dark:bg-gray-800 rounded-xl shadow-lg border border-gray-200 dark:border-gray-700 p-6"
        >
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-gray-600 dark:text-gray-400 text-sm font-medium">Total Feedback</h3>
            <MessageSquare className="w-5 h-5 text-blue-600" />
          </div>
          <p className="text-3xl font-bold text-gray-900 dark:text-white">{stats.total_feedback}</p>
          <p className="text-xs text-gray-500 dark:text-gray-400 mt-2">submissions received</p>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.1 }}
          className="bg-white dark:bg-gray-800 rounded-xl shadow-lg border border-gray-200 dark:border-gray-700 p-6"
        >
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-gray-600 dark:text-gray-400 text-sm font-medium">Model Accuracy</h3>
            <TrendingUp className="w-5 h-5 text-emerald-600" />
          </div>
          <p className="text-3xl font-bold text-gray-900 dark:text-white">{stats.accuracy_percentage.toFixed(1)}%</p>
          <p className="text-xs text-gray-500 dark:text-gray-400 mt-2">based on verified cases</p>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.2 }}
          className="bg-white dark:bg-gray-800 rounded-xl shadow-lg border border-gray-200 dark:border-gray-700 p-6"
        >
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-gray-600 dark:text-gray-400 text-sm font-medium">Correct</h3>
            <CheckCircle className="w-5 h-5 text-green-600" />
          </div>
          <p className="text-3xl font-bold text-gray-900 dark:text-white">{stats.correct_predictions}</p>
          <p className="text-xs text-gray-500 dark:text-gray-400 mt-2">of {stats.total_predictions_reviewed}</p>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.3 }}
          className="bg-white dark:bg-gray-800 rounded-xl shadow-lg border border-gray-200 dark:border-gray-700 p-6"
        >
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-gray-600 dark:text-gray-400 text-sm font-medium">Errors</h3>
            <AlertCircle className="w-5 h-5 text-red-600" />
          </div>
          <p className="text-3xl font-bold text-gray-900 dark:text-white">
            {stats.total_predictions_reviewed - stats.correct_predictions}
          </p>
          <p className="text-xs text-gray-500 dark:text-gray-400 mt-2">cases to review</p>
        </motion.div>
      </div>

      {/* Breakdown Table */}
      {stats.breakdown.length > 0 && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.4 }}
          className="bg-white dark:bg-gray-800 rounded-xl shadow-lg border border-gray-200 dark:border-gray-700 p-6"
        >
          <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">Prediction Breakdown</h3>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-gray-200 dark:border-gray-700">
                  <th className="text-left py-3 px-4 text-gray-600 dark:text-gray-400 font-medium">Model Said</th>
                  <th className="text-left py-3 px-4 text-gray-600 dark:text-gray-400 font-medium">Actually Was</th>
                  <th className="text-center py-3 px-4 text-gray-600 dark:text-gray-400 font-medium">Count</th>
                  <th className="text-center py-3 px-4 text-gray-600 dark:text-gray-400 font-medium">Model Conf.</th>
                  <th className="text-center py-3 px-4 text-gray-600 dark:text-gray-400 font-medium">Actual Conf.</th>
                </tr>
              </thead>
              <tbody>
                {stats.breakdown.map((row, idx) => (
                  <tr key={idx} className="border-b border-gray-100 dark:border-gray-700 hover:bg-gray-50 dark:hover:bg-gray-700/50">
                    <td className="py-3 px-4 text-gray-900 dark:text-white font-medium">{row.model_said}</td>
                    <td className="py-3 px-4 text-gray-900 dark:text-white font-medium">{row.actually}</td>
                    <td className="py-3 px-4 text-center text-gray-600 dark:text-gray-400">{row.count}</td>
                    <td className="py-3 px-4 text-center text-gray-600 dark:text-gray-400">
                      {(row.avg_model_confidence * 100).toFixed(0)}%
                    </td>
                    <td className="py-3 px-4 text-center text-gray-600 dark:text-gray-400">
                      {(row.avg_actual_confidence * 100).toFixed(0)}%
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </motion.div>
      )}

      {/* Recent Errors */}
      {stats.recent_errors.length > 0 && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.5 }}
          className="bg-white dark:bg-gray-800 rounded-xl shadow-lg border border-gray-200 dark:border-gray-700 p-6"
        >
          <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">Recent Model Disagreements</h3>
          <div className="space-y-3">
            {stats.recent_errors.map((error, idx) => (
              <motion.div
                key={idx}
                initial={{ opacity: 0, x: -10 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: 0.5 + idx * 0.05 }}
                className="bg-red-50 dark:bg-red-900/30 border border-red-200 dark:border-red-800 rounded-lg p-4"
              >
                <div className="flex items-start justify-between mb-2">
                  <div className="flex items-center gap-3">
                    <span className="inline-block px-3 py-1 bg-red-100 dark:bg-red-900/50 text-red-700 dark:text-red-300 text-xs font-medium rounded">
                      Model: {error.model_prediction}
                    </span>
                    <span className="text-gray-500 dark:text-gray-400">→</span>
                    <span className="inline-block px-3 py-1 bg-emerald-100 dark:bg-emerald-900/50 text-emerald-700 dark:text-emerald-300 text-xs font-medium rounded">
                      Actual: {error.actual_prediction}
                    </span>
                  </div>
                  <span className="text-xs text-gray-500 dark:text-gray-400">{error.case_id}</span>
                </div>
                <p className="text-sm text-gray-700 dark:text-gray-300">{error.reason}</p>
              </motion.div>
            ))}
          </div>
        </motion.div>
      )}

      {stats.total_feedback === 0 && (
        <div className="bg-blue-50 dark:bg-blue-900/30 border border-blue-200 dark:border-blue-800 rounded-lg p-8 text-center">
          <MessageSquare className="w-12 h-12 text-blue-600 dark:text-blue-400 mx-auto mb-3 opacity-50" />
          <p className="text-gray-700 dark:text-gray-300">No feedback submitted yet.</p>
          <p className="text-sm text-gray-600 dark:text-gray-400">Feedback from verified cases will appear here to track model performance.</p>
        </div>
      )}
    </div>
  );
}
