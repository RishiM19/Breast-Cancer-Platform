import { useState } from 'react';
import { X, CheckCircle, AlertCircle } from 'lucide-react';
import { motion } from 'motion/react';

interface FeedbackModalProps {
  caseId: string;
  originalPrediction: string;
  originalConfidence: string;
  onClose: () => void;
  onSuccess?: () => void;
}

export function FeedbackModal({
  caseId,
  originalPrediction,
  originalConfidence,
  onClose,
  onSuccess,
}: FeedbackModalProps) {
  const [correctedPrediction, setCorrectedPrediction] = useState<string>(originalPrediction);
  const [correctedConfidence, setCorrectedConfidence] = useState<number>(0.9);
  const [reasoning, setReasoning] = useState<string>('');
  const [feedbackType, setFeedbackType] = useState<string>('pathology_confirmed');
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [submissionStatus, setSubmissionStatus] = useState<'idle' | 'success' | 'error'>('idle');
  const [errorMessage, setErrorMessage] = useState<string>('');

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();

    if (!reasoning.trim()) {
      setErrorMessage('Please provide a reason for your feedback.');
      return;
    }

    setIsSubmitting(true);
    setErrorMessage('');

    try {
      const response = await fetch('http://localhost:8000/feedback/submit', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          case_id: caseId,
          corrected_prediction: correctedPrediction,
          corrected_confidence: correctedConfidence,
          reasoning: reasoning.trim(),
          feedback_type: feedbackType,
          doctor_id: 'anonymous',
        }),
      });

      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || 'Failed to submit feedback');
      }

      setSubmissionStatus('success');
      setTimeout(() => {
        onSuccess?.();
        onClose();
      }, 2000);
    } catch (err) {
      setErrorMessage(err instanceof Error ? err.message : 'Failed to submit feedback');
      setSubmissionStatus('error');
    } finally {
      setIsSubmitting(false);
    }
  };

  if (submissionStatus === 'success') {
    return (
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        exit={{ opacity: 0 }}
        className="fixed inset-0 bg-black/50 flex items-center justify-center z-50"
      >
        <motion.div
          initial={{ scale: 0.95, opacity: 0 }}
          animate={{ scale: 1, opacity: 1 }}
          className="bg-white dark:bg-gray-800 rounded-xl shadow-2xl p-8 max-w-md w-full mx-4"
        >
          <div className="flex flex-col items-center text-center gap-4">
            <motion.div
              initial={{ scale: 0 }}
              animate={{ scale: 1 }}
              transition={{ delay: 0.1 }}
              className="bg-green-100 dark:bg-green-900/30 rounded-full p-3"
            >
              <CheckCircle className="w-8 h-8 text-green-600 dark:text-green-400" />
            </motion.div>
            <h3 className="text-lg font-semibold text-gray-900 dark:text-white">
              Feedback Submitted Successfully!
            </h3>
            <p className="text-sm text-gray-600 dark:text-gray-400">
              Your feedback has been recorded and will help improve the model's accuracy.
            </p>
          </div>
        </motion.div>
      </motion.div>
    );
  }

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      className="fixed inset-0 bg-black/50 flex items-center justify-center z-50 p-4"
    >
      <motion.div
        initial={{ scale: 0.95, opacity: 0, y: 20 }}
        animate={{ scale: 1, opacity: 1, y: 0 }}
        exit={{ scale: 0.95, opacity: 0, y: 20 }}
        className="bg-white dark:bg-gray-800 rounded-xl shadow-2xl max-w-2xl w-full max-h-[90vh] overflow-y-auto"
      >
        {/* Header */}
        <div className="sticky top-0 bg-gradient-to-r from-blue-50 to-blue-100 dark:from-gray-700 dark:to-gray-600 border-b border-blue-200 dark:border-gray-600 p-6 flex items-center justify-between">
          <h2 className="text-xl font-bold text-gray-900 dark:text-white">Case Feedback</h2>
          <button
            onClick={onClose}
            disabled={isSubmitting}
            className="text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-300 disabled:opacity-50"
          >
            <X className="w-6 h-6" />
          </button>
        </div>

        {/* Content */}
        <form onSubmit={handleSubmit} className="p-6 space-y-6">
          {/* Original Analysis */}
          <div className="bg-gray-50 dark:bg-gray-900/50 rounded-lg p-4 border border-gray-200 dark:border-gray-700">
            <h3 className="text-sm font-semibold text-gray-700 dark:text-gray-300 mb-3">
              Original AI Analysis
            </h3>
            <div className="grid grid-cols-2 gap-4">
              <div>
                <p className="text-xs text-gray-600 dark:text-gray-400">Prediction</p>
                <p className="font-semibold text-gray-900 dark:text-white">
                  {originalPrediction}
                </p>
              </div>
              <div>
                <p className="text-xs text-gray-600 dark:text-gray-400">Confidence</p>
                <p className="font-semibold text-gray-900 dark:text-white">
                  {originalConfidence}
                </p>
              </div>
            </div>
          </div>

          {/* Corrected Prediction */}
          <div className="space-y-2">
            <label className="block text-sm font-semibold text-gray-900 dark:text-white">
              Actual Diagnosis
            </label>
            <select
              value={correctedPrediction}
              onChange={(e) => setCorrectedPrediction(e.target.value)}
              className="w-full px-4 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-gray-900 dark:text-white focus:ring-2 focus:ring-blue-500 outline-none"
            >
              <option value="Benign">Benign</option>
              <option value="Malignant">Malignant</option>
              <option value="Normal">Normal</option>
            </select>
          </div>

          {/* Confidence Level */}
          <div className="space-y-3">
            <div className="flex items-center justify-between">
              <label className="text-sm font-semibold text-gray-900 dark:text-white">
                Your Confidence Level
              </label>
              <span className="text-lg font-bold text-blue-600 dark:text-blue-400">
                {Math.round(correctedConfidence * 100)}%
              </span>
            </div>
            <input
              type="range"
              min="0"
              max="100"
              step="5"
              value={Math.round(correctedConfidence * 100)}
              onChange={(e) => setCorrectedConfidence(parseInt(e.target.value) / 100)}
              className="w-full h-2 bg-gray-200 dark:bg-gray-700 rounded-lg appearance-none cursor-pointer"
            />
            <div className="flex justify-between text-xs text-gray-600 dark:text-gray-400">
              <span>Not Sure</span>
              <span>Certain</span>
            </div>
          </div>

          {/* Feedback Type */}
          <div className="space-y-2">
            <label className="block text-sm font-semibold text-gray-900 dark:text-white">
              Feedback Type
            </label>
            <div className="space-y-2">
              <label className="flex items-center gap-3 cursor-pointer">
                <input
                  type="radio"
                  name="feedbackType"
                  value="pathology_confirmed"
                  checked={feedbackType === 'pathology_confirmed'}
                  onChange={(e) => setFeedbackType(e.target.value)}
                  className="w-4 h-4 accent-blue-600"
                />
                <span className="text-sm text-gray-700 dark:text-gray-300">
                  Pathology Confirmed
                  <span className="text-xs text-gray-500 dark:text-gray-400"> - Official biopsy/lab result</span>
                </span>
              </label>
              <label className="flex items-center gap-3 cursor-pointer">
                <input
                  type="radio"
                  name="feedbackType"
                  value="clinically_overridden"
                  checked={feedbackType === 'clinically_overridden'}
                  onChange={(e) => setFeedbackType(e.target.value)}
                  className="w-4 h-4 accent-blue-600"
                />
                <span className="text-sm text-gray-700 dark:text-gray-300">
                  Clinical Override
                  <span className="text-xs text-gray-500 dark:text-gray-400"> - Based on clinical judgment</span>
                </span>
              </label>
              <label className="flex items-center gap-3 cursor-pointer">
                <input
                  type="radio"
                  name="feedbackType"
                  value="uncertain"
                  checked={feedbackType === 'uncertain'}
                  onChange={(e) => setFeedbackType(e.target.value)}
                  className="w-4 h-4 accent-blue-600"
                />
                <span className="text-sm text-gray-700 dark:text-gray-300">
                  Uncertain
                  <span className="text-xs text-gray-500 dark:text-gray-400"> - More information needed</span>
                </span>
              </label>
            </div>
          </div>

          {/* Reasoning */}
          <div className="space-y-2">
            <label className="block text-sm font-semibold text-gray-900 dark:text-white">
              Reason for Correction / Confirmation *
            </label>
            <textarea
              value={reasoning}
              onChange={(e) => setReasoning(e.target.value)}
              placeholder="Explain why you agree or disagree with the AI prediction. Include relevant clinical findings, pathology results, or patient history..."
              rows={4}
              className="w-full px-4 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-gray-900 dark:text-white placeholder-gray-500 dark:placeholder-gray-400 focus:ring-2 focus:ring-blue-500 outline-none resize-none"
            />
            <p className="text-xs text-gray-500 dark:text-gray-400">
              {reasoning.length}/500 characters
            </p>
          </div>

          {/* Error Message */}
          {errorMessage && (
            <motion.div
              initial={{ opacity: 0, y: -10 }}
              animate={{ opacity: 1, y: 0 }}
              className="bg-red-50 dark:bg-red-900/30 border border-red-200 dark:border-red-800 rounded-lg p-4 flex gap-3"
            >
              <AlertCircle className="w-5 h-5 text-red-600 dark:text-red-400 flex-shrink-0 mt-0.5" />
              <p className="text-sm text-red-700 dark:text-red-300">{errorMessage}</p>
            </motion.div>
          )}

          {/* Actions */}
          <div className="flex gap-3 pt-4 border-t border-gray-200 dark:border-gray-700">
            <button
              type="button"
              onClick={onClose}
              disabled={isSubmitting}
              className="flex-1 px-4 py-2 border border-gray-300 dark:border-gray-600 rounded-lg text-gray-700 dark:text-gray-300 font-medium hover:bg-gray-50 dark:hover:bg-gray-700 disabled:opacity-50 transition-colors"
            >
              Cancel
            </button>
            <button
              type="submit"
              disabled={isSubmitting}
              className="flex-1 px-4 py-2 bg-blue-600 text-white rounded-lg font-medium hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
            >
              {isSubmitting ? 'Submitting...' : 'Submit Feedback'}
            </button>
          </div>
        </form>
      </motion.div>
    </motion.div>
  );
}
