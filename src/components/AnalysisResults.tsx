import { useState } from 'react';
import { CheckCircle, AlertTriangle } from 'lucide-react';
import { motion } from 'motion/react';
import type { AnalysisData, ModelPrediction } from '../types/patient';

interface AnalysisResultsProps {
  analysisData: AnalysisData;
  onVerifyAndAccept: () => void;
  onCorrection: () => void;
}

export function AnalysisResults({ analysisData, onVerifyAndAccept, onCorrection }: AnalysisResultsProps) {
  const [showHeatmap, setShowHeatmap] = useState(false);
  const hasHeatmap = analysisData.prediction !== 'Normal' && !!analysisData.heatmapImage;

  const modelEntries = analysisData.multiModel
    ? Object.entries(analysisData.multiModel.models) as [string, ModelPrediction][]
    : [];

  const getResultColor = () => {
    switch (analysisData.predictionColor) {
      case 'red':
        return {
          bg: 'linear-gradient(135deg, rgba(255,59,48,0.1) 0%, rgba(255,59,48,0.05) 100%)',
          border: '2px solid rgba(255,59,48,0.2)',
          shadow: '0 8px 32px rgba(255,59,48,0.15)',
          iconBg: 'bg-red-100',
          iconColor: 'text-[#FF3B30]',
          textColor: 'text-[#FF3B30]',
        };
      case 'green':
        return {
          bg: 'linear-gradient(135deg, rgba(50,215,75,0.1) 0%, rgba(50,215,75,0.05) 100%)',
          border: '2px solid rgba(50,215,75,0.2)',
          shadow: '0 8px 32px rgba(50,215,75,0.15)',
          iconBg: 'bg-green-100',
          iconColor: 'text-[#32D74B]',
          textColor: 'text-[#32D74B]',
        };
      case 'blue':
        return {
          bg: 'linear-gradient(135deg, rgba(0,122,255,0.1) 0%, rgba(0,122,255,0.05) 100%)',
          border: '2px solid rgba(0,122,255,0.2)',
          shadow: '0 8px 32px rgba(0,122,255,0.15)',
          iconBg: 'bg-blue-100',
          iconColor: 'text-[#007AFF]',
          textColor: 'text-[#007AFF]',
        };
      default:
        return {
          bg: 'linear-gradient(135deg, rgba(255,59,48,0.1) 0%, rgba(255,59,48,0.05) 100%)',
          border: '2px solid rgba(255,59,48,0.2)',
          shadow: '0 8px 32px rgba(255,59,48,0.15)',
          iconBg: 'bg-red-100',
          iconColor: 'text-[#FF3B30]',
          textColor: 'text-[#FF3B30]',
        };
    }
  };

  const resultColors = getResultColor();

  return (
    <div className="flex flex-1 min-h-0 w-full flex-col lg:flex-row">
      <div className="flex-1 p-6 min-h-0 flex flex-col">
        <motion.div
          initial={{ x: -20, opacity: 0 }}
          animate={{ x: 0, opacity: 1 }}
          transition={{ duration: 0.5 }}
          className="flex flex-col flex-1 min-h-0"
        >
          <div className="bg-white dark:bg-gray-800/90 rounded-xl shadow-lg border border-gray-200 dark:border-gray-700 p-6 flex flex-col flex-1 min-h-0">
            <h2 className="text-gray-900 dark:text-white mb-4 shrink-0">Ultrasound Viewer</h2>

            <div className="flex flex-col flex-1 min-h-0 items-center justify-center">
              <div className="relative w-full max-w-md mx-auto aspect-square bg-black rounded-xl overflow-hidden border border-white/10 shadow-inner">
                {analysisData.originalImage ? (
                  <img
                    src={analysisData.originalImage}
                    alt="Original scan"
                    className="absolute inset-0 w-full h-full object-contain"
                  />
                ) : (
                  <div className="absolute inset-0 flex items-center justify-center text-gray-400 dark:text-gray-500 text-sm">No image</div>
                )}
                {hasHeatmap ? (
                  <img
                    src={analysisData.heatmapImage}
                    alt="Model attention heatmap"
                    className={`absolute inset-0 w-full h-full object-contain transition-opacity duration-500 ease-in-out pointer-events-none ${
                      showHeatmap ? 'opacity-100' : 'opacity-0'
                    }`}
                  />
                ) : null}
              </div>

              {hasHeatmap ? (
                <div className="flex items-center justify-center gap-3 sm:gap-4 mt-4 w-full max-w-md mx-auto min-h-10">
                  <span
                    className={`text-sm font-medium transition-colors duration-300 shrink-0 ${
                      showHeatmap ? 'text-gray-400 dark:text-gray-300 font-normal' : 'text-[#007AFF] font-bold'
                    }`}
                  >
                    Original
                  </span>
                  <button
                    type="button"
                    role="switch"
                    aria-checked={showHeatmap}
                    aria-label={showHeatmap ? 'Switch to original scan' : 'Switch to model heatmap'}
                    onClick={() => setShowHeatmap((v) => !v)}
                    className={`relative h-8 w-[52px] shrink-0 rounded-full transition-colors duration-300 focus:outline-none focus-visible:ring-2 focus-visible:ring-[#007AFF] focus-visible:ring-offset-2 ${
                      showHeatmap ? 'bg-[#007AFF]' : 'bg-gray-300 dark:bg-gray-600'
                    }`}
                  >
                    <span
                      className={`absolute top-1 left-1 h-6 w-6 rounded-full bg-white shadow-md transition-transform duration-300 ease-in-out ${
                        showHeatmap ? 'translate-x-[20px]' : 'translate-x-0'
                      }`}
                    />
                  </button>
                  <span
                    className={`text-sm font-medium transition-colors duration-300 shrink-0 ${
                      showHeatmap ? 'text-[#007AFF] font-bold' : 'text-gray-400 dark:text-gray-300 font-normal'
                    }`}
                  >
                    Model Heatmap
                  </span>
                </div>
              ) : null}
            </div>
          </div>
        </motion.div>
      </div>

      <div className="flex-1 p-6 overflow-auto">
        <motion.div
          initial={{ x: 20, opacity: 0 }}
          animate={{ x: 0, opacity: 1 }}
          transition={{ duration: 0.5 }}
          className="space-y-6"
        >
          <div>
            <h2 className="text-gray-900 dark:text-white mb-1">Clinical Intelligence Report</h2>
            <p className="text-gray-500 dark:text-gray-400 text-sm">
              Generated: {new Date().toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' })} | Three independent classifiers
            </p>
          </div>

          <div
            className="rounded-xl p-8 text-center"
            style={{
              background: resultColors.bg,
              border: resultColors.border,
              boxShadow: resultColors.shadow,
            }}
          >
            <div className={`inline-flex items-center justify-center w-16 h-16 ${resultColors.iconBg} rounded-full mb-4`}>
              {analysisData.predictionColor === 'green' ? (
                <CheckCircle className={`w-8 h-8 ${resultColors.iconColor}`} />
              ) : (
                <AlertTriangle className={`w-8 h-8 ${resultColors.iconColor}`} />
              )}
            </div>
            <h1 className={`${resultColors.textColor} mb-3`} style={{ fontSize: '2.5rem' }}>
              {analysisData.prediction.toUpperCase()}
            </h1>
            <div className="inline-block px-6 py-2 bg-white dark:bg-gray-800 rounded-full">
              <p className="text-gray-700 dark:text-gray-200">
                Confidence Score: <span className={resultColors.textColor}>{analysisData.confidence}</span>
              </p>
            </div>
          </div>

          {modelEntries.length > 0 && (
            <div>
              <h2 className="text-gray-900 dark:text-white mb-3">Model Results</h2>
              <div className="grid grid-cols-1 xl:grid-cols-3 gap-4">
                {modelEntries.map(([name, result]) => (
                  <div key={name} className="bg-white dark:bg-gray-800/90 rounded-xl p-5 border border-gray-200 dark:border-gray-700">
                    <div className="flex items-start justify-between gap-3 mb-4">
                      <div>
                        <h3 className="text-gray-900 dark:text-white font-semibold">{name}</h3>
                        <p className="text-xs text-gray-500 dark:text-gray-400">EfficientNet-B0 • {result.task.replace(' classification', '')}</p>
                      </div>
                      <span className="text-sm font-semibold text-[#007AFF]">{Math.round(result.confidence * 100)}%</span>
                    </div>
                    <p className="text-xs uppercase tracking-wide text-gray-500 dark:text-gray-400 mb-1">Prediction</p>
                    <p className="text-lg font-semibold text-gray-900 dark:text-white mb-4">{result.prediction}</p>
                    <div className="space-y-2">
                      {Object.entries(result.probabilities).map(([label, probability]) => (
                        probability == null ? null : (
                          <div key={label}>
                            <div className="flex justify-between text-xs text-gray-600 dark:text-gray-300 mb-1">
                              <span>{label}</span><span>{Math.round(probability * 100)}%</span>
                            </div>
                            <div className="h-2 rounded-full bg-gray-100 dark:bg-gray-700 overflow-hidden">
                              <div className="h-full bg-[#007AFF] rounded-full" style={{ width: `${probability * 100}%` }} />
                            </div>
                          </div>
                        )
                      ))}
                    </div>
                    {result.note && <p className="text-xs text-gray-500 dark:text-gray-400 mt-4">{result.note}</p>}
                  </div>
                ))}
              </div>
            </div>
          )}

          {analysisData.multiModel && (
            <div className="rounded-xl p-6 border-2 border-[#007AFF]/30 bg-blue-50/70 dark:bg-blue-950/30">
              <p className="text-xs uppercase tracking-wide text-[#007AFF] font-semibold mb-2">Model Consensus</p>
              <p className="text-2xl font-semibold text-gray-900 dark:text-white">{analysisData.multiModel.consensus.prediction}</p>
              <p className="text-gray-700 dark:text-gray-200 mt-1">
                {analysisData.multiModel.consensus.agreement_count} / {analysisData.multiModel.consensus.participating_models ?? analysisData.multiModel.consensus.model_count} models agree
              </p>
              <p className="text-sm text-gray-600 dark:text-gray-300 mt-2">{analysisData.multiModel.consensus.status}</p>
              {analysisData.multiModel.consensus.note && <p className="text-xs text-gray-500 dark:text-gray-400 mt-2">{analysisData.multiModel.consensus.note}</p>}
            </div>
          )}

          <div className="space-y-3 pt-4">
            <button
              type="button"
              onClick={onVerifyAndAccept}
              className="w-full bg-green-500 hover:bg-green-600 text-white py-4 rounded-xl transition-colors flex items-center justify-center gap-2 shadow-lg"
            >
              <CheckCircle className="w-5 h-5" />
              <span>Verify & Accept Diagnosis</span>
            </button>

            <button
              type="button"
              onClick={onCorrection}
              className="w-full bg-white dark:bg-gray-800 border-2 border-[#007AFF] text-[#007AFF] hover:bg-blue-50 dark:hover:bg-gray-700 py-4 rounded-xl transition-colors flex items-center justify-center gap-2"
            >
              <AlertTriangle className="w-5 h-5" />
              <span>Incorrect? Enter Correction Mode</span>
            </button>
          </div>
        </motion.div>
      </div>
    </div>
  );
}
