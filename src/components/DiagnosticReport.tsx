import { ArrowLeft, FileText, Calendar, User, Activity, Printer } from 'lucide-react';
import { motion } from 'motion/react';
import type { PatientRecord } from '../types/patient';

interface DiagnosticReportProps {
  patient: PatientRecord;
  onBack: () => void;
}

function getColorClasses(color: string) {
  switch (color) {
    case 'red':
      return 'bg-red-100 text-red-700 border-red-300';
    case 'green':
      return 'bg-green-100 text-green-700 border-green-300';
    case 'blue':
      return 'bg-blue-100 text-blue-700 border-blue-300';
    default:
      return 'bg-gray-100 text-gray-700 border-gray-300';
  }
}

function confidenceToBarWidth(confidence: string): string {
  const m = confidence.match(/(\d+(?:\.\d+)?)/);
  if (!m) return '0%';
  const n = parseFloat(m[1]);
  const pct = n > 1 ? Math.min(100, n) : Math.min(100, n * 100);
  return `${pct}%`;
}

/** Map BI-RADS narrative to accent for badge (Category 5 / 3 / 1). */
function biRadsAccent(biRads: string): {
  bar: string;
  text: string;
  ring: string;
  label: string;
} {
  const s = biRads.toLowerCase();
  if (s.includes('category 5')) {
    return {
      bar: 'bg-red-600',
      text: 'text-red-800',
      ring: 'ring-red-200',
      label: 'High suspicion',
    };
  }
  if (s.includes('category 3')) {
    return {
      bar: 'bg-amber-400',
      text: 'text-amber-950',
      ring: 'ring-amber-200',
      label: 'Probably benign — short-interval follow-up',
    };
  }
  if (s.includes('category 1')) {
    return {
      bar: 'bg-emerald-600',
      text: 'text-emerald-900',
      ring: 'ring-emerald-200',
      label: 'Negative study',
    };
  }
  return {
    bar: 'bg-slate-500',
    text: 'text-slate-800',
    ring: 'ring-slate-200',
    label: 'Assessment',
  };
}

export function DiagnosticReport({ patient, onBack }: DiagnosticReportProps) {
  const { analysis: a } = patient;
  const rd = a.reportDetails;
  const bi = biRadsAccent(rd.BI_RADS);

  const scanDate = (() => {
    try {
      return new Date(patient.date).toLocaleDateString('en-US', {
        month: 'short',
        day: 'numeric',
        year: 'numeric',
      });
    } catch {
      return patient.date;
    }
  })();

  const handlePrint = () => {
    window.print();
  };

  const showHeatmap = Boolean(patient.heatmapImage?.trim()) && a.prediction !== 'Normal';

  return (
    <div className="diagnostic-report-root min-h-screen bg-[#F8F9FB] dark:bg-gray-900 print:bg-white print:min-h-0 print:text-gray-900">
      <style>{`
        @media print {
          @page {
            margin: 12mm;
          }
          .clinical-print-root {
            -webkit-print-color-adjust: exact;
            print-color-adjust: exact;
          }
          .clinical-section {
            break-inside: avoid;
            page-break-inside: avoid;
          }
          /* Keep hardcopy legible when UI is in dark mode (html.dark) */
          html.dark .diagnostic-report-root {
            background: #ffffff !important;
            color: #111827 !important;
          }
          html.dark .diagnostic-report-root .clinical-print-root {
            background-color: #ffffff !important;
            color: #111827 !important;
          }
          html.dark .diagnostic-report-root .bg-white,
          html.dark .diagnostic-report-root [class*="dark:bg-gray-800"] {
            background-color: #ffffff !important;
          }
          html.dark .diagnostic-report-root .bg-gray-50,
          html.dark .diagnostic-report-root .bg-slate-50\\/90,
          html.dark .diagnostic-report-root [class*="bg-slate-50"] {
            background-color: #f9fafb !important;
          }
          html.dark .diagnostic-report-root .text-gray-900,
          html.dark .diagnostic-report-root .dark\\:text-white {
            color: #111827 !important;
          }
          html.dark .diagnostic-report-root .text-gray-500,
          html.dark .diagnostic-report-root .text-gray-600,
          html.dark .diagnostic-report-root .text-gray-700,
          html.dark .diagnostic-report-root .text-gray-800,
          html.dark .diagnostic-report-root .dark\\:text-gray-300,
          html.dark .diagnostic-report-root .dark\\:text-gray-400 {
            color: #4b5563 !important;
          }
        }
      `}</style>

      <div className="clinical-print-root bg-[#F8F9FB] dark:bg-gray-900 print:bg-white print:text-gray-900">
        <div className="bg-white dark:bg-gray-800 border-b border-gray-200 dark:border-gray-700 print:border-0 print:bg-white clinical-section">
          <div className="max-w-6xl mx-auto px-8 py-6">
            <button
              type="button"
              onClick={onBack}
              className="flex items-center gap-2 text-[#007AFF] hover:text-[#0062CC] transition-colors mb-4 print:hidden"
            >
              <ArrowLeft className="w-5 h-5" />
              <span>Back to Patient List</span>
            </button>

            <div className="flex items-center justify-between">
              <div>
                <h1 className="text-gray-900 dark:text-white print:text-gray-900 mb-1 text-2xl font-semibold tracking-tight">AI Diagnostic Report</h1>
                <p className="text-gray-500 dark:text-gray-400 print:text-gray-600 text-sm">Structured ultrasound assessment • Patient ID: {patient.id}</p>
              </div>
              <button
                type="button"
                onClick={handlePrint}
                className="flex items-center gap-2 px-5 py-3 bg-[#007AFF] text-white rounded-lg hover:bg-[#0062CC] transition-colors shadow-md print:hidden"
              >
                <Printer className="w-5 h-5" />
                <span>Export / Print</span>
              </button>
            </div>
          </div>
        </div>

        <div className="max-w-6xl mx-auto px-8 py-8 print:px-4 print:py-4">
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            <motion.div
              className="lg:col-span-1 space-y-6"
              initial={{ x: -20, opacity: 0 }}
              animate={{ x: 0, opacity: 1 }}
              transition={{ duration: 0.5 }}
            >
              <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm border border-gray-100 dark:border-gray-700 p-6 print:shadow-none print:border-gray-200 print:bg-white clinical-section">
                <div className="flex items-center gap-3 mb-6">
                  <div className="w-10 h-10 bg-blue-100 dark:bg-blue-900/40 rounded-lg flex items-center justify-center">
                    <User className="w-5 h-5 text-[#007AFF]" />
                  </div>
                  <h2 className="text-gray-900 dark:text-white print:text-gray-900 font-semibold">Patient Information</h2>
                </div>

                <div className="space-y-4 text-sm">
                  <div>
                    <p className="text-gray-500 dark:text-gray-400 print:text-gray-600 mb-1">Patient Name</p>
                    <p className="text-gray-900 dark:text-white print:text-gray-900">{patient.name}</p>
                  </div>
                  <div>
                    <p className="text-gray-500 dark:text-gray-400 print:text-gray-600 mb-1">Patient ID</p>
                    <p className="text-gray-900 dark:text-white print:text-gray-900">{patient.id}</p>
                  </div>
                  <div>
                    <p className="text-gray-500 dark:text-gray-400 print:text-gray-600 mb-1">Age</p>
                    <p className="text-gray-900 dark:text-white print:text-gray-900">{patient.age}</p>
                  </div>
                  <div>
                    <p className="text-gray-500 dark:text-gray-400 print:text-gray-600 mb-1">Scan Date</p>
                    <div className="flex items-center gap-2 text-gray-900 dark:text-white print:text-gray-900">
                      <Calendar className="w-4 h-4 print:hidden" />
                      <span>{scanDate}</span>
                    </div>
                  </div>
                </div>
              </div>

              <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm border border-gray-100 dark:border-gray-700 p-6 print:shadow-none print:border-gray-200 print:bg-white clinical-section">
                <h3 className="text-gray-900 dark:text-white print:text-gray-900 font-semibold mb-4">Medical imaging</h3>
                <div
                  className={`grid gap-4 ${showHeatmap ? 'grid-cols-1 sm:grid-cols-2' : 'grid-cols-1'}`}
                >
                  <div className="aspect-square bg-gray-100 dark:bg-gray-700/50 rounded-lg overflow-hidden border border-gray-200/80 dark:border-gray-600 print:border-gray-300 print:bg-gray-100">
                    {patient.originalImage ? (
                      <img
                        src={patient.originalImage}
                        alt="Original scan"
                        className="w-full h-full object-contain print:object-contain"
                      />
                    ) : (
                      <div className="w-full h-full flex items-center justify-center text-gray-400 dark:text-gray-500 print:text-gray-500 text-sm">No image</div>
                    )}
                  </div>
                  {showHeatmap && (
                    <div className="aspect-square bg-gray-100 dark:bg-gray-700/50 rounded-lg overflow-hidden border border-gray-200/80 dark:border-gray-600 print:border-gray-300 print:bg-gray-100">
                      <img
                        src={patient.heatmapImage}
                        alt="AI heatmap"
                        className="w-full h-full object-contain print:object-contain"
                      />
                    </div>
                  )}
                </div>
              </div>
            </motion.div>

            <motion.div
              className="lg:col-span-2 space-y-6"
              initial={{ x: 20, opacity: 0 }}
              animate={{ x: 0, opacity: 1 }}
              transition={{ duration: 0.5 }}
            >
              <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm border border-gray-100 dark:border-gray-700 p-6 print:shadow-none print:border-gray-200 print:bg-white clinical-section">
                <div className="flex items-center gap-3 mb-6">
                  <div className="w-10 h-10 bg-purple-100 dark:bg-purple-900/40 rounded-lg flex items-center justify-center">
                    <Activity className="w-5 h-5 text-purple-600 dark:text-purple-300" />
                  </div>
                  <h2 className="text-gray-900 dark:text-white print:text-gray-900 font-semibold">AI Analysis Results</h2>
                </div>

                <div className={`rounded-xl border-2 p-6 mb-6 ${getColorClasses(a.predictionColor)}`}>
                  <p className="text-sm mb-2 opacity-75">Primary classification</p>
                  <h3 className="mb-3 text-xl font-semibold">{a.prediction}</h3>
                  <div className="flex items-center gap-2">
                    <div className="flex-1 bg-white dark:bg-gray-600 bg-opacity-50 rounded-full h-3 overflow-hidden print:bg-white">
                      <div
                        className="h-full bg-current opacity-75 transition-all duration-1000"
                        style={{ width: confidenceToBarWidth(a.confidence) }}
                      />
                    </div>
                    <span className="text-sm whitespace-nowrap">{a.confidence}</span>
                  </div>
                </div>

                <div className="grid grid-cols-2 gap-4">
                  <div className="bg-gray-50 dark:bg-gray-700/40 rounded-lg p-4 print:bg-gray-50">
                    <p className="text-gray-500 dark:text-gray-400 print:text-gray-600 text-sm mb-2">Tumor Grade / Severity</p>
                    <p className="text-gray-900 dark:text-white print:text-gray-900">{a.tumorGrade}</p>
                  </div>
                  <div className="bg-gray-50 dark:bg-gray-700/40 rounded-lg p-4 print:bg-gray-50">
                    <p className="text-gray-500 dark:text-gray-400 print:text-gray-600 text-sm mb-2">Node Status</p>
                    <p className="text-gray-900 dark:text-white print:text-gray-900">{a.nodeStatus}</p>
                  </div>
                  <div className="bg-gray-50 dark:bg-gray-700/40 rounded-lg p-4 print:bg-gray-50">
                    <p className="text-gray-500 dark:text-gray-400 print:text-gray-600 text-sm mb-2">Lesion Size</p>
                    <p className="text-gray-900 dark:text-white print:text-gray-900">{a.lesionSize}</p>
                  </div>
                  <div className="bg-gray-50 dark:bg-gray-700/40 rounded-lg p-4 print:bg-gray-50">
                    <p className="text-gray-500 dark:text-gray-400 print:text-gray-600 text-sm mb-2">T-Stage</p>
                    <p className="text-gray-900 dark:text-white print:text-gray-900">{a.subtype}</p>
                  </div>
                </div>
              </div>

              {/* Structured narrative — primary printable clinical content */}
              <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm border border-gray-100 dark:border-gray-700 overflow-hidden print:shadow-none print:border-gray-200 print:bg-white clinical-section">
                <div className="px-6 pt-6 pb-2 border-b border-gray-100 dark:border-gray-700 print:border-gray-200">
                  <h3 className="text-gray-900 dark:text-white print:text-gray-900 text-lg font-semibold tracking-tight">Clinical ultrasound report</h3>
                  <p className="text-gray-500 dark:text-gray-400 print:text-gray-600 text-sm mt-1 leading-relaxed">
                    Standardized sections derived from the structured AI report. Lesion metrics above supplement the narrative
                    below.
                  </p>
                </div>

                <div className="p-6 space-y-6">
                  <section className="space-y-2">
                    <h4 className="text-xs font-semibold uppercase tracking-wider text-gray-500 dark:text-gray-400 print:text-gray-600">
                      Clinical indication & breast composition
                    </h4>
                    <div className="text-sm text-gray-700 dark:text-gray-300 print:text-gray-700 leading-relaxed space-y-3">
                      {rd.Indication ? <p>{rd.Indication}</p> : null}
                      {rd.Composition ? <p>{rd.Composition}</p> : null}
                      {!rd.Indication && !rd.Composition ? (
                        <p className="text-gray-400 dark:text-gray-500 print:text-gray-500 italic">No indication or composition text available.</p>
                      ) : null}
                    </div>
                  </section>

                  <section>
                    <h4 className="text-xs font-semibold uppercase tracking-wider text-gray-500 dark:text-gray-400 print:text-gray-600 mb-3">Ultrasound findings</h4>
                    <div
                      className="rounded-xl border-l-4 border-[#007AFF] bg-slate-50/90 dark:bg-slate-800/60 print:bg-slate-50 px-5 py-4 shadow-inner border border-slate-200/80 dark:border-slate-600"
                      style={{ boxShadow: 'inset 0 1px 0 0 rgba(255,255,255,0.6)' }}
                    >
                      <p className="text-sm text-gray-800 dark:text-gray-200 print:text-gray-800 leading-relaxed whitespace-pre-wrap">
                        {rd.Findings || 'No findings narrative recorded.'}
                      </p>
                      <p className="text-xs text-gray-500 dark:text-gray-400 print:text-gray-600 mt-4 pt-3 border-t border-slate-200/80 dark:border-slate-600">
                        Documented lesion size (sonographic): <span className="font-medium text-gray-700 dark:text-gray-200 print:text-gray-800">{a.lesionSize}</span>
                      </p>
                    </div>
                  </section>

                  <section className="space-y-2">
                    <h4 className="text-xs font-semibold uppercase tracking-wider text-gray-500 dark:text-gray-400 print:text-gray-600">Impression</h4>
                    <p className="text-base font-bold text-gray-900 dark:text-white print:text-gray-900 leading-snug">{rd.Impression || '—'}</p>
                  </section>

                  <section
                    className={`rounded-2xl border-2 overflow-hidden ring-2 ${bi.ring} print:ring-2 clinical-section`}
                  >
                    <div className={`h-2 ${bi.bar} print:h-2`} aria-hidden />
                    <div className="px-6 py-5 bg-white dark:bg-gray-800 print:bg-white">
                      <p className="text-xs font-semibold uppercase tracking-wider text-gray-500 dark:text-gray-400 print:text-gray-600 mb-2">BI-RADS assessment</p>
                      <p className={`text-xl md:text-2xl font-bold tracking-tight ${bi.text} print:text-gray-900`}>{rd.BI_RADS || '—'}</p>
                      <p className="text-sm text-gray-600 dark:text-gray-300 print:text-gray-600 mt-2">{bi.label}</p>
                    </div>
                  </section>
                </div>
              </div>

              <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm border border-gray-100 dark:border-gray-700 p-6 print:shadow-none print:border-gray-200 print:bg-white clinical-section">
                <div className="flex items-center gap-2 mb-4">
                  <FileText className="w-5 h-5 text-gray-600 dark:text-gray-300 print:text-gray-600" />
                  <h3 className="text-gray-900 dark:text-white print:text-gray-900 font-semibold">Clinical Notes</h3>
                </div>
                <div className="space-y-3 text-gray-600 dark:text-gray-300 print:text-gray-600 text-sm">
                  {a.notes ? (
                    <p className="text-gray-800 dark:text-gray-200 print:text-gray-800 whitespace-pre-wrap">{a.notes}</p>
                  ) : (
                    <p className="text-gray-500 dark:text-gray-400 print:text-gray-500 italic">No additional notes recorded for this case.</p>
                  )}
                  <div className="bg-blue-50 dark:bg-blue-950/40 border border-blue-200 dark:border-blue-800 rounded-lg p-4 mt-4 print:bg-blue-50">
                    <p className="text-blue-900 dark:text-blue-200 print:text-blue-900 text-sm">
                      <span className="font-semibold">Note:</span> This is an AI-assisted diagnosis and should not replace
                      professional medical judgment. Consult the care team for final diagnosis and treatment decisions.
                    </p>
                  </div>
                </div>
              </div>

              <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm border border-gray-100 dark:border-gray-700 p-6 print:shadow-none print:border-gray-200 print:bg-white clinical-section">
                <h3 className="text-gray-900 dark:text-white print:text-gray-900 font-semibold mb-4">Analysis Metadata</h3>
                <div className="grid grid-cols-2 gap-4 text-sm">
                  <div>
                    <p className="text-gray-500 dark:text-gray-400 print:text-gray-600 mb-1">Model Version</p>
                    <p className="text-gray-900 dark:text-white print:text-gray-900">v2.4.1</p>
                  </div>
                  <div>
                    <p className="text-gray-500 dark:text-gray-400 print:text-gray-600 mb-1">Processing Time</p>
                    <p className="text-gray-900 dark:text-white print:text-gray-900">1.2 seconds</p>
                  </div>
                  <div>
                    <p className="text-gray-500 dark:text-gray-400 print:text-gray-600 mb-1">Image Quality</p>
                    <p className="text-gray-900 dark:text-white print:text-gray-900">Excellent</p>
                  </div>
                  <div>
                    <p className="text-gray-500 dark:text-gray-400 print:text-gray-600 mb-1">Analysis Status</p>
                    <p className="text-green-600 dark:text-green-400 print:text-green-700">Completed</p>
                  </div>
                </div>
              </div>
            </motion.div>
          </div>
        </div>
      </div>
    </div>
  );
}
