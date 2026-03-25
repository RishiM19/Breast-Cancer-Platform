import { useState, useEffect, useCallback, useMemo, useRef } from 'react';
import { Layout } from './components/Layout';
import { Activity } from 'lucide-react';
import { motion } from 'motion/react';
import { Dashboard } from './components/Dashboard';
import { AnalysisResults } from './components/AnalysisResults';
import { CorrectionMode } from './components/CorrectionMode';
import { PatientList } from './components/PatientList';
import { Settings } from './components/Settings';
import { ModelStats } from './components/ModelStats';
import { DiagnosticReport } from './components/DiagnosticReport';
import type { AnalysisData, PatientRecord, PatientAnalysis } from './types/patient';
import { parseReportDetails } from './types/patient';

function readInitialThemeIsDark(): boolean {
  if (typeof window === 'undefined') return false;
  const stored = localStorage.getItem('theme');
  if (stored === 'dark') return true;
  if (stored === 'light') return false;
  return window.matchMedia('(prefers-color-scheme: dark)').matches;
}

const API_BASE = 'http://localhost:8000';

function fileToDataUrl(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const r = new FileReader();
    r.onload = () => resolve(r.result as string);
    r.onerror = () => reject(new Error('Failed to read file'));
    r.readAsDataURL(file);
  });
}

function ensureImageSrc(s: string): string {
  if (!s) return '';
  if (s.startsWith('data:') || s.startsWith('blob:')) return s;
  return `data:image/jpeg;base64,${s}`;
}

function normalizeHeatmapDataUrl(raw: unknown): string {
  if (raw == null) return '';
  const s = String(raw).trim();
  if (!s) return '';
  if (s.startsWith('data:')) return s;
  return `data:image/jpeg;base64,${s}`;
}

function parseLesionSizeCm(lesion: string): number {
  const m = lesion.match(/(\d+(?:\.\d+)?)/);
  if (!m) return 0;
  return parseFloat(m[1]);
}

function defaultTumorGrade(prediction: string): string {
  if (prediction === 'Malignant') return 'Grade 2';
  if (prediction === 'Benign') return 'Grade 1';
  return 'N/A';
}

interface ApiPatientRow {
  id: string;
  name: string;
  age: string;
  date: string;
  prediction: string;
  confidence: number;
  size_cm: number;
  stage: string;
  report_details: unknown;
  original_image: string;
  heatmap_image: string;
}

function mapApiPatientToPatientRecord(row: ApiPatientRow): PatientRecord {
  const confPct = Number.isFinite(row.confidence)
    ? row.confidence <= 1
      ? Math.round(row.confidence * 100)
      : Math.round(row.confidence)
    : 0;
  const prediction = row.prediction;
  const lesionSize = row.size_cm > 0 ? `${row.size_cm} cm` : 'N/A';
  const stageStr = row.stage === 'N/A' || !row.stage ? 'N/A' : row.stage;

  return {
    id: row.id,
    name: row.name,
    age: row.age,
    date: row.date,
    originalImage: ensureImageSrc(row.original_image),
    heatmapImage: normalizeHeatmapDataUrl(row.heatmap_image),
    analysis: {
      prediction,
      predictionColor: predictionToColor(prediction),
      confidence: `${confPct}%`,
      tumorGrade: defaultTumorGrade(prediction),
      nodeStatus: 'Unknown',
      lesionSize,
      subtype: stageStr,
      notes: '',
      reportDetails: parseReportDetails(row.report_details),
    },
  };
}

async function persistPatientToApi(record: PatientRecord, confidenceNum: number, sizeCm: number, stage: string) {
  await fetch(`${API_BASE}/patients`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      id: record.id,
      name: record.name,
      age: record.age,
      date: record.date,
      prediction: record.analysis.prediction,
      confidence: confidenceNum,
      size_cm: sizeCm,
      stage,
      report_details: record.analysis.reportDetails,
      original_image: record.originalImage,
      heatmap_image: record.heatmapImage || '',
    }),
  });
}

type Screen =
  | 'login'
  | 'dashboard'
  | 'analysis'
  | 'correction'
  | 'patient-list'
  | 'settings'
  | 'model-stats'
  | 'diagnostic-report';

const VALID_SCREENS: Screen[] = [
  'login',
  'dashboard',
  'analysis',
  'correction',
  'patient-list',
  'settings',
  'model-stats',
  'diagnostic-report',
];

function isValidScreen(value: string): value is Screen {
  return VALID_SCREENS.includes(value as Screen);
}

export interface CaseRecord {
  id: string;
  time: string;
  prediction: string;
  predictionColor: 'red' | 'green' | 'blue';
  confidence: string;
  status: string;
  statusColor: 'orange' | 'grey' | 'blue';
  originalImage: string;
}

export function predictionToColor(prediction: string): 'red' | 'green' | 'blue' {
  const p = prediction.toLowerCase();
  if (p.includes('malign')) return 'red';
  if (p.includes('benign')) return 'green';
  return 'blue';
}

function mapPredictResponseToAnalysis(data: Record<string, unknown>): AnalysisData {
  const prediction = String(data.prediction ?? 'Normal');
  const confRaw = data.confidence;
  const confNum =
    typeof confRaw === 'number'
      ? confRaw
      : typeof confRaw === 'string'
        ? parseFloat(confRaw)
        : NaN;
  const confidencePct = Number.isFinite(confNum)
    ? confNum <= 1
      ? Math.round(confNum * 100)
      : Math.round(confNum)
    : 0;

  const severity = data.severity_score;
  const tumorGrade = severity != null ? String(severity) : 'N/A';

  const sizeRaw = data.size_cm;
  const lesionSize =
    typeof sizeRaw === 'number'
      ? `${sizeRaw} cm`
      : sizeRaw != null
        ? `${sizeRaw} cm`
        : 'N/A';

  const stage = data.stage != null ? String(data.stage) : 'Unknown';

  return {
    prediction,
    predictionColor: predictionToColor(prediction),
    confidence: `${confidencePct}%`,
    tumorGrade,
    nodeStatus: 'Unknown',
    lesionSize,
    subtype: stage,
    originalImage: '',
    heatmapImage: normalizeHeatmapDataUrl(data.heatmap_base64),
    reportDetails: parseReportDetails(data.report_details),
  };
}

function analysisDataToPatientAnalysis(data: AnalysisData): PatientAnalysis {
  return {
    prediction: data.prediction,
    predictionColor: data.predictionColor,
    confidence: data.confidence,
    tumorGrade: data.tumorGrade,
    nodeStatus: data.nodeStatus,
    lesionSize: data.lesionSize,
    subtype: data.subtype,
    notes: data.notes ?? '',
    reportDetails: data.reportDetails,
  };
}

function patientToCaseRow(p: PatientRecord): CaseRecord {
  const time = (() => {
    try {
      const d = new Date(p.date);
      return d.toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' });
    } catch {
      return p.date;
    }
  })();
  const humanVerified = p.analysis.confidence.includes('Human Verified');
  return {
    id: p.id,
    time,
    prediction: p.analysis.prediction,
    predictionColor: p.analysis.predictionColor,
    confidence: p.analysis.confidence,
    status: humanVerified ? 'Human Verified' : 'AI Complete',
    statusColor: humanVerified ? 'blue' : 'grey',
    originalImage: p.originalImage,
  };
}

export default function App() {
  const [isAuthenticated, setIsAuthenticated] = useState(() => {
    return sessionStorage.getItem('isAuthenticated') === 'true';
  });

  const [currentScreen, setCurrentScreen] = useState<Screen>(() => {
    const auth = sessionStorage.getItem('isAuthenticated') === 'true';
    if (!auth) return 'login';
    const saved = sessionStorage.getItem('currentScreen');
    if (saved && isValidScreen(saved)) {
      const s = saved as Screen;
      if (s === 'analysis' || s === 'correction') return 'dashboard';
      if (s === 'diagnostic-report') return 'patient-list';
      return s;
    }
    return 'dashboard';
  });

  const [patientDatabase, setPatientDatabase] = useState<PatientRecord[]>([]);
  const patientIdSeqRef = useRef(1042);
  const [activePatientId, setActivePatientId] = useState<string | null>(null);
  const [selectedReportPatientId, setSelectedReportPatientId] = useState<string | null>(null);

  const [currentAnalysis, setCurrentAnalysis] = useState<AnalysisData | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);

  const [totalScans, setTotalScans] = useState(() => {
    const saved = sessionStorage.getItem('totalScans');
    return saved ? parseInt(saved, 10) : 48;
  });

  const [aiAccuracy, setAiAccuracy] = useState(() => {
    const saved = sessionStorage.getItem('aiAccuracy');
    return saved ? parseFloat(saved) : 96.5;
  });

  const [userName, setUserName] = useState(() => {
    const saved = sessionStorage.getItem('userName');
    return saved || 'Dr. Elena Rossi';
  });

  const [isDarkMode, setIsDarkMode] = useState(readInitialThemeIsDark);

  const toggleTheme = useCallback(() => {
    setIsDarkMode((d) => !d);
  }, []);

  useEffect(() => {
    if (isDarkMode) {
      document.documentElement.classList.add('dark');
      localStorage.setItem('theme', 'dark');
    } else {
      document.documentElement.classList.remove('dark');
      localStorage.setItem('theme', 'light');
    }
  }, [isDarkMode]);

  useEffect(() => {
    sessionStorage.setItem('currentScreen', currentScreen);
  }, [currentScreen]);

  useEffect(() => {
    sessionStorage.setItem('totalScans', totalScans.toString());
  }, [totalScans]);

  useEffect(() => {
    sessionStorage.setItem('aiAccuracy', aiAccuracy.toString());
  }, [aiAccuracy]);

  useEffect(() => {
    sessionStorage.setItem('userName', userName);
  }, [userName]);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const res = await fetch(`${API_BASE}/patients`);
        if (!res.ok) return;
        const body = (await res.json()) as { patients: ApiPatientRow[] };
        if (cancelled || !body.patients) return;
        const mapped = body.patients.map(mapApiPatientToPatientRecord);
        setPatientDatabase(mapped);
        let maxSeq = 1041;
        for (const p of mapped) {
          const m = /^PAT-(\d+)$/.exec(p.id);
          if (m) maxSeq = Math.max(maxSeq, parseInt(m[1], 10));
        }
        patientIdSeqRef.current = maxSeq + 1;
      } catch {
        /* offline or server down */
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  const recentCases = useMemo((): CaseRecord[] => {
    return patientDatabase.slice(0, 4).map(patientToCaseRow);
  }, [patientDatabase]);

  const selectedReportPatient = useMemo(() => {
    if (!selectedReportPatientId) return null;
    return patientDatabase.find((p) => p.id === selectedReportPatientId) ?? null;
  }, [patientDatabase, selectedReportPatientId]);

  const activePatient = useMemo(() => {
    if (!activePatientId) return null;
    return patientDatabase.find((p) => p.id === activePatientId) ?? null;
  }, [patientDatabase, activePatientId]);

  useEffect(() => {
    if (currentScreen === 'diagnostic-report' && !selectedReportPatient) {
      setCurrentScreen('patient-list');
    }
  }, [currentScreen, selectedReportPatient]);

  const handleLogin = (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    const formData = new FormData(e.currentTarget);
    const hospitalId = formData.get('hospital-id');
    const password = formData.get('password');

    if (hospitalId === 'demo@lumina.ai' && password === 'demo123') {
      sessionStorage.setItem('isAuthenticated', 'true');
      setIsAuthenticated(true);
      setCurrentScreen('dashboard');
    } else {
      alert('Invalid credentials. Use:\nHospital ID: demo@lumina.ai\nPassword: demo123');
    }
  };

  const handleUpload = useCallback(async (file: File, patientName: string, patientAge: string) => {
    setIsAnalyzing(true);
    try {
      const formData = new FormData();
      formData.append('file', file);

      const res = await fetch(`${API_BASE}/predict`, {
        method: 'POST',
        body: formData,
      });

      if (!res.ok) {
        const text = await res.text();
        throw new Error(text || `Request failed (${res.status})`);
      }

      const data = (await res.json()) as Record<string, unknown>;
      const originalDataUrl = await fileToDataUrl(file);
      const analysisData: AnalysisData = {
        ...mapPredictResponseToAnalysis(data),
        originalImage: originalDataUrl,
      };
      const patientAnalysis = analysisDataToPatientAnalysis(analysisData);

      const id = `PAT-${patientIdSeqRef.current++}`;

      const record: PatientRecord = {
        id,
        name: patientName.trim(),
        age: patientAge.trim(),
        date: new Date().toISOString(),
        originalImage: originalDataUrl,
        heatmapImage: analysisData.heatmapImage,
        analysis: patientAnalysis,
      };

      const confNum =
        typeof data.confidence === 'number'
          ? data.confidence
          : parseFloat(String(data.confidence ?? 0));
      const sizeCm =
        typeof data.size_cm === 'number' ? data.size_cm : parseFloat(String(data.size_cm ?? 0));
      const stageStr = String(data.stage ?? '');

      await persistPatientToApi(record, confNum, sizeCm, stageStr);

      setPatientDatabase((prev) => {
        const filtered = prev.filter((p) => p.id !== id);
        return [record, ...filtered];
      });
      setActivePatientId(id);
      setCurrentAnalysis(analysisData);
      setAiAccuracy((prev) => Math.min(100, prev + 0.1));
      setCurrentScreen('analysis');
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Analysis request failed.';
      alert(`Could not analyze the scan. ${message}`);
    } finally {
      setIsAnalyzing(false);
    }
  }, []);

  const handleVerifyAndAccept = () => {
    if (!currentAnalysis || !activePatientId) return;

    setPatientDatabase((prev) =>
      prev.map((p) =>
        p.id === activePatientId
          ? {
              ...p,
              analysis: {
                ...p.analysis,
                notes: [p.analysis.notes, 'Verified by clinician.'].filter(Boolean).join(' ').trim(),
              },
            }
          : p
      )
    );

    setTotalScans((prev) => prev + 1);
    setCurrentScreen('dashboard');
  };

  const handleCorrection = () => {
    setCurrentScreen('correction');
  };

  const handleCorrectionSubmit = async (newDiagnosis: string, notes: string) => {
    if (!activePatientId) return;

    const note = notes.trim();
    const prior = patientDatabase.find((p) => p.id === activePatientId);
    if (!prior) return;

    const impressionSuffix = ` Clinician correction: ${newDiagnosis}.${note ? ` Notes: ${note}` : ''}`;
    const nextRecord: PatientRecord = {
      ...prior,
      analysis: {
        ...prior.analysis,
        prediction: newDiagnosis,
        predictionColor: predictionToColor(newDiagnosis),
        confidence: '100% (Human Verified)',
        notes: note || prior.analysis.notes,
        reportDetails: {
          ...prior.analysis.reportDetails,
          Impression: `${prior.analysis.reportDetails.Impression}${impressionSuffix}`,
        },
      },
    };

    setPatientDatabase((prev) => prev.map((p) => (p.id === activePatientId ? nextRecord : p)));

    setCurrentAnalysis((prev) =>
      prev
        ? {
            ...prev,
            prediction: newDiagnosis,
            predictionColor: predictionToColor(newDiagnosis),
            confidence: '100% (Human Verified)',
            notes: note || prev.notes,
            reportDetails: nextRecord.analysis.reportDetails,
          }
        : null
    );

    try {
      await persistPatientToApi(
        nextRecord,
        1.0,
        parseLesionSizeCm(nextRecord.analysis.lesionSize),
        nextRecord.analysis.subtype
      );
    } catch {
      /* ignore persist failure */
    }

    setCurrentScreen('patient-list');
  };

  const handleCorrectionCancel = () => {
    setCurrentScreen('analysis');
  };

  const handleNavigate = (screen: string) => {
    if (isValidScreen(screen)) {
      setCurrentScreen(screen);
    }
  };

  const handleViewReport = (patient: PatientRecord) => {
    setSelectedReportPatientId(patient.id);
    setCurrentScreen('diagnostic-report');
  };

  const handleDeletePatient = useCallback(async (id: string) => {
    try {
      const res = await fetch(`${API_BASE}/patients/${encodeURIComponent(id)}`, {
        method: 'DELETE',
      });
      if (!res.ok) {
        const text = await res.text();
        throw new Error(text || `Delete failed (${res.status})`);
      }
      setPatientDatabase((prev) => prev.filter((p) => p.id !== id));
      setActivePatientId((prev) => (prev === id ? null : prev));
      setSelectedReportPatientId((prev) => (prev === id ? null : prev));
    } catch (err) {
      const msg = err instanceof Error ? err.message : 'Delete request failed.';
      alert(`Could not delete patient record. ${msg}`);
    }
  }, []);

  const handleBackToPatientList = () => {
    setSelectedReportPatientId(null);
    setCurrentScreen('patient-list');
  };

  const handleUpdateUserName = (newName: string) => {
    setUserName(newName);
  };

  if (currentScreen === 'diagnostic-report' && selectedReportPatient) {
    return (
      <DiagnosticReport patient={selectedReportPatient} onBack={handleBackToPatientList} />
    );
  }

  if (currentScreen === 'diagnostic-report' && !selectedReportPatient) {
    return null;
  }

  if (!isAuthenticated) {
    return (
      <div className="min-h-screen bg-gray-100 dark:bg-gray-900 flex items-center justify-center p-4 overflow-hidden relative">
        <motion.div
          className="absolute w-32 h-32 rounded-full bg-[#007AFF] opacity-10"
          animate={{
            x: [0, 100, 0],
            y: [0, -80, 0],
          }}
          transition={{
            duration: 8,
            repeat: Infinity,
            ease: 'easeInOut',
          }}
          style={{ top: '10%', left: '10%' }}
        />

        <motion.div
          className="absolute w-24 h-24 rounded-full bg-[#007AFF] opacity-10"
          animate={{
            x: [0, -60, 0],
            y: [0, 100, 0],
          }}
          transition={{
            duration: 10,
            repeat: Infinity,
            ease: 'easeInOut',
          }}
          style={{ top: '60%', right: '15%' }}
        />

        <motion.div
          className="absolute w-40 h-40 rounded-full bg-[#007AFF] opacity-5"
          animate={{
            x: [0, -50, 0],
            y: [0, 60, 0],
          }}
          transition={{
            duration: 12,
            repeat: Infinity,
            ease: 'easeInOut',
          }}
          style={{ top: '20%', right: '5%' }}
        />

        <motion.div
          className="absolute w-20 h-20 rounded-full bg-[#007AFF] opacity-10"
          animate={{
            x: [0, 80, 0],
            y: [0, -50, 0],
          }}
          transition={{
            duration: 9,
            repeat: Infinity,
            ease: 'easeInOut',
          }}
          style={{ bottom: '15%', left: '20%' }}
        />

        <motion.div
          className="bg-white dark:bg-gray-800 rounded-lg shadow-md w-full max-w-md p-8 relative z-10 border border-transparent dark:border-gray-700"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6 }}
        >
          <div className="flex flex-col items-center mb-8">
            <div className="w-16 h-16 bg-[#007AFF] rounded-full flex items-center justify-center mb-3">
              <Activity className="w-8 h-8 text-white" strokeWidth={2} />
            </div>
            <h1 className="text-gray-900 dark:text-white">Lumina Breast AI</h1>
            <p className="text-gray-500 dark:text-gray-400 text-sm mt-2 text-center">Illuminating the path to precise diagnosis.</p>
          </div>

          <form className="space-y-5" onSubmit={handleLogin}>
            <div>
              <label htmlFor="hospital-id" className="block text-gray-700 dark:text-gray-300 mb-2">
                Hospital ID
              </label>
              <input
                type="text"
                id="hospital-id"
                name="hospital-id"
                className="w-full px-4 py-2.5 border border-gray-300 dark:bg-gray-700 dark:text-white dark:border-gray-600 rounded-md focus:outline-none focus:ring-2 focus:ring-[#007AFF] focus:border-transparent transition-all"
                placeholder="demo@lumina.ai"
              />
            </div>

            <div>
              <label htmlFor="password" className="block text-gray-700 dark:text-gray-300 mb-2">
                Password
              </label>
              <input
                type="password"
                id="password"
                name="password"
                className="w-full px-4 py-2.5 border border-gray-300 dark:bg-gray-700 dark:text-white dark:border-gray-600 rounded-md focus:outline-none focus:ring-2 focus:ring-[#007AFF] focus:border-transparent transition-all"
                placeholder="demo123"
              />
            </div>

            <button
              type="submit"
              className="w-full bg-[#007AFF] text-white py-3 rounded-md hover:bg-[#0062CC] transition-colors mt-6"
            >
              Secure Login
            </button>
          </form>

          <div className="mt-8 pt-6 border-t border-gray-200 dark:border-gray-700">
            <p className="text-center text-gray-500 dark:text-gray-400 text-sm">
              System Status: <span className="text-green-600 dark:text-green-400">Online</span> | v1.4
            </p>
          </div>
        </motion.div>
      </div>
    );
  }

  return (
    <Layout
      currentScreen={currentScreen}
      onNavigate={handleNavigate}
      userName={userName}
      isDimmed={currentScreen === 'correction'}
      isDarkMode={isDarkMode}
      toggleTheme={toggleTheme}
    >
      {currentScreen === 'dashboard' && (
        <Dashboard
          onUpload={handleUpload}
          recentCases={recentCases}
          totalScans={totalScans}
          aiAccuracy={aiAccuracy}
          userName={userName}
          isAnalyzing={isAnalyzing}
        />
      )}
      {currentScreen === 'patient-list' && (
        <PatientList
          patientDatabase={patientDatabase}
          onViewReport={handleViewReport}
          onDeletePatient={handleDeletePatient}
        />
      )}
      {currentScreen === 'model-stats' && <ModelStats aiAccuracy={aiAccuracy} />}
      {currentScreen === 'settings' && (
        <Settings onNavigate={handleNavigate} userName={userName} onUpdateUserName={handleUpdateUserName} />
      )}
      {currentScreen === 'analysis' && currentAnalysis && (
        <AnalysisResults
          analysisData={currentAnalysis}
          onVerifyAndAccept={handleVerifyAndAccept}
          onCorrection={handleCorrection}
          aiAccuracy={aiAccuracy}
        />
      )}
      {currentScreen === 'correction' && (
        <CorrectionMode
          currentPrediction={currentAnalysis?.prediction ?? 'Normal'}
          originalImage={activePatient?.originalImage}
          onSubmit={handleCorrectionSubmit}
          onCancel={handleCorrectionCancel}
        />
      )}
    </Layout>
  );
}
