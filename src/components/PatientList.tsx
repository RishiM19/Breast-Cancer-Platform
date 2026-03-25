import { Search, Filter, ChevronDown, Trash2 } from 'lucide-react';
import { motion } from 'motion/react';
import { useMemo, useState } from 'react';
import type { PatientRecord } from '../types/patient';

interface PatientListProps {
  patientDatabase: PatientRecord[];
  onViewReport: (patient: PatientRecord) => void;
  onDeletePatient: (id: string) => void;
}

export function PatientList({ patientDatabase, onViewReport, onDeletePatient }: PatientListProps) {
  const [searchQuery, setSearchQuery] = useState('');
  const [currentPage] = useState(1);
  const totalPages = Math.max(1, Math.ceil(patientDatabase.length / 10) || 1);
  const [showFilterMenu, setShowFilterMenu] = useState(false);
  const [statusFilter, setStatusFilter] = useState<string>('all');

  const getStatusColor = (color: string) => {
    switch (color) {
      case 'red':
        return 'bg-red-100 text-red-800 dark:bg-red-950/60 dark:text-red-300';
      case 'green':
        return 'bg-green-100 text-green-800 dark:bg-green-950/60 dark:text-green-300';
      case 'orange':
        return 'bg-orange-100 text-orange-800 dark:bg-orange-950/50 dark:text-orange-300';
      case 'grey':
        return 'bg-gray-100 text-gray-800 dark:bg-gray-700 dark:text-gray-200';
      default:
        return 'bg-gray-100 text-gray-800 dark:bg-gray-700 dark:text-gray-200';
    }
  };

  const diagnosisStatusForRow = (p: PatientRecord) => p.analysis.prediction;

  const statusColorForRow = (p: PatientRecord): 'red' | 'green' | 'orange' | 'grey' => {
    const c = p.analysis.predictionColor;
    if (c === 'red') return 'red';
    if (c === 'green') return 'green';
    if (c === 'blue') return 'grey';
    return 'grey';
  };

  const lastScanDisplay = (p: PatientRecord) => {
    try {
      return new Date(p.date).toLocaleDateString('en-US', {
        month: 'short',
        day: 'numeric',
        year: 'numeric',
      });
    } catch {
      return p.date;
    }
  };

  const filteredPatients = useMemo(() => {
    return patientDatabase.filter((patient) => {
      const matchesSearch =
        patient.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
        patient.id.toLowerCase().includes(searchQuery.toLowerCase());

      const d = diagnosisStatusForRow(patient);
      const matchesStatus =
        statusFilter === 'all' ||
        (statusFilter === 'malignant' && d === 'Malignant') ||
        (statusFilter === 'benign' && d === 'Benign') ||
        (statusFilter === 'normal' && d === 'Normal');

      return matchesSearch && matchesStatus;
    });
  }, [patientDatabase, searchQuery, statusFilter]);

  return (
    <div className="p-8">
      <motion.div
        className="mb-6"
        initial={{ y: -20, opacity: 0 }}
        animate={{ y: 0, opacity: 1 }}
        transition={{ duration: 0.5, delay: 0.1 }}
      >
        <div className="mb-6">
          <h1 className="text-gray-900 dark:text-white mb-1">Patient Database</h1>
          <p className="text-gray-500 dark:text-gray-400">Manage and review patient records</p>
        </div>

        <div className="flex items-center gap-4">
          <div className="flex-1 relative">
            <Search className="absolute left-4 top-1/2 transform -translate-y-1/2 w-5 h-5 text-gray-400 dark:text-gray-500" />
            <input
              type="text"
              placeholder="Search by Name or ID"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="w-full pl-12 pr-4 py-3 border border-gray-200 dark:border-gray-600 rounded-lg focus:outline-none focus:ring-2 focus:ring-[#007AFF] focus:border-transparent transition-all bg-white dark:bg-gray-700 dark:text-white"
            />
          </div>
          <div className="relative">
            <button
              onClick={() => setShowFilterMenu(!showFilterMenu)}
              className="px-6 py-3 border border-gray-200 dark:border-gray-600 rounded-lg hover:bg-gray-50 dark:hover:bg-gray-700 transition-colors flex items-center gap-2 bg-white dark:bg-gray-800"
            >
              <Filter className="w-5 h-5 text-gray-600 dark:text-gray-300" />
              <span className="text-gray-700 dark:text-gray-200">Filter</span>
              <ChevronDown className="w-5 h-5 text-gray-600 dark:text-gray-300" />
            </button>
            {showFilterMenu && (
              <div className="absolute right-0 mt-2 bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-lg shadow-lg z-50 min-w-[160px]">
                <button
                  onClick={() => {
                    setStatusFilter('all');
                    setShowFilterMenu(false);
                  }}
                  className={`w-full text-left px-4 py-2 text-gray-700 dark:text-gray-200 hover:bg-gray-100 dark:hover:bg-gray-700 rounded-t-lg ${
                    statusFilter === 'all' ? 'bg-gray-100 dark:bg-gray-700' : ''
                  }`}
                >
                  All
                </button>
                <button
                  onClick={() => {
                    setStatusFilter('malignant');
                    setShowFilterMenu(false);
                  }}
                  className={`w-full text-left px-4 py-2 text-gray-700 dark:text-gray-200 hover:bg-gray-100 dark:hover:bg-gray-700 ${
                    statusFilter === 'malignant' ? 'bg-gray-100 dark:bg-gray-700' : ''
                  }`}
                >
                  Malignant
                </button>
                <button
                  onClick={() => {
                    setStatusFilter('benign');
                    setShowFilterMenu(false);
                  }}
                  className={`w-full text-left px-4 py-2 text-gray-700 dark:text-gray-200 hover:bg-gray-100 dark:hover:bg-gray-700 ${
                    statusFilter === 'benign' ? 'bg-gray-100 dark:bg-gray-700' : ''
                  }`}
                >
                  Benign
                </button>
                <button
                  onClick={() => {
                    setStatusFilter('normal');
                    setShowFilterMenu(false);
                  }}
                  className={`w-full text-left px-4 py-2 text-gray-700 dark:text-gray-200 hover:bg-gray-100 dark:hover:bg-gray-700 rounded-b-lg ${
                    statusFilter === 'normal' ? 'bg-gray-100 dark:bg-gray-700' : ''
                  }`}
                >
                  Normal
                </button>
              </div>
            )}
          </div>
        </div>
      </motion.div>

      <motion.div
        initial={{ y: 20, opacity: 0 }}
        animate={{ y: 0, opacity: 1 }}
        transition={{ duration: 0.5, delay: 0.2 }}
      >
        <div className="rounded-xl shadow-sm border border-gray-200/90 dark:border-gray-600 overflow-hidden bg-white/95 dark:bg-gray-800/95 backdrop-blur-sm dark:shadow-gray-950/30">
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead className="bg-gray-100/90 dark:bg-gray-900/70 border-b border-gray-200 dark:border-gray-600">
                <tr>
                  <th className="px-6 py-4 text-left text-xs font-medium text-gray-700 dark:text-gray-200">Patient ID</th>
                  <th className="px-6 py-4 text-left text-xs font-medium text-gray-700 dark:text-gray-200">Name</th>
                  <th className="px-6 py-4 text-left text-xs font-medium text-gray-700 dark:text-gray-200">Age</th>
                  <th className="px-6 py-4 text-left text-xs font-medium text-gray-700 dark:text-gray-200">Last Scan Date</th>
                  <th className="px-6 py-4 text-left text-xs font-medium text-gray-700 dark:text-gray-200">Diagnosis Status</th>
                  <th className="px-6 py-4 text-left text-xs font-medium text-gray-700 dark:text-gray-200">Actions</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-gray-200 dark:divide-gray-600 bg-white dark:bg-gray-800/80">
                {filteredPatients.length === 0 ? (
                  <tr>
                    <td colSpan={6} className="px-6 py-12 text-center text-gray-600 dark:text-gray-300 text-sm bg-white dark:bg-gray-800/80">
                      No patient records yet. Upload a scan from the dashboard to add a case.
                    </td>
                  </tr>
                ) : (
                  filteredPatients.map((patient) => (
                    <tr key={patient.id} className="hover:bg-gray-50 dark:hover:bg-gray-700/50 transition-colors">
                      <td className="px-6 py-4 text-gray-900 dark:text-white">{patient.id}</td>
                      <td className="px-6 py-4 text-gray-900 dark:text-white">{patient.name}</td>
                      <td className="px-6 py-4 text-gray-600 dark:text-gray-300">{patient.age}</td>
                      <td className="px-6 py-4 text-gray-600 dark:text-gray-300">{lastScanDisplay(patient)}</td>
                      <td className="px-6 py-4">
                        <span
                          className={`inline-flex px-3 py-1 rounded-full text-xs ${getStatusColor(
                            statusColorForRow(patient)
                          )}`}
                        >
                          {diagnosisStatusForRow(patient)}
                        </span>
                      </td>
                      <td className="px-6 py-4">
                        <div className="flex items-center gap-4">
                          <button
                            type="button"
                            onClick={() => onViewReport(patient)}
                            className="text-[#007AFF] dark:text-blue-400 hover:text-[#0062CC] dark:hover:text-blue-300 hover:underline transition-colors text-sm font-medium"
                          >
                            View Report
                          </button>
                          <button
                            type="button"
                            onClick={() => {
                              const ok = window.confirm('Are you sure you want to delete this patient record?');
                              if (ok) onDeletePatient(patient.id);
                            }}
                            className="inline-flex items-center gap-1 text-red-500 hover:text-red-700 transition-colors text-sm font-medium"
                          >
                            <Trash2 className="w-4 h-4" />
                            Delete
                          </button>
                        </div>
                      </td>
                    </tr>
                  ))
                )}
              </tbody>
            </table>
          </div>

          <div className="px-6 py-4 border-t border-gray-200 dark:border-gray-600 bg-gray-50/80 dark:bg-gray-900/40 flex items-center justify-end">
            <p className="text-sm text-gray-600 dark:text-gray-300">
              Page {currentPage} of {totalPages}
            </p>
          </div>
        </div>
      </motion.div>
    </div>
  );
}
