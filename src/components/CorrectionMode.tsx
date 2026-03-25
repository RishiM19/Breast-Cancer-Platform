import { Square, Move, Pen, Eraser } from 'lucide-react';
import { useState, useRef, MouseEvent, useEffect } from 'react';
import { motion } from 'motion/react';

const DIAGNOSIS_OPTIONS = ['Benign', 'Malignant', 'Normal'] as const;

interface CorrectionModeProps {
  currentPrediction: string;
  /** Object URL for the active scan — no stock imagery */
  originalImage?: string;
  onSubmit: (newDiagnosis: string, notes: string) => void;
  onCancel: () => void;
}

type ResizeHandle = 'tl' | 'tr' | 'bl' | 'br' | null;

export function CorrectionMode({ currentPrediction, originalImage, onSubmit, onCancel }: CorrectionModeProps) {
  const initialDx =
    DIAGNOSIS_OPTIONS.find((o) => o.toLowerCase() === currentPrediction.toLowerCase()) ?? 'Normal';
  const [correctedDiagnosis, setCorrectedDiagnosis] = useState<string>(initialDx);
  const [correctionNotes, setCorrectionNotes] = useState('');

  useEffect(() => {
    const match = DIAGNOSIS_OPTIONS.find((o) => o.toLowerCase() === currentPrediction.toLowerCase());
    setCorrectedDiagnosis(match ?? 'Normal');
  }, [currentPrediction]);

  const [mouseCoords, setMouseCoords] = useState({ x: 0, y: 0 });
  const [boundingBox, setBoundingBox] = useState({ x: 245, y: 112, w: 145, h: 130 });
  const [activeTool, setActiveTool] = useState<'cursor' | 'box' | 'freehand' | 'eraser'>('box');
  const [isDragging, setIsDragging] = useState(false);
  const [isResizing, setIsResizing] = useState<ResizeHandle>(null);
  const [dragStart, setDragStart] = useState({ x: 0, y: 0 });
  const canvasRef = useRef<HTMLDivElement>(null);

  const handleMouseMove = (e: MouseEvent<HTMLDivElement>) => {
    if (!canvasRef.current) return;
    const rect = canvasRef.current.getBoundingClientRect();
    const x = Math.round(e.clientX - rect.left);
    const y = Math.round(e.clientY - rect.top);
    setMouseCoords({ x, y });

    // Handle dragging
    if (isDragging && activeTool === 'cursor') {
      const dx = x - dragStart.x;
      const dy = y - dragStart.y;
      setBoundingBox(prev => ({
        ...prev,
        x: Math.max(0, Math.min(rect.width - prev.w, prev.x + dx)),
        y: Math.max(0, Math.min(rect.height - prev.h, prev.y + dy)),
      }));
      setDragStart({ x, y });
    }

    // Handle resizing
    if (isResizing && activeTool === 'cursor') {
      const newBox = { ...boundingBox };
      
      switch (isResizing) {
        case 'tl':
          newBox.w = boundingBox.w + (boundingBox.x - x);
          newBox.h = boundingBox.h + (boundingBox.y - y);
          newBox.x = x;
          newBox.y = y;
          break;
        case 'tr':
          newBox.w = x - boundingBox.x;
          newBox.h = boundingBox.h + (boundingBox.y - y);
          newBox.y = y;
          break;
        case 'bl':
          newBox.w = boundingBox.w + (boundingBox.x - x);
          newBox.h = y - boundingBox.y;
          newBox.x = x;
          break;
        case 'br':
          newBox.w = x - boundingBox.x;
          newBox.h = y - boundingBox.y;
          break;
      }

      // Ensure minimum size
      if (newBox.w > 30 && newBox.h > 30) {
        setBoundingBox(newBox);
      }
    }
  };

  const handleMouseDown = (e: MouseEvent<HTMLDivElement>, handle?: ResizeHandle) => {
    if (activeTool !== 'cursor') return;
    
    e.stopPropagation();
    
    if (handle) {
      setIsResizing(handle);
    } else {
      setIsDragging(true);
      const rect = canvasRef.current?.getBoundingClientRect();
      if (rect) {
        setDragStart({
          x: e.clientX - rect.left,
          y: e.clientY - rect.top,
        });
      }
    }
  };

  const handleMouseUp = () => {
    setIsDragging(false);
    setIsResizing(null);
  };

  return (
    <div className="flex flex-1 min-h-0 w-full flex-col lg:flex-row overflow-hidden">
        {/* Center Canvas */}
        <motion.div 
          className="flex-1 p-6"
          initial={{ opacity: 0, x: -20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ duration: 0.5 }}
        >
          <div 
            className="h-full bg-[#1C1C1E] rounded-xl relative overflow-hidden" 
            ref={canvasRef} 
            onMouseMove={handleMouseMove}
            onMouseUp={handleMouseUp}
            onMouseLeave={handleMouseUp}
          >
            {/* Ultrasound Image */}
            <div className="absolute inset-0 flex items-center justify-center bg-black/40">
              {originalImage ? (
                <img
                  src={originalImage}
                  alt="Ultrasound for annotation"
                  className="max-w-full max-h-full object-contain"
                />
              ) : (
                <span className="text-gray-500 dark:text-gray-400 text-sm px-4 text-center">No scan image loaded for this session.</span>
              )}
            </div>

            {/* Green Bounding Box Annotation */}
            <div
              className={`absolute border-[3px] border-[#32D74B] bg-[#32D74B] bg-opacity-10 ${activeTool === 'cursor' ? 'cursor-move' : 'pointer-events-none'}`}
              style={{
                left: `${boundingBox.x}px`,
                top: `${boundingBox.y}px`,
                width: `${boundingBox.w}px`,
                height: `${boundingBox.h}px`,
                pointerEvents: activeTool === 'cursor' ? 'auto' : 'none',
              }}
              onMouseDown={(e) => handleMouseDown(e as any)}
            >
              {/* Resize Handles */}
              {activeTool === 'cursor' && (
                <>
                  <div 
                    className="absolute -top-1 -left-1 w-3 h-3 bg-white border-2 border-[#32D74B] cursor-nw-resize"
                    style={{ pointerEvents: 'auto' }}
                    onMouseDown={(e) => handleMouseDown(e as any, 'tl')}
                  ></div>
                  <div 
                    className="absolute -top-1 -right-1 w-3 h-3 bg-white border-2 border-[#32D74B] cursor-ne-resize"
                    style={{ pointerEvents: 'auto' }}
                    onMouseDown={(e) => handleMouseDown(e as any, 'tr')}
                  ></div>
                  <div 
                    className="absolute -bottom-1 -left-1 w-3 h-3 bg-white border-2 border-[#32D74B] cursor-sw-resize"
                    style={{ pointerEvents: 'auto' }}
                    onMouseDown={(e) => handleMouseDown(e as any, 'bl')}
                  ></div>
                  <div 
                    className="absolute -bottom-1 -right-1 w-3 h-3 bg-white border-2 border-[#32D74B] cursor-se-resize"
                    style={{ pointerEvents: 'auto' }}
                    onMouseDown={(e) => handleMouseDown(e as any, 'br')}
                  ></div>
                </>
              )}
            </div>

            {/* Floating Toolbar */}
            <div className="absolute top-6 left-1/2 transform -translate-x-1/2 bg-white dark:bg-gray-800 rounded-full px-4 py-3 shadow-lg flex items-center gap-2 border border-gray-200 dark:border-gray-700">
              <button
                onClick={() => setActiveTool('cursor')}
                className={`p-2 rounded-lg transition-colors ${
                  activeTool === 'cursor' ? 'bg-[#007AFF] text-white' : 'text-gray-600 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-700'
                }`}
                title="Cursor"
              >
                <Move className="w-5 h-5" />
              </button>
              
              <button
                onClick={() => setActiveTool('box')}
                className={`p-2 rounded-lg transition-colors ${
                  activeTool === 'box' ? 'bg-[#007AFF] text-white' : 'text-gray-600 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-700'
                }`}
                title="Bounding Box"
              >
                <Square className="w-5 h-5" />
              </button>
              
              <button
                onClick={() => setActiveTool('freehand')}
                className={`p-2 rounded-lg transition-colors ${
                  activeTool === 'freehand' ? 'bg-[#007AFF] text-white' : 'text-gray-600 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-700'
                }`}
                title="Freehand"
              >
                <Pen className="w-5 h-5" />
              </button>
              
              <button
                onClick={() => setActiveTool('eraser')}
                className={`p-2 rounded-lg transition-colors ${
                  activeTool === 'eraser' ? 'bg-[#007AFF] text-white' : 'text-gray-600 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-700'
                }`}
                title="Eraser"
              >
                <Eraser className="w-5 h-5" />
              </button>
            </div>
          </div>
        </motion.div>

        {/* Right Panel */}
        <motion.div 
          className="w-96 p-6 bg-white dark:bg-gray-800 border-l border-gray-200 dark:border-gray-700 overflow-auto"
          initial={{ opacity: 0, x: 20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ duration: 0.5 }}
        >
          <div className="space-y-6">
            {/* Header */}
            <div>
              <h2 className="text-gray-900 dark:text-white mb-1">Manual Refinement</h2>
              <p className="text-gray-500 dark:text-gray-400 text-sm">Correct AI predictions with precision</p>
            </div>

            {/* Alert Card */}
            <div className="bg-yellow-50 dark:bg-yellow-950/40 border-2 border-yellow-300 dark:border-yellow-700 rounded-lg p-4">
              <div className="flex items-start gap-3">
                <div className="w-5 h-5 text-yellow-600 mt-0.5">⚠️</div>
                <div>
                  <p className="text-yellow-800 dark:text-yellow-200 text-sm">
                    AI heatmap rejected. Please define exact tumor boundaries.
                  </p>
                </div>
              </div>
            </div>

            {/* Active Tool */}
            <div className="bg-gray-50 dark:bg-gray-700/40 rounded-lg p-4">
              <p className="text-gray-600 dark:text-gray-300 text-sm mb-2">Active Tool</p>
              <p className="text-gray-900 dark:text-white capitalize">{activeTool === 'box' ? 'Bounding Box' : activeTool}</p>
            </div>

            {/* Coordinate Data */}
            <div>
              <p className="text-gray-600 dark:text-gray-300 text-sm mb-2">Live Coordinates</p>
              <div className="bg-gray-900 rounded-lg p-4 font-mono text-sm text-green-400">
                <div className="space-y-1">
                  <div>Mouse: X: {mouseCoords.x}, Y: {mouseCoords.y}</div>
                  <div className="border-t border-gray-700 pt-1 mt-1">
                    Box: X: {boundingBox.x}, Y: {boundingBox.y}
                  </div>
                  <div>Size: W: {boundingBox.w}, H: {boundingBox.h}</div>
                </div>
              </div>
            </div>

            {/* Instructions */}
            <div className="bg-blue-50 dark:bg-blue-950/40 rounded-lg p-4">
              <p className="text-blue-900 dark:text-blue-200 text-sm">
                Use the bounding box tool to precisely outline the lesion area. The system will re-run analysis on the defined region.
              </p>
            </div>

            <div>
              <label htmlFor="corrected-diagnosis" className="block text-gray-600 dark:text-gray-300 text-sm mb-2">
                Corrected diagnosis
              </label>
              <select
                id="corrected-diagnosis"
                value={correctedDiagnosis}
                onChange={(e) => setCorrectedDiagnosis(e.target.value)}
                className="w-full px-4 py-3 border border-gray-200 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 dark:text-white focus:outline-none focus:ring-2 focus:ring-[#007AFF]"
              >
                {DIAGNOSIS_OPTIONS.map((opt) => (
                  <option key={opt} value={opt}>
                    {opt}
                  </option>
                ))}
              </select>
              <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">Current AI prediction: {currentPrediction}</p>
            </div>

            <div>
              <label htmlFor="correction-notes" className="block text-gray-600 dark:text-gray-300 text-sm mb-2">
                Clinician notes
              </label>
              <textarea
                id="correction-notes"
                value={correctionNotes}
                onChange={(e) => setCorrectionNotes(e.target.value)}
                rows={4}
                placeholder="Document rationale for the correction, findings, or follow-up."
                className="w-full px-4 py-3 border border-gray-200 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 dark:text-white placeholder:text-gray-400 dark:placeholder:text-gray-500 focus:outline-none focus:ring-2 focus:ring-[#007AFF] resize-y min-h-[100px]"
              />
            </div>

            {/* Action Buttons */}
            <div className="space-y-3 pt-4">
              <button
                type="button"
                onClick={() => onSubmit(correctedDiagnosis, correctionNotes)}
                className="w-full bg-[#007AFF] hover:bg-[#0062CC] text-white py-4 rounded-xl transition-colors shadow-lg"
              >
                Save correction &amp; update record
              </button>
              
              <button
                onClick={onCancel}
                className="w-full text-[#007AFF] hover:text-[#0062CC] py-2 transition-colors"
              >
                Cancel Correction
              </button>
            </div>
          </div>
        </motion.div>
    </div>
  );
}
