// Execute actual pure TypeScript response-mapping functions with saved real BUSI results.
// No browser, production server, database or frontend source mutation.
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const assert = require('node:assert/strict');
const esbuild = require('esbuild');
const root = path.resolve(__dirname, '..');
const base = path.join(root, 'data/processed/busi_final_reliability_audit');
const out = process.argv[2] ? path.resolve(process.argv[2]) : base;
assert.ok(out === base || out.startsWith(base + path.sep), 'Output must remain within audit root');
function extract(source, name) {
  const pattern = new RegExp('(?:export )?function ' + name + '\\b[\\s\\S]*?^}', 'm');
  const match = source.match(pattern);
  assert.ok(match, 'Missing actual source function ' + name);
  return match[0].replace(/^export /, '');
}
const app = fs.readFileSync(path.join(root, 'src/App.tsx'), 'utf8');
const types = fs.readFileSync(path.join(root, 'src/types/patient.ts'), 'utf8');
const code = [extract(types, 'emptyReportDetails'), extract(types, 'parseReportDetails'),
  ...['normalizeHeatmapDataUrl', 'predictionToColor', 'mapPredictResponseToAnalysis'].map(n => extract(app, n))].join('\n');
const compiled = esbuild.transformSync(code, { loader: 'ts', format: 'cjs' }).code;
const context = vm.createContext({});
vm.runInContext(compiled, context);
// Saved JSON rows are exported by the independent audit without CSV parser dependencies.
const rows = JSON.parse(fs.readFileSync(path.join(out, 'frontend_validation_inputs.json'), 'utf8'));
for (const row of rows) {
  const record = { prediction: row.predicted_class, confidence: row.score, probabilities: JSON.parse(row.probabilities) };
  const payload = { models: { BUSI: record, UCLM: record, 'BUS-BRA': record },
    consensus: { prediction: row.predicted_class }, heatmap_base64: '' };
  const result = context.mapPredictResponseToAnalysis(payload);
  assert.equal(result.multiModel.models.BUSI.prediction, row.predicted_class);
  assert.equal(result.prediction, row.predicted_class);
  assert.deepEqual(Object.keys(result.multiModel.models.BUSI.probabilities), ['Normal', 'Benign', 'Malignant']);
  assert.equal(result.predictionColor, {Normal:'blue', Benign:'green', Malignant:'red'}[row.predicted_class]);
}
const demonstration = context.mapPredictResponseToAnalysis({
  models: { BUSI: { prediction: 'Normal', confidence: .8 } }, consensus: { prediction: 'Benign' }
});
const report = { status: 'PASSED', real_validation_response_cases: rows.length,
  source: 'Actual pure functions extracted from App.tsx and types/patient.ts, transpiled using installed esbuild.',
  class_names_and_color_mapping: 'PASSED', probability_keys: 'PASSED',
  known_score_association: { consensus: demonstration.prediction, displayed_or_stored_score: demonstration.confidence,
    note: 'Consensus label is paired with BUSI score in the adapter. This is a score-semantics issue, not a class-index mapping error.' },
  browser_rendering: 'NOT EXECUTED', production_database_accessed: false };
fs.writeFileSync(path.join(out, 'frontend_mapping_test.json'), JSON.stringify(report, null, 2));
console.log(JSON.stringify(report, null, 2));
