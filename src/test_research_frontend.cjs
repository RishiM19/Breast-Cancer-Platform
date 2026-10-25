// Actual TypeScript adapters and React components, with saved real API payloads.
// Fetch is isolated for persistence-shape tests; no HTTP/database calls occur.
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const assert = require('node:assert/strict');
const esbuild = require('esbuild');
const React = require('react');
const { renderToStaticMarkup } = require('react-dom/server');
const root = path.resolve(__dirname, '..');
const out = process.argv[2] ? path.resolve(process.argv[2]) : path.join(root, 'data/processed/lumina_integration_stage2/frontend_regression');
assert.ok([2,3].some(stage=>out.startsWith(path.join(root,`data/processed/lumina_integration_stage${stage}`)+path.sep)) || out.startsWith(path.join(root,'data/processed/lumina_ui_verdict_upgrade')+path.sep));
fs.mkdirSync(out,{recursive:true});
const checks = [];
function load(relative) {
  const result = esbuild.buildSync({ entryPoints: [path.join(root, relative)], bundle: true, write: false,
    platform: 'node', format: 'cjs', packages: 'external', jsx: 'automatic', logLevel: 'silent' });
  const module = { exports: {} };
  new Function('require', 'module', 'exports', result.outputFiles[0].text)(require, module, module.exports);
  return module.exports;
}
async function test(name, fn) {
  try { await fn(); checks.push({ name, status: 'PASSED' }); }
  catch (error) { checks.push({ name, status: 'FAILED', error: error.stack }); }
}
function copy(value) { return JSON.parse(JSON.stringify(value)); }

(async () => {
  const api = load('src/lib/inferenceAdapter.ts');
  const { AnalysisResults } = load('src/components/AnalysisResults.tsx');
  const { DiagnosticReport } = load('src/components/DiagnosticReport.tsx');
  const { ResearchModelResults } = load('src/components/ResearchModelResults.tsx');
  const real = JSON.parse(fs.readFileSync(path.join(root, 'data/processed/lumina_integration_stage1/real_inference_responses.json'), 'utf8'));
  const saved = JSON.parse(fs.readFileSync(path.join(root, 'data/processed/busi_final_reliability_audit/frontend_validation_inputs.json'), 'utf8'));
  const first = real[0].response;
  const analysis = api.mapPredictResponseToAnalysis(first);
  const patientAnalysis = api.analysisDataToPatientAnalysis(analysis);
  const row = { id: 'INTEGRATION-TEST-ONLY', name: 'Isolated fixture', age: '0', date: '2026-10-25',
    prediction: api.ABSTENTION, confidence: 0, size_cm: 0, stage: 'N/A',
    report_details: first.report_details, original_image: '', heatmap_image: first.heatmap_base64,
    multi_model_result: patientAnalysis.multiModel };

  await test('All four real API responses map to abstention with unavailable confidence', () => {
    assert.equal(real.length, 4);
    for (const r of real) {
      const a = api.mapPredictResponseToAnalysis(r.response);
      assert.equal(a.prediction, api.ABSTENTION); assert.equal(a.confidence, 'Unavailable');
      assert.equal(a.multiModel.models.BUSI.prediction, r.response.models.BUSI.prediction);
      assert.equal(a.multiModel.models['BUS-BRA BI-RADS'].task, 'birads');
    }
  });
  await test('Per-class scores remain attached to each original model label', () => {
    for (const r of real) for (const m of Object.values(api.mapPredictResponseToAnalysis(r.response).multiModel.models)) {
      assert.equal(m.confidence, m.probabilities[m.prediction]);
      assert.equal(m.ordered_class_labels[m.predicted_class_index], m.prediction);
    }
  });
  await test('Live-to-patient conversion preserves the complete versioned model result', () => {
    assert.deepEqual(patientAnalysis.multiModel, first);
  });
  await test('Zero legacy storage placeholder reloads as unavailable, never zero confidence', () => {
    const record = api.mapApiPatientToPatientRecord(row);
    assert.equal(record.analysis.confidence, 'Unavailable'); assert.equal(record.analysis.prediction, api.ABSTENTION);
    assert.equal(record.analysis.multiModel.consensus.confidence, null);
  });
  await test('Human review stays separate from frozen model results after reload', () => {
    const corrected = copy(row);
    corrected.prediction = 'Benign'; corrected.confidence = 1;
    corrected.multi_model_result.human_review = { prediction: 'Benign', notes: 'Isolated review fixture', status: 'recorded' };
    const result = api.mapApiPatientToPatientRecord(corrected);
    assert.equal(result.analysis.prediction, api.ABSTENTION); assert.equal(result.analysis.confidence, 'Unavailable');
    assert.equal(result.analysis.humanReview.prediction, 'Benign'); assert.equal(result.analysis.reviewStatus, 'reviewed');
    assert.deepEqual(result.analysis.multiModel.models, first.models);
    assert.equal(result.analysis.multiModel.consensus.prediction, api.ABSTENTION);
  });
  await test('Borrowed or fabricated consensus score fails closed', () => {
    const invalid = copy(first); invalid.consensus.confidence = first.models.BUSI.confidence;
    assert.throws(() => api.mapPredictResponseToAnalysis(invalid), /score association/);
  });
  await test('Unknown contract versions fail closed', () => {
    const invalid = copy(first); invalid.contract_version = '999.0.0';
    assert.throws(() => api.mapPredictResponseToAnalysis(invalid), /Unsupported/);
  });
  await test('Model score or index mismatch fails closed', () => {
    const invalid = copy(first); invalid.models.BUSI.score = .123;
    assert.throws(() => api.mapPredictResponseToAnalysis(invalid), /score association/);
    const index = copy(first); index.models.BUSI.predicted_class_index = 999;
    assert.throws(() => api.mapPredictResponseToAnalysis(index), /score association/);
  });
  await test('Unavailable model records display missing/error/not-run without invented scores', () => {
    for (const status of ['missing', 'not_run', 'error']) {
      const payload = copy(first); const m = payload.models.BUSI;
      Object.assign(m, { status, prediction: null, score: null, confidence: null, predicted_class_index: null, probabilities: {} });
      const a = api.mapPredictResponseToAnalysis(payload);
      const html = renderToStaticMarkup(React.createElement(ResearchModelResults, { result: a.multiModel }));
      assert.match(html, /Prediction unavailable/); assert.match(html, /Model score: Unavailable/);
    }
  });
  await test('Failed attribution never displays an arbitrary heatmap', () => {
    const payload = copy(first); payload.attribution.status = 'error'; payload.heatmap_base64 = 'fabricated-fixture';
    assert.equal(api.mapPredictResponseToAnalysis(payload).heatmapImage, '');
  });
  await test('No synthetic grade/size/stage is generated from a class label', () => {
    assert.equal(analysis.tumorGrade, 'N/A'); assert.equal(analysis.lesionSize, 'N/A'); assert.equal(analysis.subtype, 'N/A');
    assert.equal(api.mapApiPatientToPatientRecord(row).analysis.tumorGrade, 'N/A');
  });
  await test('All 78 saved historical BUSI records keep their individual mapping without adopting old votes', () => {
    assert.equal(saved.length, 78);
    for (const r of saved) {
      const m = { prediction: r.predicted_class, confidence: r.score, probabilities: JSON.parse(r.probabilities) };
      const a = api.mapPredictResponseToAnalysis({ models: { BUSI: m, UCLM: m, 'BUS-BRA': m }, consensus: { prediction: r.predicted_class } });
      assert.equal(a.multiModel.models.BUSI.prediction, r.predicted_class);
      assert.deepEqual(Object.keys(a.multiModel.models.BUSI.probabilities), ['Normal', 'Benign', 'Malignant']);
      assert.equal(a.prediction, api.ABSTENTION); assert.equal(a.confidence, 'Unavailable');
    }
  });
  await test('Legacy single-model label and its own score remain compatible', () => {
    const a = api.mapPredictResponseToAnalysis({ prediction: 'Benign', confidence: .8 });
    assert.equal(a.prediction, 'Benign'); assert.equal(a.confidence, '80% (uncalibrated)');
  });
  await test('Legacy stored labels remain historical and misleading consensus scores are suppressed', () => {
    const legacy = { ...row, prediction: 'Benign', confidence: .8, multi_model_result: { models: { BUSI: { prediction: 'Normal', confidence: .8 } }, consensus: { prediction: 'Benign' } } };
    const a = api.mapApiPatientToPatientRecord(legacy).analysis;
    assert.equal(a.prediction, 'Benign'); assert.equal(a.confidence, 'Unavailable (legacy consensus score)');
  });
  await test('Older database rows with an empty model JSON object remain readable', () => {
    const record = api.mapApiPatientToPatientRecord({ ...row, prediction: 'Benign', multi_model_result: {} });
    assert.equal(record.analysis.prediction, 'Benign'); assert.equal(record.analysis.multiModel, undefined);
  });
  await test('Legacy abstention without metadata never displays a numeric placeholder as confidence', () => {
    const record = api.mapApiPatientToPatientRecord({ ...row, multi_model_result: {} });
    assert.equal(record.analysis.confidence, 'Unavailable');
    assert.equal(api.mapPredictResponseToAnalysis({ prediction: api.ABSTENTION, confidence: 0 }).confidence, 'Unavailable');
  });
  await test('Analysis React view renders BUSI and separates BI-RADS with abstention semantics', () => {
    const html = renderToStaticMarkup(React.createElement(AnalysisResults, { analysisData: analysis, onVerifyAndAccept() {}, onCorrection() {} }));
    assert.match(html, /Individual pathology research predictions/); assert.match(html, /BI-RADS research category/);
    assert.match(html, /BUSI/); assert.match(html, /Historical baseline/); assert.match(html, /Consensus score: unavailable/);
    fs.writeFileSync(path.join(out, 'analysis_render_snapshot.html'), html);
  });
  await test('Actual versioned saved report bypasses demo and displays real individual results', () => {
    const patient = api.mapApiPatientToPatientRecord(row);
    const html = renderToStaticMarkup(React.createElement(DiagnosticReport, { patient, onBack() {} }));
    assert.match(html, /Research Inference Report/); assert.match(html, /BUS-BRA BI-RADS/);
    assert.match(html, /No validated consensus available/); assert.doesNotMatch(html, /Benign \(Demo Example\)/);
    assert.doesNotMatch(html, /synthetic example content/); assert.doesNotMatch(html, /short-interval follow-up/);
    fs.writeFileSync(path.join(out, 'report_render_snapshot.html'), html);
  });
  await test('Historical saved reports display original stored data without demo substitution', () => {
    const patient = api.mapApiPatientToPatientRecord({ ...row, multi_model_result: undefined });
    const html = renderToStaticMarkup(React.createElement(DiagnosticReport, { patient, onBack() {} }));
    assert.match(html, /Historical Analysis Record/); assert.doesNotMatch(html, /Demo Example|synthetic example content/);
  });
  await test('Persistence payload preserves all four models and null consensus score without database calls', async () => {
    const source = fs.readFileSync(path.join(root, 'src/App.tsx'), 'utf8');
    const match = source.match(/async function persistPatientToApi\b[\s\S]*?^}/m); assert.ok(match);
    const code = esbuild.transformSync(match[0], { loader: 'ts', format: 'cjs' }).code;
    let captured;
    const context = vm.createContext({ API_BASE: 'http://isolated.test', fetch: async (url, options) => {
      captured = JSON.parse(options.body); return { ok: true, status: 200 };
    } });
    vm.runInContext(code, context);
    await context.persistPatientToApi(api.mapApiPatientToPatientRecord(row), null, 0, 'N/A');
    assert.equal(captured.confidence, 0); assert.equal(captured.prediction, api.ABSTENTION);
    assert.deepEqual(captured.multi_model_result.models, first.models);
    assert.equal(captured.multi_model_result.consensus.confidence, null);
    const reloaded = api.mapApiPatientToPatientRecord({ ...row, ...captured });
    assert.equal(reloaded.analysis.confidence, 'Unavailable');
  });
  await test('Persistence rejects a failed save instead of reporting success', async () => {
    const source = fs.readFileSync(path.join(root, 'src/App.tsx'), 'utf8');
    const match = source.match(/async function persistPatientToApi\b[\s\S]*?^}/m);
    const code = esbuild.transformSync(match[0], { loader: 'ts', format: 'cjs' }).code;
    const context = vm.createContext({ API_BASE: 'http://isolated.test', fetch: async () => ({ ok: false, status: 500 }) });
    vm.runInContext(code, context);
    await assert.rejects(context.persistPatientToApi(api.mapApiPatientToPatientRecord(row), null, 0, 'N/A'), /Could not save/);
  });
  const failed = checks.filter(c => c.status !== 'PASSED');
  const result = { status: failed.length ? 'FAILED' : 'PASSED', tests_run: checks.length, passed: checks.length - failed.length,
    checks, real_response_cases: real.length, saved_historical_records: saved.length,
    rendering: 'Actual React server rendering with saved real API responses; not interactive browser testing.',
    database_accessed: false, network_requests: false };
  fs.writeFileSync(path.join(out, 'frontend_test_results.json'), JSON.stringify(result, null, 2));
  console.log(JSON.stringify(result, null, 2));
  if (failed.length) process.exitCode = 1;
})().catch(error => { console.error(error); process.exitCode = 1; });
