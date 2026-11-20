// Report-only presentation checks using genuine saved inference envelopes.
// Mutated availability/category fixtures below are explicit presentation tests.
const fs = require('node:fs');
const path = require('node:path');
const assert = require('node:assert/strict');
const esbuild = require('esbuild');
const React = require('react');
const { renderToStaticMarkup } = require('react-dom/server');
const root = path.resolve(__dirname, '..');
const out = path.join(root, 'data/processed/report_redesign_20261120');
fs.mkdirSync(out, { recursive: true });
function load(file) {
  const built = esbuild.buildSync({ entryPoints: [path.join(root, file)], bundle: true, write: false, platform: 'node', format: 'cjs', packages: 'external', jsx: 'automatic', logLevel: 'silent' });
  const m = { exports: {} };
  new Function('require', 'module', 'exports', built.outputFiles[0].text)(require, m, m.exports);
  return m.exports;
}
const { DiagnosticReport } = load('src/components/DiagnosticReport.tsx');
const { mapApiPatientToPatientRecord } = load('src/lib/inferenceAdapter.ts');
const { diagnosticReportStyles } = load('src/components/diagnosticReportStyles.ts');
const read = file => JSON.parse(fs.readFileSync(path.join(root, file), 'utf8'));
const input = read('data/processed/lumina_ui_verdict_upgrade/persistence_inputs.json');
const cases = read('data/processed/lumina_ui_verdict_upgrade/genuine_inference_responses.json');
const original = mapApiPatientToPatientRecord(input.after_review);
const copy = value => JSON.parse(JSON.stringify(value));
const render = patient => renderToStaticMarkup(React.createElement(DiagnosticReport, { patient, onBack() {} }));
const checks = [];
function test(name, fn) { try { fn(); checks.push({ name, status: 'PASSED' }); } catch (error) { checks.push({ name, status: 'FAILED', error: error.stack }); } }
function withResponse(response) { return mapApiPatientToPatientRecord({ ...input.before_review, multi_model_result: response }); }
const textOnly = html => html.replace(/<style>[\s\S]*?<\/style>/g, '').replace(/<[^>]+>/g, ' ');

test('Patient identity and full record identifier are retained', () => {
  const html = render(original);
  assert.ok(html.includes(original.id)); assert.ok(html.includes(original.name));
  for (const label of ['Patient name', 'Age', 'Record created', 'Laterality / location']) assert.ok(html.includes(label));
});
test('Report renders all seven numbered sections', () => {
  const html = render(original);
  assert.equal((html.match(/class="lr-section-heading"/g) || []).length, 7);
  assert.match(html, /Breast Ultrasound Report/); assert.match(html, /Clinical context &amp; examination details/);
});
test('Primary finding remains the saved verdict even when review disagrees', () => {
  const html = render(original);
  assert.ok(html.includes(`The selected model assigned a ${original.analysis.multiModel.research_verdict.prediction} image class`));
  assert.match(html, /Independent human assessment: Malignant/);
});
test('Genuine malignant result is provisional and does not invent cancer histology', () => {
  const response = cases.find(c => c.response.research_verdict?.prediction === 'Malignant').response;
  const html = render(withResponse(response));
  assert.match(html, /Malignant image class/); assert.match(html, /cancer is not confirmed/);
  assert.match(html, /Specific cancer type \/ histology/); assert.match(html, /Not determined by these image classifiers/);
  assert.doesNotMatch(html, /invasive ductal|lobular carcinoma|Stage II|Grade 2/);
});
test('Normal fixture does not imply a reliable negative or absence of cancer', () => {
  const p = copy(original); p.analysis.multiModel.research_verdict.prediction = 'Normal';
  assert.match(render(p), /does not establish a reliable negative finding or exclude malignancy/);
});
test('All four supported BI-RADS labels have their own category terminology', () => {
  for (const [category, description] of Object.entries({ '2': 'Benign findings', '3': 'Probably benign findings', '4': 'Suspicious abnormality', '5': 'Highly suggestive of malignancy' })) {
    const p = copy(original); p.analysis.multiModel.models['BUS-BRA BI-RADS'].prediction = category;
    assert.ok(render(p).includes(description));
  }
});
test('Missing BI-RADS never borrows the pathology result', () => {
  const p = copy(original); delete p.analysis.multiModel.models['BUS-BRA BI-RADS'];
  const html = render(p); assert.match(html, /A BI-RADS category was not produced/);
  assert.doesNotMatch(html, /class="lr-category-scores"/);
});
test('Unavailable primary finding is not replaced by another model', () => {
  const p = copy(original); Object.assign(p.analysis.multiModel.research_verdict, { status: 'unavailable', prediction: null, primary_model_score: null });
  const html = render(p); assert.match(html, /No stored primary research finding/); assert.doesNotMatch(html, /The selected model assigned a/);
});
test('Old contracts keep individual results without computing a new primary finding', () => {
  const html = render(withResponse(cases.find(c => c.contract_version === '1.0.0').response));
  assert.match(html, /No stored primary research finding/); assert.match(html, /has not been reclassified/);
});
test('Pathology table preserves each own prediction and score', () => {
  const html = render(original); const table = html.match(/<table class="lr-model-table">[\s\S]*?<\/table>/)[0];
  for (const name of ['BUSI', 'UCLM', 'BUS-BRA']) {
    const m = original.analysis.multiModel.models[name];
    assert.ok(table.includes(name)); assert.ok(table.includes(`<td>${m.prediction}</td>`));
    assert.ok(table.includes(`<td>${(m.score * 100).toFixed(1)}%</td>`));
  }
  assert.doesNotMatch(table, /BUS-BRA BI-RADS/); assert.match(table, /Not supported/);
});
test('Zero model score is displayed as zero rather than missing', () => {
  const p = copy(original); p.analysis.multiModel.models.BUSI.score = 0;
  assert.match(render(p), /<td>0\.0%<\/td>/);
});
test('Unavailable model hides stale prediction and probability values', () => {
  const p = copy(original); const m = p.analysis.multiModel.models.BUSI; m.status = 'error'; m.prediction = 'STALE-LABEL'; m.score = .123;
  assert.doesNotMatch(render(p), /STALE-LABEL|<td>12\.3%<\/td>/);
});
test('Unsupported clinical fields are explicitly unassessed', () => {
  const html = render(original);
  for (const label of ['Tumour grade / cancer stage', 'Lymph node involvement', 'Receptor status / molecular subtype', 'Biopsy / histopathology']) assert.ok(html.includes(label));
  assert.match(html, /Not measured by these classifiers/);
});
test('Saved findings and independent review history are retained', () => {
  const html = render(original);
  assert.match(html, /View review history/); assert.match(html, /Identity unverified/);
  for (const text of Object.values(original.analysis.reportDetails)) if (text) assert.ok(textOnly(html).includes(text.replaceAll('&', '&amp;')));
});
test('Historical record and unavailable images render without invented outputs', () => {
  const p = mapApiPatientToPatientRecord({ ...input.before_review, multi_model_result: undefined, prediction: 'Historical test label', original_image: '', heatmap_image: '' });
  const html = render(p); assert.match(html, /Historical test label/); assert.match(html, /Historical Analysis Record/);
  assert.match(html, /No stored image available/); assert.match(html, /Grad-CAM unavailable/);
});
test('Historical BI-RADS output remains present separately from pathology', () => {
  const p = copy(original); p.analysis.recordKind = 'historical';
  const html = render(p); assert.match(html, /Historical BI-RADS output retained as stored/);
  assert.match(html, /aria-label="Historical BI-RADS class scores"/);
  assert.match(html, /BUS-BRA BI-RADS/);
});
test('Unknown dates and blank identity fields have readable placeholders', () => {
  const p = copy(original); p.date = 'not-a-date'; p.name = ''; p.age = '';
  const html = render(p); assert.match(html, /Date unavailable/); assert.doesNotMatch(html, /Invalid Date/);
  assert.match(html, /<dd>Not recorded<\/dd>/);
});
test('Patient and stored note content is escaped', () => {
  const p = copy(original); p.name = '<script>alert(1)</script>'; p.analysis.reportDetails.Findings = '<img src=x onerror=alert(1)>';
  const html = render(p); assert.doesNotMatch(html, /<script>alert|<img src=x/); assert.match(html, /&lt;script&gt;/);
});
test('Rendering is pure and preserves all model envelopes', () => {
  const p = copy(original); const before = JSON.stringify(p); render(p); assert.equal(JSON.stringify(p), before);
});
test('Print content retains both original and Grad-CAM independently of screen toggle', () => {
  const html = render(original);
  assert.match(html, /Original ultrasound for printing/); assert.match(html, /Stored Grad-CAM attribution for printing/);
  assert.match(html, /AI-predicted binary lesion mask/); assert.match(html, /SHA-256/);
});
test('Print CSS is report scoped and keeps controls out of the document', () => {
  assert.match(diagnosticReportStyles, /@media print/); assert.match(diagnosticReportStyles, /size:A4/);
  assert.match(diagnosticReportStyles, /\.lumina-report \.research-report-actions/);
  assert.doesNotMatch(diagnosticReportStyles, /(?:^|\n)\s*(?:body|:root|\.dark|\.l-app)\s*\{/);
});
const failed = checks.filter(c => c.status === 'FAILED');
fs.writeFileSync(path.join(out, 'report_test_results.json'), JSON.stringify({ tests_run: checks.length, passed: checks.length - failed.length, failed: failed.length, checks, database_accessed: false, models_run: false }, null, 2));
console.log(JSON.stringify({ tests_run: checks.length, passed: checks.length - failed.length, failed }, null, 2));
if (failed.length) process.exitCode = 1;
