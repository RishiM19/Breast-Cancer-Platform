// Real saved API/SQL payloads through actual TypeScript and React code.
// Additional malformed/availability fixtures are explicitly isolated test inputs.
const fs=require('node:fs');const path=require('node:path');const assert=require('node:assert/strict');
const esbuild=require('esbuild');const React=require('react');const {renderToStaticMarkup}=require('react-dom/server');
const root=path.resolve(__dirname,'..');const inputRoot=path.join(root,'data/processed/lumina_integration_stage2');const out=process.env.LUMINA_TEST_ARTIFACT_ROOT?path.resolve(process.env.LUMINA_TEST_ARTIFACT_ROOT):inputRoot;const checks=[];
assert.ok([2,3].some(stage=>out===path.join(root,`data/processed/lumina_integration_stage${stage}`)) || out===path.join(root,'data/processed/lumina_ui_verdict_upgrade'));fs.mkdirSync(out,{recursive:true});
function load(name){const built=esbuild.buildSync({entryPoints:[path.join(root,name)],bundle:true,write:false,platform:'node',format:'cjs',packages:'external',jsx:'automatic',logLevel:'silent'});const m={exports:{}};new Function('require','module','exports',built.outputFiles[0].text)(require,m,m.exports);return m.exports;}
function test(name,fn){try{fn();checks.push({name,status:'PASSED'});}catch(e){checks.push({name,status:'FAILED',error:e.stack});}}
const api=load('src/lib/inferenceAdapter.ts');
const {DiagnosticReport}=load('src/components/DiagnosticReport.tsx');
const {ReviewSummary}=load('src/components/ReviewSummary.tsx');
const {ResearchModelResults}=load('src/components/ResearchModelResults.tsx');
const {Dashboard}=load('src/components/Dashboard.tsx');
const {FeedbackModal}=load('src/components/FeedbackModal.tsx');
const {CorrectionMode}=load('src/components/CorrectionMode.tsx');
const {FeedbackStats}=load('src/components/FeedbackStats.tsx');
const input=JSON.parse(fs.readFileSync(path.join(inputRoot,'persistence_test_inputs.json'),'utf8'));
const before=api.mapApiPatientToPatientRecord(input.before_review);const after=api.mapApiPatientToPatientRecord(input.after_review);
const legacyRow=input.all_rows.find(r=>r.record_kind==='historical');const legacy=api.mapApiPatientToPatientRecord(legacyRow);
const render=(C,props)=>renderToStaticMarkup(React.createElement(C,props));
test('Real inference envelope survives SQL, fetch and frontend reconstruction',()=>assert.deepEqual(before.analysis.multiModel,input.actual_response));
test('All model distributions, class orders, hashes and statuses survive',()=>{
  assert.deepEqual(after.analysis.multiModel.models,input.actual_response.models);
  for(const m of Object.values(after.analysis.multiModel.models)){assert.equal(m.status,'ok');assert.equal(m.confidence,m.probabilities[m.prediction]);assert.ok(m.checkpoint.sha256);assert.equal(m.eligibility.consensus_eligible,false);}
});
test('Source-domain verification warning survives',()=>{
  assert.equal(after.analysis.multiModel.source_domain_verification,'caller_declared_unverified');
  assert.match(render(ResearchModelResults,{result:after.analysis.multiModel}),/Upload source: BUS-BRA \(unverified\)/);
});
test('Zero compatibility storage placeholder is never displayed as consensus confidence',()=>{
  assert.equal(input.after_review.confidence,0);assert.equal(after.analysis.confidence,'Unavailable');assert.equal(after.analysis.multiModel.consensus.score,null);
});
test('Human assessment does not replace original AI label or report details',()=>{
  assert.equal(after.analysis.prediction,api.ABSTENTION);assert.deepEqual(after.analysis.reportDetails,before.analysis.reportDetails);
  assert.equal(after.analysis.humanReview.prediction,'Benign');assert.equal(after.analysis.reviewHistory.length,2);
});
test('Reviewed and unreviewed states differ after actual save/reload',()=>{
  assert.equal(before.analysis.reviewStatus,'unreviewed');assert.equal(after.analysis.reviewStatus,'reviewed');
});
test('Human ID, note, timestamp, null confidence and unverified identity survive',()=>{
  const h=after.analysis.humanReview;assert.equal(h.reviewer_id,'SYNTHETIC-REVIEWER-02');assert.match(h.notes,/Synthetic independent/);
  assert.equal(h.identity_verified,false);assert.equal(h.declared_confidence,null);assert.ok(h.created_at);assert.ok(h.feedback_id);
});
test('Genuine attribution bytes and metadata survive SQL and frontend mapping',()=>{
  assert.equal(after.heatmapImage,input.actual_response.heatmap_base64);assert.deepEqual(after.analysis.multiModel.attribution,input.actual_response.attribution);
});
test('Saved report displays actual model outputs and independent review',()=>{
  const html=render(DiagnosticReport,{patient:after,onBack(){}});assert.match(html,/Research Inference Report/);assert.match(html,/Independent human assessment: Benign/);
  assert.match(html,/No validated consensus available/);assert.match(html,/Saved review entries: 2/);assert.doesNotMatch(html,/Demo Example|Grade 2|1\.2 x 0\.8/);
  fs.writeFileSync(path.join(out,'roundtrip_report_snapshot.html'),html);
});
test('Pathology and BI-RADS are separate; BUSI excluded and UCLM cross-dataset',()=>{
  const html=render(ResearchModelResults,{result:after.analysis.multiModel});assert.match(html,/Individual pathology research predictions/);assert.match(html,/BI-RADS research category/);
  assert.match(html,/Historical baseline/);assert.match(html,/cross dataset/);assert.match(html,/cross-dataset uploads are not supported for voting/);assert.match(html,/uncalibrated/);
});
test('Historical stored data is retained without upgrading or inventing confidence',()=>{
  assert.equal(legacy.analysis.prediction,'Benign');assert.equal(legacy.analysis.recordKind,'historical');assert.equal(legacy.analysis.confidence,'Unavailable (historical)');
  const html=render(DiagnosticReport,{patient:legacy,onBack(){}});assert.match(html,/Historical Analysis Record/);assert.match(html,/Historical synthetic text only/);
  assert.doesNotMatch(html,/Demo Example|90%|80%/);
});
test('Historical/absent attribution shows disabled unavailable control',()=>{
  const html=render(DiagnosticReport,{patient:legacy,onBack(){}});assert.match(html,/disabled=""[^>]*>Grad-CAM unavailable/);
});
test('Historical latest review without a history list shows Unknown rather than an invented count',()=>{
  const row=JSON.parse(JSON.stringify(legacyRow));delete row.multi_model_result.review_history;
  const record=api.mapApiPatientToPatientRecord(row);
  assert.match(render(ReviewSummary,{analysis:record.analysis}),/Unknown \(history not recorded\)/);
});
test('Unsupported stored contract remains readable as historical with a warning',()=>{
  const row=JSON.parse(JSON.stringify(input.before_review));row.multi_model_result.contract_version='unsupported';
  const result=api.mapApiPatientToPatientRecord(row);assert.equal(result.analysis.recordKind,'historical');assert.equal(result.analysis.multiModel,undefined);
  assert.ok(result.analysis.storageWarnings.some(w=>/Unsupported/.test(w)));
});
test('Fresh unsupported inference still fails closed',()=>{
  assert.throws(()=>api.mapPredictResponseToAnalysis({...input.actual_response,contract_version:'unsupported'}),/Unsupported/);
});
test('Missing/error/not-run UI fixtures never become negative predictions',()=>{
  for(const status of ['missing','error','not_run']){
    const data=JSON.parse(JSON.stringify(input.actual_response));Object.assign(data.models.UCLM,{status,prediction:null,score:null,confidence:null,predicted_class_index:null,logits:null,probabilities:{}});
    const html=render(ResearchModelResults,{result:api.mapPredictResponseToAnalysis(data).multiModel});assert.match(html,/Prediction unavailable/);assert.match(html,/Model score: Unavailable/);
  }
});
test('Dashboard date counts use actual record dates; review/historical counts remain separate',()=>{
  const a={...after,date:'2026-10-25T12:00:00Z'},b={...legacy,date:'2026-10-24T12:00:00Z'};
  const c=api.summarizeRecords([a,b],new Date('2026-10-25T12:00:00Z'));
  assert.equal(c.scansToday,1);assert.equal(c.historical,1);assert.equal(c.reviewed,2);assert.equal(c.latestAvailable,4);
});
test('No saved records means no invented active-model count',()=>{
  assert.equal(api.summarizeRecords([]).latestAvailable,null);
  const html=render(Dashboard,{onUpload(){},recentCases:[],patientDatabase:[],userName:'Synthetic reviewer'});assert.match(html,/Not run/);assert.doesNotMatch(html,/Models Active|DICOM/);
});
test('Review form starts without assessment or invented confidence and requires identity/note',()=>{
  const html=render(FeedbackModal,{caseId:'SYNTHETIC',originalPrediction:api.ABSTENTION,originalConfidence:'Unavailable',onClose(){}});
  assert.match(html,/Reviewer ID \(unverified\)/);assert.match(html,/required=""/);assert.match(html,/Mark reviewed without an assessment/);assert.doesNotMatch(html,/value="0\.9"|90%/);
});
test('Independent assessment form never defaults an abstention to Normal',()=>{
  const html=render(CorrectionMode,{currentPrediction:api.ABSTENTION,onSubmit(){},onCancel(){}});assert.match(html,/Select assessment/);assert.match(html,/No AI label, score, heatmap or measurement is overwritten/);
  assert.doesNotMatch(html,/<option value="Normal" selected/);
});
test('Statistics loading state is explicit rather than fabricated zero counts',()=>{
  const html=render(FeedbackStats,{});assert.match(html,/Loading research workflow statistics/);assert.doesNotMatch(html,/Model Accuracy|0\.0%/);
});
test('Print style hides controls and retains research limitations (not a native print test)',()=>{
  const html=render(DiagnosticReport,{patient:after,onBack(){}});assert.match(html,/@media print/);assert.match(html,/research-report-actions/);assert.match(html,/No model or ensemble has established clinical deployment readiness/);
});

(async()=>{
  const transport=load('src/lib/api.ts');
  async function asyncTest(name,fn){try{await fn();checks.push({name,status:'PASSED'});}catch(e){checks.push({name,status:'FAILED',error:e.stack});}}
  await asyncTest('Acknowledged review is not submitted twice when reload fails then retries',async()=>{
    const state={saved:false};let posts=0;let reloads=0;
    const submit=async()=>{posts++;};const reload=async()=>{reloads++;if(reloads===1)throw new Error('Synthetic reload failure');};
    await assert.rejects(transport.commitReviewOnce(state,submit,reload),/Synthetic reload/);
    assert.equal(state.saved,true);await transport.commitReviewOnce(state,submit,reload);
    assert.equal(posts,1);assert.equal(reloads,2);
  });
  await asyncTest('Rejected review submission does not claim saved or reload records',async()=>{
    const state={saved:false};let reloads=0;
    await assert.rejects(transport.commitReviewOnce(state,async()=>{throw new Error('Synthetic rejection');},async()=>{reloads++;}),/Synthetic rejection/);
    assert.equal(state.saved,false);assert.equal(reloads,0);
  });
  const failed=checks.filter(c=>c.status==='FAILED');
  const report={status:failed.length?'FAILED':'PASSED',tests_run:checks.length,passed:checks.length-failed.length,checks,
    inputs:'Actual Stage 2 API/SQL save-reload payloads; explicitly labeled malformed/availability/transport fixtures only for error handling.',
    interactive_browser:false,network_requests:false,database_accessed:false,native_print_executed:false};
  fs.writeFileSync(path.join(out,'frontend_workflow_test_results.json'),JSON.stringify(report,null,2));
  console.log(JSON.stringify(report,null,2));if(failed.length)process.exitCode=1;
})().catch(e=>{console.error(e);process.exitCode=1;});
