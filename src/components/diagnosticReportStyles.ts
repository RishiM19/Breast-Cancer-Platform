/** Report-only styles: every screen selector is scoped to the report root. */
export const diagnosticReportStyles = `
.lumina-report { padding:32px 24px 64px; }
.lumina-report .lr-workspace { max-width:1060px;margin:0 auto; }
.lumina-report .lr-actions { display:flex;align-items:center;justify-content:space-between;gap:12px;flex-wrap:wrap;margin:0 0 22px; }
.lumina-report .lr-action-group { display:flex;gap:10px;flex-wrap:wrap; }
.lumina-report .lr-print-help { flex-basis:100%;font-size:12px;color:var(--l-muted);text-align:right; }
.lumina-report .lr-document { background:var(--l-surface);border:1px solid var(--l-line);border-top:5px solid #175cd3;border-radius:3px;box-shadow:0 12px 45px #12243d0b;padding:42px 48px 24px;overflow-wrap:anywhere; }
.lumina-report .lr-letterhead { display:flex;justify-content:space-between;gap:24px;align-items:center;margin-bottom:32px; }
.lumina-report .l-brand small { display:block; }
.lumina-report .lr-document-kind { text-align:right;display:grid;gap:3px; }
.lumina-report .lr-document-kind span { font-size:10px;letter-spacing:2px;color:var(--l-blue);font-weight:750; }
.lumina-report .lr-document-kind strong { font-size:13px;font-weight:600; }
.lumina-report .lr-document-kind small { font-size:11px;color:var(--l-muted); }
.lumina-report h1 { font-size:32px;letter-spacing:-.7px;line-height:1.2; }
.lumina-report .lr-subtitle { color:var(--l-muted);font-size:13px;margin-top:8px; }
.lumina-report .lr-patient-grid { display:grid;grid-template-columns:1.6fr .8fr 1.2fr 1.5fr;column-gap:22px;row-gap:20px;margin-top:26px;padding:22px 0;border-top:1px solid var(--l-line);border-bottom:1px solid var(--l-line); }
.lumina-report dt { font-size:11px;color:var(--l-muted);font-weight:500; }
.lumina-report dd { margin:5px 0 0;font-size:13px;line-height:1.55; }
.lumina-report .lr-patient-name dd { font-size:17px;font-weight:650; }
.lumina-report .lr-muted { color:var(--l-muted); }
.lumina-report .lr-research-note { margin:16px 0 0;font-size:11px;color:var(--l-muted);max-width:800px; }
.lumina-report .lr-section { padding:28px 0;border-bottom:1px solid var(--l-line); }
.lumina-report .lr-section-heading { display:flex;gap:12px;align-items:baseline;margin-bottom:20px; }
.lumina-report .lr-section-heading > span { font-size:11px;letter-spacing:.5px;font-weight:700;color:var(--l-blue); }
.lumina-report .lr-section-heading h2 { font-size:18px;letter-spacing:-.2px;font-weight:650; }
.lumina-report .lr-section-heading p { font-size:12px;color:var(--l-muted);margin-top:4px; }
.lumina-report .lr-assessment-grid { display:grid;grid-template-columns:1.35fr 1fr;gap:24px;align-items:start; }
.lumina-report .lr-assessment .l-card { box-shadow:none;border-radius:3px;padding:20px; }
.lumina-report .lr-assessment .l-verdict { background:var(--l-blue-soft);border:0;border-left:3px solid #175cd3; }
.lumina-report .l-verdict-class { font-size:37px;letter-spacing:-1px; }
.lumina-report .l-verdict-meta { gap:12px;margin:14px 0;padding:12px 0; }
.lumina-report .l-verdict-meta strong { font-size:15px; }
.lumina-report .l-verdict-meta small { font-size:11px; }
.lumina-report .lr-assessment .l-muted,.lumina-report .l-disagreement,.lumina-report .l-limitations,.lumina-report .l-warning,.lumina-report .l-notice { font-size:12px; }
.lumina-report .lr-assessment .l-section-head { align-items:start;margin-bottom:12px; }
.lumina-report .lr-assessment .l-eyebrow { font-size:10px;letter-spacing:1px; }
.lumina-report .l-birads { border:0!important;border-bottom:1px solid var(--l-line)!important;padding:0 0 16px!important;gap:12px; }
.lumina-report .l-birads h2 { font-size:17px; }
.lumina-report .l-birads-value { font-size:38px; }
.lumina-report .l-birads-label { font-size:11px; }
.lumina-report .lr-birads-column { padding-top:8px; }
.lumina-report .lr-birads-meaning { display:grid;gap:5px;margin:15px 0;font-size:13px; }
.lumina-report .lr-birads-meaning span { font-size:12px;color:var(--l-muted); }
.lumina-report .lr-category-scores { display:grid;grid-template-columns:repeat(4,1fr);gap:8px;margin:16px 0; }
.lumina-report .lr-category-scores dd { font-weight:650; }
.lumina-report .lr-category-scores dt { font-size:10px; }
.lumina-report .lr-impression { margin-top:22px;border-left:3px solid var(--l-line);padding-left:16px; }
.lumina-report h3 { font-size:13px;font-weight:650; }
.lumina-report .lr-impression p { font-size:13px;margin-top:8px; }
.lumina-report .lr-small { font-size:11px;color:var(--l-muted);line-height:1.6;margin-top:12px; }
.lumina-report .lr-facts { display:grid;grid-template-columns:1fr 1fr;column-gap:32px; }
.lumina-report .lr-facts > div { padding:12px 0;border-bottom:1px solid var(--l-line);min-width:0; }
.lumina-report .lr-facts > div:nth-last-child(-n+2) { border-bottom:0; }
.lumina-report .lr-model-table { border-collapse:collapse;width:100%;text-align:left;table-layout:fixed;font-size:13px; }
.lumina-report .lr-model-table caption { text-align:left;color:var(--l-muted);font-size:12px;margin-bottom:12px; }
.lumina-report .lr-model-table th,.lumina-report .lr-model-table td { padding:15px 12px;border-bottom:1px solid var(--l-line);vertical-align:top; }
.lumina-report .lr-model-table thead th { background:var(--l-bg);font-size:10px;text-transform:uppercase;letter-spacing:.6px;font-weight:650; }
.lumina-report .lr-model-table th:nth-child(1) { width:23%; }
.lumina-report .lr-model-table th:nth-child(2) { width:22%; }
.lumina-report .lr-model-table th:nth-child(3) { width:20%; }
.lumina-report .lr-model-table th:nth-child(4) { width:35%; }
.lumina-report .lr-model-table tbody th { font-weight:650; }
.lumina-report .lr-model-table small { display:block;font-weight:400;color:var(--l-muted);font-size:10px;margin-top:3px; }
.lumina-report .lr-distribution { display:grid;gap:3px; }
.lumina-report .lr-distribution > div { display:flex;justify-content:space-between;gap:8px; }
.lumina-report .lr-distribution dt,.lumina-report .lr-distribution dd { margin:0;font-size:11px; }
.lumina-report .lr-table-notes { display:grid;gap:6px;margin-top:14px;font-size:11px;color:var(--l-muted); }
.lumina-report .lr-evidence-grid { display:grid;grid-template-columns:1.2fr 1fr;gap:24px;margin-bottom:22px;align-items:start; }
.lumina-report .lr-evidence { margin:0;min-width:0; }
.lumina-report .lr-image-frame { display:grid;place-items:center;background:#101925;border-radius:3px;min-height:180px;color:#dbe5f3; }
.lumina-report .lr-image-frame img { max-height:310px;width:100%;object-fit:contain; }
.lumina-report .lr-image-frame p { padding:30px;font-size:12px; }
.lumina-report .lr-evidence figcaption { margin:9px 0 12px;font-size:11px;color:var(--l-muted); }
.lumina-report .lr-evidence-note { font-size:12px;color:var(--l-muted); }
.lumina-report .lr-evidence-note h3 { color:var(--l-ink); }
.lumina-report .lr-evidence-note dl { margin:10px 0 16px; }
.lumina-report .lr-evidence-note dl > div { display:flex;justify-content:space-between;gap:15px;padding:7px 0;border-bottom:1px solid var(--l-line); }
.lumina-report .lr-evidence-note dd { margin:0;font-size:11px;text-align:right; }
.lumina-report .lr-evidence-note p { margin-top:10px; }
.lumina-report .lr-print-image,.lumina-report .lr-print-method { display:none; }
.lumina-report .lr-section > .l-card { box-shadow:none;border:0;border-radius:0;padding:0; }
.lumina-report .lr-section > .l-card h2 { font-size:15px; }
.lumina-report .l-segmentation-grid img { max-height:200px;object-fit:contain; }
.lumina-report .l-segmentation-grid figcaption { font-size:11px;color:var(--l-muted);margin-top:7px; }
.lumina-report .lr-narrative { display:grid;gap:17px; }
.lumina-report .lr-narrative dt { font-size:12px;font-weight:650;color:var(--l-ink); }
.lumina-report .lr-narrative dd { color:var(--l-muted);white-space:pre-wrap; }
.lumina-report .lr-review-section .l-review h3 { color:var(--l-ink); }
.lumina-report .lr-review-section .l-review p { font-size:12px;color:var(--l-muted);overflow-wrap:anywhere;white-space:pre-wrap; }
.lumina-report .lr-checkpoints { margin:18px 0;font-size:11px;color:var(--l-muted); }
.lumina-report summary { cursor:pointer; }
.lumina-report .lr-checkpoints summary { color:var(--l-ink);font-weight:650;font-size:12px; }
.lumina-report .lr-checkpoint-list { display:grid;gap:14px;margin:16px 0; }
.lumina-report .lr-checkpoint-list > div { border-left:2px solid var(--l-line);padding-left:12px; }
.lumina-report .lr-checkpoint-list p { margin-top:4px; }
.lumina-report .lr-mono { font-family:ui-monospace,Consolas,monospace;font-size:10px;overflow-wrap:anywhere; }
.lumina-report .lr-footer { display:flex;justify-content:space-between;gap:16px;font-size:10px;color:var(--l-muted);padding-top:22px; }
.lumina-report .lr-stored-finding { background:var(--l-bg);padding:20px;display:grid;gap:8px;font-size:13px; }
.lumina-report .lr-stored-finding strong { font-size:26px; }
.lumina-report .lr-historical { margin-top:22px;padding:14px;border:1px solid var(--l-line);font-size:12px; }
@media(max-width:760px) {
  .lumina-report { padding:16px 10px 32px; }
  .lumina-report .lr-document { padding:26px 20px 20px; }
  .lumina-report .lr-letterhead { align-items:start;gap:12px;margin-bottom:24px; }
  .lumina-report .lr-document-kind { max-width:160px; }
  .lumina-report .lr-document-kind strong { font-size:11px; }
  .lumina-report .lr-document-kind small { font-size:10px; }
  .lumina-report .l-logo { width:34px;height:34px;border-radius:8px; }
  .lumina-report .l-brand { gap:8px; }
  .lumina-report .l-brand strong { font-size:20px; }
  .lumina-report .l-brand small { font-size:10px; }
  .lumina-report h1 { font-size:27px; }
  .lumina-report .lr-patient-grid { grid-template-columns:1fr 1fr;gap:18px; }
  .lumina-report .lr-assessment-grid,.lumina-report .lr-facts,.lumina-report .lr-evidence-grid { grid-template-columns:1fr; }
  .lumina-report .lr-facts > div:nth-last-child(2) { border-bottom:1px solid var(--l-line); }
  .lumina-report .lr-table-wrap { overflow-x:auto; }
  .lumina-report .lr-model-table { min-width:500px; }
  .lumina-report .lr-actions { align-items:stretch; }
  .lumina-report .lr-actions > .l-button { flex-basis:100%; }
  .lumina-report .lr-action-group { flex:1; }
  .lumina-report .lr-action-group button { flex:1; }
  .lumina-report .lr-print-help { text-align:left; }
  .lumina-report .lr-footer { flex-wrap:wrap; }
}
@media print {
  @page { size:A4;margin:14mm; }
  .lumina-report { --l-bg:#f5f7fb;--l-surface:#fff;--l-ink:#15253d;--l-muted:#4c5b70;--l-line:#d0d8e2;--l-blue:#175cd3;--l-blue-soft:#eef4ff;padding:0!important;background:#fff!important;color:#15253d!important;min-height:0;font-size:10px;print-color-adjust:exact;-webkit-print-color-adjust:exact; }
  .lumina-report .research-report-actions,.lumina-report .lr-screen-image { display:none!important; }
  .lumina-report .lr-print-image { display:block; }
  .lumina-report .lr-assessment .l-limitations { display:none; }
  .lumina-report .lr-print-method { display:block;margin-top:18px;font-size:10px; }
  .lumina-report .lr-print-method li { margin-top:6px; }
  .lumina-report .lr-workspace { max-width:none; }
  .lumina-report .lr-document { box-shadow:none;border:0;border-top:3px solid #175cd3;padding:16px 0 0;border-radius:0; }
  .lumina-report header,.lumina-report section { margin-bottom:0; }
  .lumina-report .lr-header { break-inside:avoid; }
  .lumina-report .lr-letterhead { margin-bottom:18px; }
  .lumina-report h1 { font-size:25px; }
  .lumina-report .lr-patient-grid { grid-template-columns:1.4fr .7fr 1.2fr 1.4fr;gap:12px 16px;padding:14px 0;margin-top:18px; }
  .lumina-report .lr-patient-name dd { font-size:13px; }
  .lumina-report .lr-section { break-inside:auto;padding:18px 0; }
  .lumina-report .lr-section-heading { break-after:avoid;margin-bottom:12px; }
  .lumina-report .lr-section-heading h2 { font-size:15px; }
  .lumina-report p,.lumina-report li,.lumina-report dd { font-size:10px; }
  .lumina-report dt,.lumina-report .lr-small,.lumina-report .lr-research-note,.lumina-report .lr-section-heading p { font-size:9px; }
  .lumina-report .lr-assessment-grid { display:grid;grid-template-columns:1.25fr 1fr;gap:20px; }
  .lumina-report .lr-assessment .l-card { padding:14px;break-inside:auto; }
  .lumina-report .l-verdict-class { font-size:29px; }
  .lumina-report .l-verdict-meta strong { font-size:12px; }
  .lumina-report .l-verdict-meta small,.lumina-report .l-birads-label,.lumina-report .lr-category-scores dt { font-size:9px; }
  .lumina-report .lr-assessment .l-section-head { flex-wrap:wrap;gap:6px; }
  .lumina-report .l-birads h2 { font-size:13px; }
  .lumina-report .l-birads-value { font-size:30px; }
  .lumina-report .l-limitations { font-size:9px; }
  .lumina-report .l-limitations li { margin-top:4px; }
  .lumina-report .l-warning,.lumina-report .l-notice { padding:8px;border-radius:3px; }
  .lumina-report .lr-impression { margin-top:14px; }
  .lumina-report .lr-impression p { font-size:10px;margin-top:6px; }
  .lumina-report .lr-facts { grid-template-columns:1fr 1fr;gap:0 24px; }
  .lumina-report .lr-facts > div { padding:8px 0;break-inside:avoid; }
  .lumina-report .lr-table-wrap { overflow:visible; }
  .lumina-report .lr-model-table { min-width:0;font-size:10px; }
  .lumina-report .lr-model-table thead { display:table-header-group; }
  .lumina-report .lr-model-table tr { break-inside:avoid; }
  .lumina-report .lr-model-table td,.lumina-report .lr-model-table th { padding:9px 8px; }
  .lumina-report .lr-distribution dt,.lumina-report .lr-distribution dd { font-size:9px; }
  .lumina-report .lr-evidence-grid { grid-template-columns:1fr 1fr;gap:18px; }
  .lumina-report .lr-evidence { break-inside:avoid; }
  .lumina-report .lr-evidence-note { grid-column:1/-1; }
  .lumina-report .lr-image-frame { min-height:130px; }
  .lumina-report .lr-image-frame img { max-height:210px; }
  .lumina-report .lr-evidence figcaption { font-size:9px;margin:6px 0; }
  .lumina-report .lr-evidence-note dl { display:grid;grid-template-columns:repeat(3,1fr);gap:16px;margin:8px 0; }
  .lumina-report .lr-evidence-note dl > div { display:block; }
  .lumina-report .lr-evidence-note dd { text-align:left;font-size:9px; }
  .lumina-report .lr-section > .l-card { break-inside:auto; }
  .lumina-report .l-segmentation-grid { grid-template-columns:repeat(3,1fr)!important;break-inside:avoid; }
  .lumina-report .l-segmentation-grid img { max-height:150px; }
  .lumina-report .lr-narrative { gap:10px; }
  .lumina-report .lr-narrative > div,.lumina-report .lr-checkpoint-list > div { break-inside:avoid; }
  .lumina-report .lr-review-section .l-review p { font-size:10px; }
  .lumina-report .lr-provenance { break-before:page; }
  .lumina-report .lr-checkpoint-list { gap:12px; }
  .lumina-report .lr-mono { font-size:9px; }
  .lumina-report .lr-footer { font-size:9px;break-inside:avoid;padding-top:14px; }
}
`;
