// Test-process-only output redirection. Original inputs and assertions are untouched.
const fs=require('node:fs'),path=require('node:path');
const root=path.resolve(__dirname,'..'),out=path.join(root,'data/processed/busi_runtime_diagnosis_20261026');
const roots=['lumina_integration_stage2','lumina_integration_stage3','lumina_ui_verdict_upgrade'].map(name=>({root:path.join(root,'data/processed',name),name}));
const write=fs.writeFileSync.bind(fs),mkdir=fs.mkdirSync.bind(fs);
function redirected(value){
 if(typeof value!=='string' && !Buffer.isBuffer(value))return value;
 const resolved=path.resolve(String(value));
 for(const entry of roots)if(resolved===entry.root||resolved.startsWith(entry.root+path.sep))return path.join(out,'regressions',entry.name,path.relative(entry.root,resolved));
 if(resolved===out||resolved.startsWith(out+path.sep))return resolved;
 // Existing toolchain temp files retain their usual semantics; this redirect is
 // confined to the original test report roots and never rewrites input reads.
 return value;
}
fs.mkdirSync=(p,options)=>mkdir(redirected(p),options);
fs.writeFileSync=(p,data,options)=>{const dest=redirected(p);if(typeof dest==='string'&&dest.startsWith(out+path.sep))mkdir(path.dirname(dest),{recursive:true});return write(dest,data,options);};
