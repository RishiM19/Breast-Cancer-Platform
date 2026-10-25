// Loopback-only, no automatic browser window, no report-file reload noise.
process.env.VITE_API_BASE_URL='http://127.0.0.1:8765';
import('vite').then(async ({createServer})=>{
  const server=await createServer({server:{host:'127.0.0.1',port:5177,strictPort:true,open:false,
    watch:{ignored:['**/data/**','**/*.py','**/*.cjs','**/build/**']}}});
  await server.listen();console.log('Stage 3 research test UI: http://127.0.0.1:5177');
});
