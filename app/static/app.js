const fileInput=document.querySelector('#fileInput'),dropZone=document.querySelector('#dropZone'),fileCard=document.querySelector('#fileCard'),fileName=document.querySelector('#fileName'),fileMeta=document.querySelector('#fileMeta'),clearFile=document.querySelector('#clearFile'),useSample=document.querySelector('#useSample'),runButton=document.querySelector('#runButton'),statusLine=document.querySelector('#statusLine'),modelState=document.querySelector('#modelState');let selectedFile=null,modelReady=false;
const palette={x:'#78838c',y:'#81a8e7',z:'#9c8fc4'};
function drawAxes(ctx,w,h,yLabel,xLabel){ctx.clearRect(0,0,w,h);ctx.strokeStyle='#7d858b';ctx.lineWidth=2;ctx.beginPath();ctx.moveTo(55,12);ctx.lineTo(55,h-32);ctx.lineTo(w-12,h-32);ctx.stroke();ctx.fillStyle='#6e777e';ctx.font='14px sans-serif';ctx.fillText(xLabel,w/2-45,h-8);ctx.save();ctx.translate(18,h/2+30);ctx.rotate(-Math.PI/2);ctx.fillText(yLabel,0,0);ctx.restore()}
function drawEmpty(){let c=document.querySelector('#timeChart'),x=c.getContext('2d');drawAxes(x,c.width,c.height,'Acceleration','Sample index');x.fillStyle='#a1a7ac';x.fillText('Upload a CSV file to preview the signal',160,105);c=document.querySelector('#frequencyChart');x=c.getContext('2d');drawAxes(x,c.width,c.height,'Magnitude','Frequency bin (X axis)');x.fillText('FFT spectrum will appear here',205,80)}
function drawLines(series){const c=document.querySelector('#timeChart'),ctx=c.getContext('2d'),w=c.width,h=c.height;drawAxes(ctx,w,h,'Acceleration','Sample index');const all=[...series.x,...series.y,...series.z],min=Math.min(...all),max=Math.max(...all),span=max-min||1;for(const key of ['x','y','z']){const data=series[key];ctx.strokeStyle=palette[key];ctx.lineWidth=3;ctx.beginPath();data.forEach((v,i)=>{const px=62+i*(w-82)/Math.max(1,data.length-1),py=15+(max-v)*(h-55)/span;i?ctx.lineTo(px,py):ctx.moveTo(px,py)});ctx.stroke()}}
function drawBars(values){const c=document.querySelector('#frequencyChart'),ctx=c.getContext('2d'),w=c.width,h=c.height;drawAxes(ctx,w,h,'Magnitude','Frequency bin (X axis)');const max=Math.max(...values,1),barW=(w-82)/values.length;ctx.fillStyle='#8baee5';values.forEach((v,i)=>{const height=v/max*(h-58);ctx.fillRect(62+i*barW,h-34-height,Math.max(1,barW*.7),height)})}
const connectionMessage='Mất kết nối máy chủ. Hãy chạy start_web.cmd, giữ cửa sổ chạy web mở và truy cập http://127.0.0.1:8000.';
// Distinguish transport failures from CSV validation failures.
async function request(url,options){
  if(location.protocol==='file:')throw new TypeError(connectionMessage);
  let response;
  try{response=await fetch(url,options)}catch{throw new TypeError(connectionMessage)}
  const type=response.headers.get('content-type')||'';
  if(!type.includes('application/json'))throw new TypeError(connectionMessage);
  const data=await response.json();
  if(!response.ok)throw new Error(typeof data.detail==='string'?data.detail:'Máy chủ không thể xử lý yêu cầu.');
  return data;
}
function resetPrediction(){document.querySelector('#predictedLabel').textContent='—';document.querySelector('#confidenceText').textContent='—';document.querySelector('#confidenceBar').style.width='0%';document.querySelector('#probabilities').replaceChildren();}
async function checkModel(){try{const data=await request('/api/health');modelReady=data.ready;document.querySelector('#requiredSamples').textContent=data.required_samples;modelState.className='model-state '+(data.ready?'ready':'error');modelState.querySelector('p').textContent=data.message;runButton.disabled=!(selectedFile&&modelReady)}catch{modelReady=false;runButton.disabled=true;modelState.className='model-state error';modelState.querySelector('p').textContent=connectionMessage}}
let previewVersion=0;
async function preview(file){
  const version=++previewVersion;
  selectedFile=null;runButton.disabled=true;drawEmpty();resetPrediction();
  fileName.textContent=file.name;fileMeta.textContent=`${(file.size/1024).toFixed(1)} KB · đang đọc…`;fileCard.hidden=false;statusLine.textContent='Đang phân tích tín hiệu…';
  const body=new FormData();body.append('file',file);
  try{
    const data=await request('/api/preview',{method:'POST',body});
    if(version!==previewVersion)return;
    selectedFile=file;drawLines(data.time);drawBars(data.frequency);
    fileMeta.textContent=`${data.sample_count.toLocaleString('vi-VN')} mẫu · ${(file.size/1024).toFixed(1)} KB`;
    await checkModel();
    if(version!==previewVersion)return;
    statusLine.textContent=modelReady?'Tín hiệu đã sẵn sàng. Bấm Run Prediction để nhận dạng.':'Đã đọc dữ liệu. Cần model đã huấn luyện để dự đoán.';
    runButton.disabled=!modelReady;
  }catch(error){
    if(version!==previewVersion)return;
    selectedFile=null;statusLine.textContent=error.message;
    fileMeta.textContent=error instanceof TypeError?'Chưa đọc được file do mất kết nối máy chủ':'File không hợp lệ: '+error.message;
    if(error instanceof TypeError){modelReady=false;modelState.className='model-state error';modelState.querySelector('p').textContent=connectionMessage}
    runButton.disabled=true;
  }
}
function clearSelection(){previewVersion++;selectedFile=null;fileInput.value='';fileCard.hidden=true;runButton.disabled=true;statusLine.textContent='Hãy tải file dữ liệu để bắt đầu.';drawEmpty();resetPrediction()}
fileInput.addEventListener('change',()=>fileInput.files[0]&&preview(fileInput.files[0]));clearFile.addEventListener('click',clearSelection);['dragenter','dragover'].forEach(e=>dropZone.addEventListener(e,event=>{event.preventDefault();dropZone.classList.add('dragover')}));['dragleave','drop'].forEach(e=>dropZone.addEventListener(e,event=>{event.preventDefault();dropZone.classList.remove('dragover')}));dropZone.addEventListener('drop',event=>event.dataTransfer.files[0]&&preview(event.dataTransfer.files[0]));
useSample.addEventListener('click',async()=>{useSample.disabled=true;statusLine.textContent='Đang mở dữ liệu mẫu…';try{const response=await fetch('/sample-data/gesture_example.csv');if(!response.ok)throw new Error('Không tải được CSV mẫu. Hãy khởi động lại start_web.cmd.');const blob=await response.blob();await preview(new File([blob],'gesture_example.csv',{type:'text/csv'}))}catch(error){statusLine.textContent=error instanceof TypeError?connectionMessage:error.message;await checkModel()}finally{useSample.disabled=false}});
runButton.addEventListener('click',async()=>{
  if(!selectedFile||!modelReady)return;
  const file=selectedFile,version=previewVersion;
  runButton.disabled=true;runButton.textContent='Đang nhận dạng…';
  statusLine.textContent='DINOv3 đang phân tích time và frequency views…';
  const body=new FormData();body.append('file',file);
  try{
    const data=await request('/api/predict',{method:'POST',body});
    if(version!==previewVersion)return;
    document.querySelector('#predictedLabel').textContent=data.label.replaceAll('_',' ');
    const percent=data.confidence*100;
    document.querySelector('#confidenceText').textContent=`${percent.toFixed(1)}%`;
    document.querySelector('#confidenceBar').style.width=`${percent}%`;
    const container=document.querySelector('#probabilities');container.replaceChildren();
    Object.entries(data.probabilities).sort((a,b)=>b[1]-a[1]).forEach(([name,value],i)=>{
      const row=document.createElement('div');row.className='probability-row'+(i===0?' top':'');
      const label=document.createElement('span');label.className='label';label.title=name;label.textContent=name.replaceAll('_',' ');
      const bar=document.createElement('span');bar.className='bar';
      const fill=document.createElement('i');fill.style.width=`${value*100}%`;bar.append(fill);
      const number=document.createElement('b');number.textContent=`${(value*100).toFixed(1)}%`;
      row.append(label,bar,number);container.append(row);
    });
    statusLine.textContent=`Đã phân tích ${data.sample_count.toLocaleString('vi-VN')} mẫu qua ${data.windows_used} cửa sổ.`;
  }catch(error){
    if(version!==previewVersion)return;
    resetPrediction();statusLine.textContent=error.message;
    if(error instanceof TypeError)await checkModel();
  }finally{
    runButton.disabled=!(selectedFile&&modelReady);runButton.innerHTML='<span>▶</span> Run Prediction';
  }
});
drawEmpty();checkModel();
