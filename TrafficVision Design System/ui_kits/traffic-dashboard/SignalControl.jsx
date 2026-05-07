// SignalControl.jsx — Traffic signal timing panel
const signalData = [
  { id: 'IK7KP', name: '建國南一段/市民三段', cycle: 90, phases: [{dur:42,color:'#22c55e',label:'直行'},{dur:28,color:'#ef4444',label:'左轉'},{dur:20,color:'#f59e0b',label:'黃燈'}], status: 'optimized' },
  { id: 'IK9KC', name: '八德路二段/市民三段', cycle: 80, phases: [{dur:35,color:'#22c55e',label:'直行'},{dur:30,color:'#ef4444',label:'左轉'},{dur:15,color:'#f59e0b',label:'黃燈'}], status: 'normal' },
  { id: 'IKGKP', name: '八德路二段/建國北', cycle: 100, phases: [{dur:50,color:'#22c55e',label:'直行'},{dur:35,color:'#ef4444',label:'左轉'},{dur:15,color:'#f59e0b',label:'黃燈'}], status: 'congested' },
  { id: 'IJHKR', name: '忠孝東三段/建國南', cycle: 75, phases: [{dur:38,color:'#22c55e',label:'直行'},{dur:25,color:'#ef4444',label:'左轉'},{dur:12,color:'#f59e0b',label:'黃燈'}], status: 'optimized' },
];

const statusConfig = {
  optimized: { label: '已優化', color: '#4ade80', bg: 'rgba(34,197,94,0.10)' },
  normal: { label: '正常', color: '#60a5fa', bg: 'rgba(59,130,246,0.10)' },
  congested: { label: '壅塞中', color: '#f87171', bg: 'rgba(239,68,68,0.10)' },
};

function SignalControl() {
  const [selected, setSelected] = React.useState('IK7KP');
  const sel = signalData.find(s => s.id === selected);

  return (
    <div style={signalStyles.wrap}>
      <div style={signalStyles.header}>
        <span style={signalStyles.title}>號誌控制</span>
        <span style={{...signalStyles.badge, color: '#4ade80', background: 'rgba(34,197,94,0.1)'}}>6 組號誌優化中</span>
      </div>
      <div style={signalStyles.list}>
        {signalData.map(s => {
          const cfg = statusConfig[s.status];
          return (
            <div key={s.id} style={{...signalStyles.item, ...(selected===s.id ? signalStyles.itemActive : {})}} onClick={() => setSelected(s.id)}>
              <div style={signalStyles.itemLeft}>
                <div style={signalStyles.itemId}>{s.id}</div>
                <div style={signalStyles.itemName}>{s.name}</div>
              </div>
              <div style={{display:'flex', alignItems:'center', gap: 8}}>
                <span style={signalStyles.cycle}>{s.cycle}s</span>
                <span style={{...signalStyles.badge, color: cfg.color, background: cfg.bg}}>{cfg.label}</span>
              </div>
            </div>
          );
        })}
      </div>
      {sel && (
        <div style={signalStyles.detail}>
          <div style={signalStyles.detailTitle}>{sel.id} — 週期 {sel.cycle}s</div>
          <div style={signalStyles.phaseRow}>
            {sel.phases.map((p, i) => (
              <div key={i} style={{...signalStyles.phaseBar, width: `${(p.dur/sel.cycle)*100}%`, background: p.color}}>
                <span style={signalStyles.phaseLabel}>{p.dur}s</span>
              </div>
            ))}
          </div>
          <div style={signalStyles.phaseLegend}>
            {sel.phases.map((p, i) => (
              <div key={i} style={signalStyles.phaseLegendItem}>
                <span style={{width:8,height:8,borderRadius:'50%',background:p.color,display:'inline-block'}}></span>
                <span>{p.label} {p.dur}s</span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

const signalStyles = {
  wrap: { background: '#1a2236', border: '1px solid #2a3555', borderRadius: 8, overflow: 'hidden', display: 'flex', flexDirection: 'column' },
  header: { display: 'flex', alignItems: 'center', justifyContent: 'space-between', padding: '10px 14px', borderBottom: '1px solid #2a3555' },
  title: { fontFamily:"'Space Grotesk',sans-serif", fontSize: 13, fontWeight: 600, color: '#e2e8f0' },
  badge: { fontFamily:"'Noto Sans TC',sans-serif", fontSize: 11, fontWeight: 500, borderRadius: 4, padding: '2px 8px' },
  list: { padding: 8 },
  item: { display: 'flex', alignItems: 'center', justifyContent: 'space-between', padding: '8px 10px', borderRadius: 6, cursor: 'pointer', marginBottom: 2, transition: 'background 120ms' },
  itemActive: { background: 'rgba(59,130,246,0.10)' },
  itemLeft: { flex: 1 },
  itemId: { fontFamily:"'JetBrains Mono',monospace", fontSize: 11, color: '#60a5fa', marginBottom: 2 },
  itemName: { fontFamily:"'Noto Sans TC',sans-serif", fontSize: 12, color: '#94a3b8' },
  cycle: { fontFamily:"'Space Grotesk',sans-serif", fontSize: 12, color: '#475569' },
  detail: { borderTop: '1px solid #2a3555', padding: '10px 14px' },
  detailTitle: { fontFamily:"'JetBrains Mono',monospace", fontSize: 11, color: '#94a3b8', marginBottom: 8 },
  phaseRow: { display: 'flex', borderRadius: 4, overflow: 'hidden', height: 22 },
  phaseBar: { display: 'flex', alignItems: 'center', justifyContent: 'center' },
  phaseLabel: { fontFamily:"'Space Grotesk',sans-serif", fontSize: 10, color: 'rgba(255,255,255,0.8)', fontWeight: 600 },
  phaseLegend: { display: 'flex', gap: 12, marginTop: 7 },
  phaseLegendItem: { display: 'flex', alignItems: 'center', gap: 4, fontFamily:"'Noto Sans TC',sans-serif", fontSize: 11, color: '#94a3b8' },
};

Object.assign(window, { SignalControl });
