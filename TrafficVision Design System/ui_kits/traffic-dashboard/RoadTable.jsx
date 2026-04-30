// RoadTable.jsx — Road segment data table
const roadData = [
  { name: '建國南路  忠孝–仁愛', id: 'ZJHKR40', speed: 34.6, vol: 61, occ: 2.1, moe: 1 },
  { name: '建國北路  長安–忠孝', id: 'ZK7KP40', speed: 9.8, vol: 30, occ: 47.7, moe: 2 },
  { name: '市民大道  建國–金山', id: 'ZK9KC60', speed: 42.8, vol: 123, occ: 9.0, moe: 0 },
  { name: '八德路    建國–新生', id: 'ZK5JW60', speed: 30.5, vol: 33, occ: 3.1, moe: 1 },
  { name: '忠孝東路  復興–建國', id: 'ZJHKR20', speed: 47.5, vol: 154, occ: 4.3, moe: 1 },
  { name: '市民大道  金山–建國', id: 'ZKJJP20', speed: 42.1, vol: 202, occ: 5.6, moe: 1 },
  { name: '忠孝東路  金山–中山', id: 'ZJSJD40', speed: 20.5, vol: 83, occ: 12.6, moe: 2 },
  { name: '建國南路  仁愛–信義', id: 'ZINKW40', speed: 41.0, vol: 52, occ: 1.4, moe: 0 },
];

const MOEConfig = {
  0: { label: '暢通', color: '#4ade80', bg: 'rgba(34,197,94,0.10)', dot: '#22c55e' },
  1: { label: '普通', color: '#fbbf24', bg: 'rgba(245,158,11,0.10)', dot: '#f59e0b' },
  2: { label: '壅塞', color: '#f87171', bg: 'rgba(239,68,68,0.10)', dot: '#ef4444' },
};

function RoadTable({ onSelectRoad, selectedId }) {
  return (
    <div style={tableStyles.wrap}>
      <div style={tableStyles.header}>
        <div style={tableStyles.headerTitle}>路段監控列表</div>
        <div style={tableStyles.headerRight}>
          <div style={tableStyles.searchBox}>
            <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="#475569" strokeWidth="2"><circle cx="11" cy="11" r="8"/><path d="m21 21-4.35-4.35"/></svg>
            <span style={{fontFamily:"'Noto Sans TC'",fontSize:12,color:'#475569'}}>搜尋路段...</span>
          </div>
        </div>
      </div>
      <table style={tableStyles.table}>
        <thead>
          <tr>
            {['路段名稱', '平均車速', '車流量', '佔有率', '狀態'].map(h => (
              <th key={h} style={tableStyles.th}>{h}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {roadData.map(row => {
            const moe = MOEConfig[row.moe];
            const isSelected = selectedId === row.id;
            return (
              <tr key={row.id} style={{...tableStyles.tr, ...(isSelected ? tableStyles.trSelected : {})}} onClick={() => onSelectRoad && onSelectRoad(row)}>
                <td style={tableStyles.td}>
                  <div style={tableStyles.roadName}>{row.name}</div>
                  <div style={tableStyles.roadId}>{row.id}</div>
                </td>
                <td style={tableStyles.td}>
                  <span style={{fontFamily:"'Space Grotesk',sans-serif",fontWeight:600,fontSize:14,color: row.moe===2?'#f87171':row.moe===0?'#4ade80':'#e2e8f0'}}>{row.speed}</span>
                  <span style={{fontSize:12,color:'#94a3b8',marginLeft:3}}>km/h</span>
                </td>
                <td style={tableStyles.td}><span style={{fontFamily:"'Space Grotesk',sans-serif",fontSize:13,color:'#e2e8f0'}}>{row.vol}</span><span style={{fontSize:11,color:'#94a3b8',marginLeft:3}}>輛</span></td>
                <td style={tableStyles.td}><span style={{fontFamily:"'JetBrains Mono',monospace",fontSize:12,color:'#94a3b8'}}>{row.occ}%</span></td>
                <td style={tableStyles.td}>
                  <span style={{display:'inline-flex',alignItems:'center',gap:5,background:moe.bg,color:moe.color,borderRadius:4,padding:'2px 8px',fontSize:12,fontWeight:500,fontFamily:"'Noto Sans TC',sans-serif"}}>
                    <span style={{width:6,height:6,borderRadius:'50%',background:moe.dot,flexShrink:0}}></span>
                    {moe.label}
                  </span>
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

const tableStyles = {
  wrap: { background: '#1a2236', border: '1px solid #2a3555', borderRadius: 8, overflow: 'hidden', flex: 1 },
  header: { display: 'flex', alignItems: 'center', justifyContent: 'space-between', padding: '12px 16px', borderBottom: '1px solid #2a3555' },
  headerTitle: { fontFamily:"'Space Grotesk',sans-serif", fontSize: 14, fontWeight: 600, color: '#e2e8f0' },
  headerRight: { display: 'flex', gap: 8 },
  searchBox: { display: 'flex', alignItems: 'center', gap: 6, background: '#111827', border: '1px solid #2a3555', borderRadius: 6, padding: '5px 10px' },
  table: { width: '100%', borderCollapse: 'collapse' },
  th: { fontFamily:"'Noto Sans TC',sans-serif", fontSize: 11, fontWeight: 500, letterSpacing: '0.05em', textTransform: 'uppercase', color: '#475569', padding: '8px 14px', textAlign: 'left', borderBottom: '1px solid #2a3555', background: '#111827' },
  tr: { cursor: 'pointer', transition: 'background 120ms', borderBottom: '1px solid rgba(42,53,85,0.5)' },
  trSelected: { background: 'rgba(59,130,246,0.08)' },
  td: { padding: '9px 14px' },
  roadName: { fontFamily:"'Noto Sans TC',sans-serif", fontSize: 13, fontWeight: 500, color: '#e2e8f0' },
  roadId: { fontFamily:"'JetBrains Mono',monospace", fontSize: 10, color: '#475569', marginTop: 2 },
};

Object.assign(window, { RoadTable, roadData, MOEConfig });
