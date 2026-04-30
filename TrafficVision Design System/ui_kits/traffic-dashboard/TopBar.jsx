// TopBar.jsx — TrafficVision top navigation bar
function TopBar({ title, lastUpdated, onRefresh }) {
  return (
    <div style={topBarStyles.bar}>
      <div style={topBarStyles.left}>
        <div style={topBarStyles.title}>{title}</div>
        <div style={topBarStyles.timestamp}>
          <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="#475569" strokeWidth="2" style={{flexShrink:0}}><circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/></svg>
          最後更新：{lastUpdated}
        </div>
      </div>
      <div style={topBarStyles.right}>
        <div style={topBarStyles.badge}>
          <span style={topBarStyles.liveDoc}></span>
          <span style={{fontFamily:"'Noto Sans TC'",fontSize:12,color:'#4ade80'}}>即時資料</span>
        </div>
        <button style={topBarStyles.btn} onClick={onRefresh}>
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><polyline points="23 4 23 10 17 10"/><path d="M20.49 15a9 9 0 1 1-.08-7.49"/></svg>
          重新整理
        </button>
        <button style={{...topBarStyles.btn, background:'#2563eb', color:'#fff', border:'none'}}>
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/></svg>
          啟動優化
        </button>
      </div>
    </div>
  );
}

const topBarStyles = {
  bar: { height: 56, background: '#111827', borderBottom: '1px solid #2a3555', display: 'flex', alignItems: 'center', justifyContent: 'space-between', padding: '0 20px', flexShrink: 0 },
  left: { display: 'flex', alignItems: 'center', gap: 14 },
  title: { fontFamily: "'Space Grotesk', sans-serif", fontSize: 16, fontWeight: 600, color: '#e2e8f0' },
  timestamp: { fontFamily: "'JetBrains Mono', monospace", fontSize: 11, color: '#475569', display: 'flex', alignItems: 'center', gap: 5 },
  right: { display: 'flex', alignItems: 'center', gap: 10 },
  badge: { display: 'flex', alignItems: 'center', gap: 6, background: 'rgba(34,197,94,0.10)', borderRadius: 6, padding: '5px 10px' },
  liveDoc: { width: 7, height: 7, borderRadius: '50%', background: '#22c55e', boxShadow: '0 0 5px rgba(34,197,94,0.7)' },
  btn: { display: 'inline-flex', alignItems: 'center', gap: 6, height: 32, padding: '0 12px', borderRadius: 6, border: '1px solid #2a3555', background: '#1f2a42', color: '#94a3b8', fontFamily: "'Noto Sans TC', sans-serif", fontSize: 12, cursor: 'pointer' },
};

Object.assign(window, { TopBar });
