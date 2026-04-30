// MetricsPanel.jsx — KPI metrics cards
const metricsData = [
  { label: '監控路段', value: '15', unit: '條', delta: null, color: '#60a5fa', bar: 1 },
  { label: '平均車速', value: '34.6', unit: 'km/h', delta: { dir: 'down', val: '-12.3', note: '較上輪' }, color: '#f59e0b', bar: 0.58 },
  { label: '總車流量', value: '1,240', unit: '輛', delta: { dir: 'up', val: '+87', note: '較上輪' }, color: '#ef4444', bar: 0.74 },
  { label: '壅塞路段', value: '3', unit: '條', delta: { dir: 'up', val: '+1', note: '較上輪' }, color: '#ef4444', bar: 0.20 },
  { label: '暢通路段', value: '9', unit: '條', delta: { dir: 'down', val: '-2', note: '較上輪' }, color: '#22c55e', bar: 0.60 },
  { label: '號誌優化', value: '6', unit: '組', delta: null, color: '#3b82f6', bar: 0.5 },
];

function MetricsPanel() {
  return (
    <div style={metricsStyles.grid}>
      {metricsData.map((m, i) => (
        <div key={i} style={metricsStyles.card}>
          <div style={metricsStyles.label}>{m.label}</div>
          <div style={metricsStyles.valueRow}>
            <span style={{...metricsStyles.value, color: m.bar < 0.4 ? '#e2e8f0' : m.bar > 0.65 ? m.color : '#e2e8f0'}}>{m.value}</span>
            <span style={metricsStyles.unit}>{m.unit}</span>
          </div>
          {m.delta && (
            <div style={{...metricsStyles.delta, color: m.delta.dir === 'up' ? '#f87171' : '#4ade80'}}>
              {m.delta.dir === 'up' ? '▲' : '▼'} {m.delta.val} {m.delta.note}
            </div>
          )}
          <div style={metricsStyles.barTrack}>
            <div style={{...metricsStyles.barFill, width: `${m.bar*100}%`, background: m.color}}></div>
          </div>
        </div>
      ))}
    </div>
  );
}

const metricsStyles = {
  grid: { display: 'grid', gridTemplateColumns: 'repeat(6, 1fr)', gap: 10, padding: '16px 20px 0' },
  card: { background: '#1a2236', border: '1px solid #2a3555', borderRadius: 8, padding: '12px 14px' },
  label: { fontFamily: "'Noto Sans TC', sans-serif", fontSize: 11, fontWeight: 500, letterSpacing: '0.05em', textTransform: 'uppercase', color: '#475569', marginBottom: 5 },
  valueRow: { display: 'flex', alignItems: 'baseline', gap: 4 },
  value: { fontFamily: "'Space Grotesk', sans-serif", fontSize: 26, fontWeight: 700, color: '#e2e8f0', lineHeight: 1 },
  unit: { fontFamily: "'Noto Sans TC', sans-serif", fontSize: 12, color: '#94a3b8' },
  delta: { fontSize: 11, marginTop: 4, fontFamily: "'Noto Sans TC', sans-serif" },
  barTrack: { height: 3, borderRadius: 2, background: '#1f2a42', marginTop: 8, overflow: 'hidden' },
  barFill: { height: '100%', borderRadius: 2 },
};

Object.assign(window, { MetricsPanel });
