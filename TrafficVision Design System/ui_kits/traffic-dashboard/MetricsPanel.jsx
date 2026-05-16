// MetricsPanel.jsx — TrafficVision v2 · KPI Cards

/* ── v2 tokens ─────────────────────────────────── */
const _MP = {
  ff:   "'Inter', system-ui, sans-serif",
  body: "'Noto Sans TC', 'PingFang TC', sans-serif",
  mono: "'JetBrains Mono', monospace",
  bgCard:     '#0f1629',
  bgElev:     '#162036',
  borderMed:  'rgba(99,130,200,0.25)',
  shadowCard: '0 0 0 1px rgba(99,130,200,0.25), 0 1px 3px rgba(0,0,0,0.40)',
  fg:  '#f1f5f9',
  fg2: '#94a3b8',
  fg3: '#4e6080',
  blue400:  '#60a5fa',
  blue500:  '#3b82f6',
  green400: '#4ade80',
  green500: '#22c55e',
  amber400: '#fbbf24',
  amber500: '#f59e0b',
  red400:   '#f87171',
  red500:   '#ef4444',
  violet400:'#a78bfa',
  violet500:'#8b5cf6',
};

const metricsData = [
  { label: '監控路段', value: '15',    unit: '條',   delta: null,                                              accentColor: _MP.blue500,   barW: 1.0  },
  { label: '平均車速', value: '34.6',  unit: 'km/h', delta: { dir: 'down', val: '12.3', note: '較上輪' },      accentColor: _MP.amber500,  barW: 0.58 },
  { label: '總車流量', value: '1,240', unit: '輛',   delta: { dir: 'up',   val: '87',   note: '較上輪' },      accentColor: _MP.red500,    barW: 0.74 },
  { label: '壅塞路段', value: '3',     unit: '條',   delta: { dir: 'up',   val: '1',    note: '較上輪' },      accentColor: _MP.red500,    barW: 0.20 },
  { label: '暢通路段', value: '9',     unit: '條',   delta: { dir: 'down', val: '2',    note: '較上輪' },      accentColor: _MP.green500,  barW: 0.60 },
  { label: '號誌優化', value: '6',     unit: '組',   delta: null,                                              accentColor: _MP.blue500,   barW: 0.50 },
];

function MetricCard({ label, value, unit, delta, accentColor, barW }) {
  const M = _MP;
  const isNegative = delta?.dir === 'up';   // "up" = 負面 (壅塞多/量多)
  const deltaColor = isNegative ? M.red400 : M.green400;
  const deltaArrow = isNegative ? '▲' : '▼';

  return (
    <div style={{
      background: M.bgCard,
      border: `1px solid ${M.borderMed}`,
      borderRadius: 12,
      padding: '14px 16px',
      boxShadow: M.shadowCard,
      display: 'flex', flexDirection: 'column', gap: 0,
    }}>
      {/* Label */}
      <div style={{
        fontFamily: M.body, fontSize: 11, fontWeight: 500,
        letterSpacing: '0.06em', textTransform: 'uppercase',
        color: M.fg3, marginBottom: 8,
      }}>{label}</div>

      {/* Value row */}
      <div style={{ display: 'flex', alignItems: 'baseline', gap: 5, marginBottom: delta ? 4 : 0 }}>
        <span style={{
          fontFamily: M.ff, fontSize: 28, fontWeight: 700,
          lineHeight: 1, letterSpacing: '-0.03em',
          fontVariantNumeric: 'tabular-nums',
          color: M.fg,
        }}>{value}</span>
        <span style={{ fontFamily: M.body, fontSize: 12, color: M.fg2 }}>{unit}</span>
      </div>

      {/* Delta */}
      {delta && (
        <div style={{ fontFamily: M.body, fontSize: 11, color: deltaColor, marginBottom: 8 }}>
          {deltaArrow} {delta.val} <span style={{ color: M.fg3 }}>{delta.note}</span>
        </div>
      )}
      {!delta && <div style={{ height: 19 }}/>}

      {/* Progress bar */}
      <div style={{ height: 3, borderRadius: 2, background: M.bgElev, overflow: 'hidden' }}>
        <div style={{
          height: '100%', borderRadius: 2,
          width: `${barW * 100}%`,
          background: accentColor,
          transition: 'width 600ms cubic-bezier(0.2,0,0,1)',
        }}/>
      </div>
    </div>
  );
}

function MetricsPanel() {
  return (
    <div style={{
      display: 'grid',
      gridTemplateColumns: 'repeat(6, 1fr)',
      gap: 10, padding: '14px 20px 0',
    }}>
      {metricsData.map((m, i) => <MetricCard key={i} {...m}/>)}
    </div>
  );
}

Object.assign(window, { MetricsPanel });
