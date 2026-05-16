// SignalControl.jsx — TrafficVision v2 · Signal Timing Panel

/* ── v2 tokens ─────────────────────────────────── */
const _SC = {
  ff:   "'Inter', system-ui, sans-serif",
  body: "'Noto Sans TC', 'PingFang TC', sans-serif",
  mono: "'JetBrains Mono', monospace",
  bgCard:    '#0f1629',
  bgSurf:    '#0a0f1e',
  bgElev:    '#162036',
  borderMed: 'rgba(99,130,200,0.25)',
  divider:   'rgba(99,130,200,0.08)',
  shadowCard:'0 0 0 1px rgba(99,130,200,0.25), 0 1px 3px rgba(0,0,0,0.40)',
  blue400:   '#60a5fa',
  blue500:   '#3b82f6',
  blueSubtle:'rgba(59,130,246,0.08)',
  blueBg:    'rgba(59,130,246,0.10)',
  blueBd:    'rgba(59,130,246,0.25)',
  green400:  '#4ade80',
  green500:  '#22c55e',
  green600:  '#16a34a',
  greenBg:   'rgba(34,197,94,0.10)',
  greenBd:   'rgba(34,197,94,0.25)',
  amber500:  '#f59e0b',
  amber600:  '#d97706',
  amberBg:   'rgba(245,158,11,0.10)',
  amberBd:   'rgba(245,158,11,0.25)',
  red400:    '#f87171',
  red500:    '#ef4444',
  red600:    '#dc2626',
  redBg:     'rgba(239,68,68,0.10)',
  redBd:     'rgba(239,68,68,0.25)',
  fg:  '#f1f5f9',
  fg2: '#94a3b8',
  fg3: '#4e6080',
};

const signalData = [
  { id: 'IK7KP', name: '建國南一段/市民三段', cycle: 90,
    phases: [{ dur:42, bg:'#16a34a', label:'直行' },{ dur:28, bg:'#dc2626', label:'左轉' },{ dur:20, bg:'#d97706', label:'黃燈' }],
    status: 'optimized' },
  { id: 'IK9KC', name: '八德路二段/市民三段', cycle: 80,
    phases: [{ dur:35, bg:'#16a34a', label:'直行' },{ dur:30, bg:'#dc2626', label:'左轉' },{ dur:15, bg:'#d97706', label:'黃燈' }],
    status: 'normal' },
  { id: 'IKGKP', name: '八德路二段/建國北', cycle: 100,
    phases: [{ dur:50, bg:'#16a34a', label:'直行' },{ dur:35, bg:'#dc2626', label:'左轉' },{ dur:15, bg:'#d97706', label:'黃燈' }],
    status: 'congested' },
  { id: 'IJHKR', name: '忠孝東三段/建國南', cycle: 75,
    phases: [{ dur:38, bg:'#16a34a', label:'直行' },{ dur:25, bg:'#dc2626', label:'左轉' },{ dur:12, bg:'#d97706', label:'黃燈' }],
    status: 'optimized' },
];

const statusCfg = {
  optimized: { label: '已優化', textColor: _SC.green400, bg: _SC.greenBg, border: _SC.greenBd },
  normal:    { label: '正常',   textColor: _SC.blue400,  bg: _SC.blueBg,  border: _SC.blueBd  },
  congested: { label: '壅塞中', textColor: _SC.red400,   bg: _SC.redBg,   border: _SC.redBd   },
};

function SignalControl() {
  const [selected, setSelected] = React.useState('IK7KP');
  const S = _SC;
  const sel = signalData.find(s => s.id === selected);

  return (
    <div style={{
      background: S.bgCard, border: `1px solid ${S.borderMed}`,
      borderRadius: 12, boxShadow: S.shadowCard,
      overflow: 'hidden', display: 'flex', flexDirection: 'column',
    }}>

      {/* Header */}
      <div style={{
        display: 'flex', alignItems: 'center', justifyContent: 'space-between',
        padding: '11px 14px', borderBottom: `1px solid ${S.borderMed}`,
      }}>
        <span style={{ fontFamily: S.ff, fontSize: 13, fontWeight: 600, color: S.fg, letterSpacing: '-0.01em' }}>號誌控制</span>
        <span style={{
          fontFamily: S.body, fontSize: 11, fontWeight: 500,
          color: S.green400, background: S.greenBg, border: `1px solid ${S.greenBd}`,
          borderRadius: 4, padding: '2px 8px',
        }}>6 組優化中</span>
      </div>

      {/* Signal list */}
      <div style={{ padding: '6px 8px' }}>
        {signalData.map(s => {
          const cfg = statusCfg[s.status];
          const isActive = selected === s.id;
          return (
            <div
              key={s.id}
              onClick={() => setSelected(s.id)}
              style={{
                display: 'flex', alignItems: 'center', justifyContent: 'space-between',
                padding: '8px 10px', borderRadius: 8, cursor: 'pointer', marginBottom: 2,
                background: isActive ? S.blueSubtle : 'transparent',
                transition: 'background 120ms',
              }}
            >
              <div style={{ flex: 1, minWidth: 0 }}>
                <div style={{ fontFamily: S.mono, fontSize: 11, color: S.blue400, marginBottom: 2 }}>{s.id}</div>
                <div style={{ fontFamily: S.body, fontSize: 12, color: S.fg2, whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>{s.name}</div>
              </div>
              <div style={{ display: 'flex', alignItems: 'center', gap: 8, flexShrink: 0 }}>
                <span style={{ fontFamily: S.mono, fontSize: 11, color: S.fg3 }}>{s.cycle}s</span>
                <span style={{
                  fontFamily: S.body, fontSize: 11, fontWeight: 500,
                  color: cfg.textColor, background: cfg.bg, border: `1px solid ${cfg.border}`,
                  borderRadius: 4, padding: '1px 7px',
                }}>{cfg.label}</span>
              </div>
            </div>
          );
        })}
      </div>

      {/* Phase detail */}
      {sel && (
        <div style={{ borderTop: `1px solid ${S.borderMed}`, padding: '10px 14px' }}>
          <div style={{ fontFamily: S.mono, fontSize: 11, color: S.fg2, marginBottom: 8 }}>
            {sel.id} — 週期 {sel.cycle}s
          </div>
          {/* Phase bar */}
          <div style={{ display: 'flex', borderRadius: 5, overflow: 'hidden', height: 22 }}>
            {sel.phases.map((p, i) => (
              <div key={i} style={{
                width: `${(p.dur / sel.cycle) * 100}%`,
                background: p.bg,
                display: 'flex', alignItems: 'center', justifyContent: 'center',
              }}>
                <span style={{ fontFamily: S.mono, fontSize: 10, color: 'rgba(255,255,255,0.9)', fontWeight: 600 }}>{p.dur}s</span>
              </div>
            ))}
          </div>
          {/* Legend */}
          <div style={{ display: 'flex', gap: 12, marginTop: 8 }}>
            {sel.phases.map((p, i) => (
              <div key={i} style={{ display: 'flex', alignItems: 'center', gap: 4, fontFamily: S.body, fontSize: 11, color: S.fg2 }}>
                <span style={{ width: 8, height: 8, borderRadius: '50%', background: p.bg, display: 'inline-block' }}/>
                {p.label} {p.dur}s
              </div>
            ))}
          </div>
        </div>
      )}

    </div>
  );
}

Object.assign(window, { SignalControl });
