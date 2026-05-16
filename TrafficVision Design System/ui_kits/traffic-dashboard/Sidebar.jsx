// Sidebar.jsx — TrafficVision v2 · Dark OLED + Glassmorphism

/* ── v2 tokens ─────────────────────────────────── */
const _S = {
  ff:   "'Inter', system-ui, sans-serif",
  body: "'Noto Sans TC', 'PingFang TC', sans-serif",
  mono: "'JetBrains Mono', monospace",
  bgSurf:     '#0a0f1e',
  borderMed:  'rgba(99,130,200,0.25)',
  blueSubtle: 'rgba(59,130,246,0.09)',
  blue400:    '#60a5fa',
  blue500:    '#3b82f6',
  blueBar:    '0 0 10px rgba(59,130,246,0.5)',
  green500:   '#22c55e',
  greenGlow:  '0 0 8px rgba(34,197,94,0.5), 0 0 3px #22c55e',
  fg:         '#f1f5f9',
  fg2:        '#94a3b8',
  fg3:        '#4e6080',
};

const SidebarItems = [
  { id: 'dashboard',  icon: 'grid',           label: '總覽儀表板' },
  { id: 'roads',      icon: 'map',            label: '路段監控'   },
  { id: 'signals',    icon: 'zap',            label: '號誌控制'   },
  { id: 'prediction', icon: 'activity',       label: '流量預測'   },
  { id: 'chat',       icon: 'message-square', label: 'AI 助理'    },
  { id: 'export',     icon: 'download',       label: '匯出報告'   },
];

const icons = {
  grid: <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round"><rect x="3" y="3" width="7" height="7" rx="1"/><rect x="14" y="3" width="7" height="7" rx="1"/><rect x="14" y="14" width="7" height="7" rx="1"/><rect x="3" y="14" width="7" height="7" rx="1"/></svg>,
  map: <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"><polygon points="1 6 1 22 8 18 16 22 23 18 23 2 16 6 8 2 1 6"/><line x1="8" y1="2" x2="8" y2="18"/><line x1="16" y1="6" x2="16" y2="22"/></svg>,
  zap: <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"><polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/></svg>,
  activity: <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/></svg>,
  'message-square': <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"><path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/></svg>,
  download: <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/></svg>,
};

function Sidebar({ active, onNav }) {
  return (
    <aside style={{
      width: 220, background: _S.bgSurf,
      borderRight: `1px solid ${_S.borderMed}`,
      display: 'flex', flexDirection: 'column', height: '100%', flexShrink: 0,
    }}>

      {/* Logo */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 10, padding: '16px', borderBottom: `1px solid ${_S.borderMed}` }}>
        <div style={{
          width: 34, height: 34, borderRadius: 9,
          background: 'rgba(59,130,246,0.12)',
          border: '1px solid rgba(59,130,246,0.22)',
          display: 'flex', alignItems: 'center', justifyContent: 'center', flexShrink: 0,
        }}>
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#60a5fa" strokeWidth="2" strokeLinecap="round">
            <circle cx="12" cy="12" r="10"/>
            <polyline points="12 6 12 12 15.5 14.5"/>
          </svg>
        </div>
        <div>
          <div style={{ fontFamily: _S.ff, fontSize: 14, fontWeight: 700, color: _S.fg, letterSpacing: '-0.02em' }}>TrafficVision</div>
          <div style={{ fontFamily: _S.body, fontSize: 10, color: _S.fg3, marginTop: 2 }}>NTUT 交通監控</div>
        </div>
      </div>

      {/* Section divider */}
      <div style={{ padding: '14px 16px 6px', fontFamily: _S.body, fontSize: 10, fontWeight: 500, letterSpacing: '0.08em', textTransform: 'uppercase', color: _S.fg3 }}>主選單</div>

      {/* Nav */}
      <nav style={{ flex: 1, padding: '0 8px', display: 'flex', flexDirection: 'column', gap: 1 }}>
        {SidebarItems.map(item => {
          const on = active === item.id;
          return (
            <button
              key={item.id}
              onClick={() => onNav && onNav(item.id)}
              style={{
                display: 'flex', alignItems: 'center', gap: 10,
                height: 36, padding: '0 10px',
                borderRadius: 8, border: 'none',
                background: on ? _S.blueSubtle : 'transparent',
                color: on ? _S.blue400 : _S.fg2,
                fontFamily: _S.body, fontSize: 13, fontWeight: on ? 500 : 400,
                cursor: 'pointer', width: '100%', textAlign: 'left',
                position: 'relative', transition: 'background 150ms, color 150ms',
              }}
            >
              {on && <div style={{ position: 'absolute', left: 0, top: '22%', bottom: '22%', width: 3, background: _S.blue500, borderRadius: '0 2px 2px 0', boxShadow: _S.blueBar }}/>}
              <span style={{ color: on ? _S.blue400 : _S.fg3, display: 'flex', flexShrink: 0 }}>{icons[item.icon]}</span>
              {item.label}
            </button>
          );
        })}
      </nav>

      {/* System status */}
      <div style={{ padding: '12px 16px', borderTop: `1px solid ${_S.borderMed}`, display: 'flex', alignItems: 'center', gap: 10 }}>
        <div style={{ width: 8, height: 8, borderRadius: '50%', background: _S.green500, boxShadow: _S.greenGlow, flexShrink: 0 }}/>
        <div>
          <div style={{ fontFamily: _S.body, fontSize: 12, color: _S.fg2 }}>系統運行中</div>
          <div style={{ fontFamily: _S.mono, fontSize: 10, color: _S.fg3, marginTop: 2 }}>更新週期 5 分鐘</div>
        </div>
      </div>

    </aside>
  );
}

Object.assign(window, { Sidebar });
