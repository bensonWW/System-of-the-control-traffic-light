// TopBar.jsx — TrafficVision v2 · Dark OLED

/* ── v2 tokens ─────────────────────────────────── */
const _TB = {
  ff:   "'Inter', system-ui, sans-serif",
  body: "'Noto Sans TC', 'PingFang TC', sans-serif",
  mono: "'JetBrains Mono', monospace",
  bgSurf:    '#0a0f1e',
  borderMed: 'rgba(99,130,200,0.25)',
  blue400:   '#60a5fa',
  blue500:   '#3b82f6',
  blue600:   '#2563eb',
  blueSubtle:'rgba(59,130,246,0.10)',
  blueBorder:'rgba(59,130,246,0.25)',
  green400:  '#4ade80',
  green500:  '#22c55e',
  greenSubtle:'rgba(34,197,94,0.10)',
  greenBorder:'rgba(34,197,94,0.25)',
  fg:        '#f1f5f9',
  fg2:       '#94a3b8',
  fg3:       '#4e6080',
  cardBg:    '#0f1629',
  cardBorder:'rgba(99,130,200,0.25)',
};

function TopBar({ title, lastUpdated, onRefresh, onOptimize }) {
  return (
    <div style={{
      height: 52, background: _TB.bgSurf,
      borderBottom: `1px solid ${_TB.borderMed}`,
      display: 'flex', alignItems: 'center', justifyContent: 'space-between',
      padding: '0 20px', flexShrink: 0,
    }}>

      {/* Left: title + timestamp */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 16 }}>
        <div style={{ fontFamily: _TB.ff, fontSize: 15, fontWeight: 600, color: _TB.fg, letterSpacing: '-0.01em' }}>
          {title || '總覽儀表板'}
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: 5 }}>
          <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke={_TB.fg3} strokeWidth="2" strokeLinecap="round">
            <circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/>
          </svg>
          <span style={{ fontFamily: _TB.mono, fontSize: 11, color: _TB.fg3 }}>
            最後更新：{lastUpdated || '—'}
          </span>
        </div>
      </div>

      {/* Right: badges + actions */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>

        {/* Live data badge */}
        <div style={{
          display: 'flex', alignItems: 'center', gap: 6,
          background: _TB.greenSubtle, border: `1px solid ${_TB.greenBorder}`,
          borderRadius: 6, padding: '4px 10px',
        }}>
          <div style={{ width: 6, height: 6, borderRadius: '50%', background: _TB.green500, boxShadow: '0 0 5px rgba(34,197,94,0.7)' }}/>
          <span style={{ fontFamily: _TB.body, fontSize: 11, color: _TB.green400 }}>即時資料</span>
        </div>

        {/* Refresh button */}
        <button
          onClick={onRefresh}
          style={{
            display: 'inline-flex', alignItems: 'center', gap: 6,
            height: 30, padding: '0 12px', borderRadius: 6,
            border: `1px solid ${_TB.cardBorder}`,
            background: _TB.cardBg, color: _TB.fg2,
            fontFamily: _TB.body, fontSize: 12,
            cursor: 'pointer', transition: 'background 150ms, color 150ms',
          }}
          onMouseEnter={e => { e.currentTarget.style.background = 'rgba(99,130,200,0.08)'; e.currentTarget.style.color = _TB.fg; }}
          onMouseLeave={e => { e.currentTarget.style.background = _TB.cardBg; e.currentTarget.style.color = _TB.fg2; }}
        >
          <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round"><polyline points="23 4 23 10 17 10"/><path d="M20.49 15a9 9 0 1 1-.08-7.49"/></svg>
          重新整理
        </button>

        {/* Optimize button (primary) */}
        <button
          onClick={onOptimize}
          style={{
            display: 'inline-flex', alignItems: 'center', gap: 6,
            height: 30, padding: '0 14px', borderRadius: 6,
            border: 'none', background: _TB.blue500, color: '#fff',
            fontFamily: _TB.body, fontSize: 12, fontWeight: 500,
            cursor: 'pointer', transition: 'background 150ms, box-shadow 150ms',
          }}
          onMouseEnter={e => { e.currentTarget.style.background = _TB.blue400; e.currentTarget.style.boxShadow = '0 0 12px rgba(59,130,246,0.40)'; }}
          onMouseLeave={e => { e.currentTarget.style.background = _TB.blue500; e.currentTarget.style.boxShadow = 'none'; }}
        >
          <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"><polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/></svg>
          啟動優化
        </button>

      </div>
    </div>
  );
}

Object.assign(window, { TopBar });
