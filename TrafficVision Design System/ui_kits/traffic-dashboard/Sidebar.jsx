// Sidebar.jsx — TrafficVision navigation sidebar
const SidebarItems = [
  { id: 'dashboard', icon: 'grid', label: '總覽儀表板' },
  { id: 'roads', icon: 'map', label: '路段監控' },
  { id: 'signals', icon: 'zap', label: '號誌控制' },
  { id: 'prediction', icon: 'activity', label: '流量預測' },
  { id: 'chat', icon: 'message-square', label: 'AI 助理' },
  { id: 'export', icon: 'download', label: '匯出報告' },
];

const icons = {
  grid: <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5"><rect x="3" y="3" width="7" height="7"/><rect x="14" y="3" width="7" height="7"/><rect x="14" y="14" width="7" height="7"/><rect x="3" y="14" width="7" height="7"/></svg>,
  map: <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5"><polygon points="1 6 1 22 8 18 16 22 23 18 23 2 16 6 8 2 1 6"/><line x1="8" y1="2" x2="8" y2="18"/><line x1="16" y1="6" x2="16" y2="22"/></svg>,
  zap: <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5"><polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/></svg>,
  activity: <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/></svg>,
  'message-square': <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5"><path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/></svg>,
  download: <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/></svg>,
  settings: <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5"><circle cx="12" cy="12" r="3"/><path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-4 0v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83-2.83l.06-.06A1.65 1.65 0 0 0 4.68 15a1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1 0-4h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 2.83-2.83l.06.06A1.65 1.65 0 0 0 9 4.68a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 4 0v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 2.83l-.06.06A1.65 1.65 0 0 0 19.4 9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1z"/></svg>,
};

function Sidebar({ active, onNav }) {
  return (
    <aside style={sidebarStyles.sidebar}>
      <div style={sidebarStyles.logo}>
        <div style={sidebarStyles.logoMark}>
          <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#60a5fa" strokeWidth="2">
            <circle cx="12" cy="12" r="10"/>
            <line x1="12" y1="8" x2="12" y2="12"/>
            <line x1="12" y1="12" x2="15" y2="15"/>
          </svg>
        </div>
        <div>
          <div style={sidebarStyles.logoTitle}>TrafficVision</div>
          <div style={sidebarStyles.logoSub}>NTUT 交通監控</div>
        </div>
      </div>
      <nav style={sidebarStyles.nav}>
        {SidebarItems.map(item => (
          <button key={item.id} style={{...sidebarStyles.navItem, ...(active === item.id ? sidebarStyles.navItemActive : {})}} onClick={() => onNav(item.id)}>
            <span style={{color: active === item.id ? '#60a5fa' : '#94a3b8'}}>{icons[item.icon]}</span>
            <span style={sidebarStyles.navLabel}>{item.label}</span>
            {active === item.id && <div style={sidebarStyles.activeBar}></div>}
          </button>
        ))}
      </nav>
      <div style={sidebarStyles.bottom}>
        <div style={sidebarStyles.statusDot}></div>
        <div>
          <div style={sidebarStyles.statusText}>系統運行中</div>
          <div style={sidebarStyles.statusSub}>更新週期 5 分鐘</div>
        </div>
      </div>
    </aside>
  );
}

const sidebarStyles = {
  sidebar: { width: 220, background: '#111827', borderRight: '1px solid #2a3555', display: 'flex', flexDirection: 'column', height: '100%', flexShrink: 0 },
  logo: { display: 'flex', alignItems: 'center', gap: 10, padding: '18px 16px', borderBottom: '1px solid #2a3555' },
  logoMark: { width: 36, height: 36, background: 'rgba(59,130,246,0.12)', borderRadius: 8, display: 'flex', alignItems: 'center', justifyContent: 'center', flexShrink: 0 },
  logoTitle: { fontFamily: "'Space Grotesk', sans-serif", fontSize: 15, fontWeight: 700, color: '#e2e8f0', letterSpacing: '-0.01em' },
  logoSub: { fontFamily: "'Noto Sans TC', sans-serif", fontSize: 11, color: '#475569', marginTop: 1 },
  nav: { flex: 1, padding: '12px 8px', display: 'flex', flexDirection: 'column', gap: 2 },
  navItem: { display: 'flex', alignItems: 'center', gap: 10, padding: '9px 10px', borderRadius: 7, border: 'none', background: 'transparent', cursor: 'pointer', width: '100%', textAlign: 'left', position: 'relative', transition: 'background 150ms' },
  navItemActive: { background: 'rgba(59,130,246,0.10)' },
  navLabel: { fontFamily: "'Noto Sans TC', sans-serif", fontSize: 13, color: '#e2e8f0', fontWeight: 400 },
  activeBar: { position: 'absolute', right: 0, top: '20%', bottom: '20%', width: 3, background: '#3b82f6', borderRadius: 2 },
  bottom: { padding: '14px 16px', borderTop: '1px solid #2a3555', display: 'flex', alignItems: 'center', gap: 10 },
  statusDot: { width: 8, height: 8, borderRadius: '50%', background: '#22c55e', boxShadow: '0 0 6px rgba(34,197,94,0.6)', flexShrink: 0 },
  statusText: { fontFamily: "'Noto Sans TC', sans-serif", fontSize: 12, color: '#94a3b8' },
  statusSub: { fontFamily: "'JetBrains Mono', monospace", fontSize: 10, color: '#475569', marginTop: 2 },
};

Object.assign(window, { Sidebar });
