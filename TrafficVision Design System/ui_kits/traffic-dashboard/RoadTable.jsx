// RoadTable.jsx — TrafficVision v2 · Road Segment Data Table

/* ── v2 tokens ─────────────────────────────────── */
const _RT = {
  ff:   "'Inter', system-ui, sans-serif",
  body: "'Noto Sans TC', 'PingFang TC', sans-serif",
  mono: "'JetBrains Mono', monospace",
  bgCard:     '#0f1629',
  bgSurf:     '#0a0f1e',
  bgElev:     '#162036',
  borderMed:  'rgba(99,130,200,0.25)',
  divider:    'rgba(99,130,200,0.08)',
  shadowCard: '0 0 0 1px rgba(99,130,200,0.25), 0 1px 3px rgba(0,0,0,0.40)',
  blue400:    '#60a5fa',
  blue500:    '#3b82f6',
  blueSubtle: 'rgba(59,130,246,0.08)',
  green400:   '#4ade80',
  green500:   '#22c55e',
  greenBg:    'rgba(34,197,94,0.10)',
  greenBd:    'rgba(34,197,94,0.25)',
  amber400:   '#fbbf24',
  amber500:   '#f59e0b',
  amberBg:    'rgba(245,158,11,0.10)',
  amberBd:    'rgba(245,158,11,0.25)',
  red400:     '#f87171',
  red500:     '#ef4444',
  redBg:      'rgba(239,68,68,0.10)',
  redBd:      'rgba(239,68,68,0.25)',
  fg:  '#f1f5f9',
  fg2: '#94a3b8',
  fg3: '#4e6080',
  fgD: '#2d3d55',
};

const roadData = [
  { name: '建國南路  忠孝–仁愛', id: 'ZJHKR40', speed: 34.6, vol: 61,  occ: 2.1,  moe: 1 },
  { name: '建國北路  長安–忠孝', id: 'ZK7KP40', speed: 9.8,  vol: 30,  occ: 47.7, moe: 2 },
  { name: '市民大道  建國–金山', id: 'ZK9KC60', speed: 42.8, vol: 123, occ: 9.0,  moe: 0 },
  { name: '八德路    建國–新生', id: 'ZK5JW60', speed: 30.5, vol: 33,  occ: 3.1,  moe: 1 },
  { name: '忠孝東路  復興–建國', id: 'ZJHKR20', speed: 47.5, vol: 154, occ: 4.3,  moe: 1 },
  { name: '市民大道  金山–建國', id: 'ZKJJP20', speed: 42.1, vol: 202, occ: 5.6,  moe: 1 },
  { name: '忠孝東路  金山–中山', id: 'ZJSJD40', speed: 20.5, vol: 83,  occ: 12.6, moe: 2 },
  { name: '建國南路  仁愛–信義', id: 'ZINKW40', speed: 41.0, vol: 52,  occ: 1.4,  moe: 0 },
];

const MOEConfig = {
  0: { label: '暢通', textColor: _RT.green400,  bg: _RT.greenBg,  border: _RT.greenBd,  dot: _RT.green500,  speedColor: _RT.green400  },
  1: { label: '普通', textColor: _RT.amber400,  bg: _RT.amberBg,  border: _RT.amberBd,  dot: _RT.amber500,  speedColor: _RT.fg        },
  2: { label: '壅塞', textColor: _RT.red400,    bg: _RT.redBg,    border: _RT.redBd,    dot: _RT.red500,    speedColor: _RT.red400    },
};

const COLS = ['路段名稱', '平均車速', '車流量', '佔有率', '狀態'];

function RoadTable({ onSelectRoad, selectedId }) {
  const R = _RT;
  const [hovered, setHovered] = React.useState(null);

  return (
    <div style={{
      background: R.bgCard,
      border: `1px solid ${R.borderMed}`,
      borderRadius: 12,
      boxShadow: R.shadowCard,
      overflow: 'hidden', flex: 1,
      display: 'flex', flexDirection: 'column',
    }}>

      {/* Header */}
      <div style={{
        display: 'flex', alignItems: 'center', justifyContent: 'space-between',
        padding: '11px 16px',
        borderBottom: `1px solid ${R.borderMed}`,
      }}>
        <div style={{ fontFamily: R.ff, fontSize: 13, fontWeight: 600, color: R.fg, letterSpacing: '-0.01em' }}>
          路段監控列表
        </div>
        {/* Search mock */}
        <div style={{
          display: 'flex', alignItems: 'center', gap: 6,
          background: R.bgSurf, border: `1px solid ${R.borderMed}`,
          borderRadius: 6, padding: '5px 10px',
        }}>
          <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke={R.fg3} strokeWidth="2" strokeLinecap="round"><circle cx="11" cy="11" r="8"/><path d="m21 21-4.35-4.35"/></svg>
          <span style={{ fontFamily: R.body, fontSize: 12, color: R.fg3 }}>搜尋路段...</span>
        </div>
      </div>

      {/* Table */}
      <div style={{ overflowY: 'auto', flex: 1 }}>
        <table style={{ width: '100%', borderCollapse: 'collapse' }}>
          <thead>
            <tr style={{ background: R.bgSurf }}>
              {COLS.map(h => (
                <th key={h} style={{
                  fontFamily: R.body, fontSize: 10, fontWeight: 500,
                  letterSpacing: '0.07em', textTransform: 'uppercase',
                  color: R.fg3, padding: '7px 14px', textAlign: 'left',
                  borderBottom: `1px solid ${R.borderMed}`,
                  whiteSpace: 'nowrap',
                }}>{h}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {roadData.map(row => {
              const moe = MOEConfig[row.moe];
              const isSelected = selectedId === row.id;
              const isHovered = hovered === row.id;
              return (
                <tr
                  key={row.id}
                  onClick={() => onSelectRoad && onSelectRoad(row)}
                  onMouseEnter={() => setHovered(row.id)}
                  onMouseLeave={() => setHovered(null)}
                  style={{
                    cursor: 'pointer',
                    background: isSelected
                      ? R.blueSubtle
                      : isHovered ? 'rgba(99,130,200,0.05)' : 'transparent',
                    borderBottom: `1px solid ${R.divider}`,
                    transition: 'background 120ms',
                  }}
                >
                  {/* Road name */}
                  <td style={{ padding: '9px 14px' }}>
                    <div style={{ fontFamily: R.body, fontSize: 13, fontWeight: 500, color: R.fg }}>{row.name}</div>
                    <div style={{ fontFamily: R.mono, fontSize: 10, color: R.fg3, marginTop: 2 }}>{row.id}</div>
                  </td>

                  {/* Speed */}
                  <td style={{ padding: '9px 14px', whiteSpace: 'nowrap' }}>
                    <span style={{
                      fontFamily: R.ff, fontWeight: 600, fontSize: 14,
                      fontVariantNumeric: 'tabular-nums',
                      color: moe.speedColor,
                    }}>{row.speed}</span>
                    <span style={{ fontFamily: R.body, fontSize: 12, color: R.fg2, marginLeft: 3 }}>km/h</span>
                  </td>

                  {/* Volume */}
                  <td style={{ padding: '9px 14px', whiteSpace: 'nowrap' }}>
                    <span style={{ fontFamily: R.ff, fontWeight: 500, fontSize: 13, fontVariantNumeric: 'tabular-nums', color: R.fg }}>{row.vol}</span>
                    <span style={{ fontFamily: R.body, fontSize: 11, color: R.fg2, marginLeft: 3 }}>輛</span>
                  </td>

                  {/* Occupancy */}
                  <td style={{ padding: '9px 14px' }}>
                    <span style={{ fontFamily: R.mono, fontSize: 12, fontVariantNumeric: 'tabular-nums', color: R.fg2 }}>{row.occ}%</span>
                  </td>

                  {/* MOE badge */}
                  <td style={{ padding: '9px 14px' }}>
                    <span style={{
                      display: 'inline-flex', alignItems: 'center', gap: 5,
                      background: moe.bg,
                      color: moe.textColor,
                      border: `1px solid ${moe.border}`,
                      borderRadius: 4, padding: '2px 9px',
                      fontSize: 12, fontWeight: 500, fontFamily: R.body,
                      whiteSpace: 'nowrap',
                    }}>
                      <span style={{ width: 6, height: 6, borderRadius: '50%', background: moe.dot, flexShrink: 0 }}/>
                      {moe.label}
                    </span>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

    </div>
  );
}

Object.assign(window, { RoadTable, roadData, MOEConfig });
