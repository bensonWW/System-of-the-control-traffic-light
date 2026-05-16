// TrafficMap.jsx — TrafficVision v2 · Schematic Road Network (SVG)

/* ── v2 tokens ─────────────────────────────────── */
const _TM = {
  ff:   "'Inter', system-ui, sans-serif",
  body: "'Noto Sans TC', 'PingFang TC', sans-serif",
  mono: "'JetBrains Mono', monospace",
  bgBase:    '#020617',
  bgCard:    '#0f1629',
  bgElev:    '#162036',
  mapBg:     '#060c1a',
  borderMed: 'rgba(99,130,200,0.25)',
  divider:   'rgba(99,130,200,0.08)',
  shadowCard:'0 0 0 1px rgba(99,130,200,0.25), 0 1px 3px rgba(0,0,0,0.40)',
  blue400:   '#60a5fa',
  blue500:   '#3b82f6',
  green500:  '#22c55e',
  green400:  '#4ade80',
  greenGlow: 'rgba(34,197,94,0.25)',
  amber500:  '#f59e0b',
  amber400:  '#fbbf24',
  red500:    '#ef4444',
  red400:    '#f87171',
  redGlow:   'rgba(239,68,68,0.30)',
  fg:  '#f1f5f9',
  fg2: '#94a3b8',
  fg3: '#4e6080',
};

/* moe → { stroke, dim, label } */
const moeStyle = {
  0: { stroke: _TM.green500, dim: 'rgba(34,197,94,0.30)',  label: '暢通', labelColor: _TM.green400 },
  1: { stroke: _TM.amber500, dim: 'rgba(245,158,11,0.30)', label: '普通', labelColor: _TM.amber400 },
  2: { stroke: _TM.red500,   dim: 'rgba(239,68,68,0.30)',  label: '壅塞', labelColor: _TM.red400   },
};

const mapRoads = [
  /* Horizontal */
  { id: 'zhongxiao', name: '忠孝東路', x1:50,  y1:195, x2:550, y2:195, moe:1 },
  { id: 'bade',      name: '八德路',   x1:50,  y1:280, x2:550, y2:280, moe:2 },
  { id: 'shimin',    name: '市民大道', x1:50,  y1:365, x2:550, y2:365, moe:1 },
  /* Vertical */
  { id: 'jianguo-n', name: '建國北路', x1:260, y1:80,  x2:260, y2:430, moe:2, v:true },
  { id: 'jianguo-s', name: '建國南路', x1:320, y1:80,  x2:320, y2:430, moe:0, v:true },
  { id: 'xinsheng',  name: '新生南路', x1:170, y1:80,  x2:170, y2:430, moe:0, v:true },
  { id: 'songjiang', name: '松江路',   x1:390, y1:80,  x2:390, y2:430, moe:1, v:true },
];

/* Intersections */
const intersections = [
  [170,195],[260,195],[320,195],[390,195],
  [170,280],[260,280],[320,280],[390,280],
  [170,365],[260,365],[320,365],[390,365],
];

/* Congested nodes (moe=2 overlap points) */
const congestedNodes = [
  [260,195],[260,280],
];

function TrafficMap({ selectedRoad, onSelectRoad }) {
  const [hovered, setHovered] = React.useState(null);
  const T = _TM;

  return (
    <div style={{
      background: T.bgCard, border: `1px solid ${T.borderMed}`,
      borderRadius: 12, boxShadow: T.shadowCard,
      overflow: 'hidden', display: 'flex', flexDirection: 'column',
    }}>

      {/* Header */}
      <div style={{
        display: 'flex', alignItems: 'center', justifyContent: 'space-between',
        padding: '11px 14px', borderBottom: `1px solid ${T.borderMed}`,
      }}>
        <span style={{ fontFamily: T.ff, fontSize: 13, fontWeight: 600, color: T.fg, letterSpacing: '-0.01em' }}>路網示意圖</span>
        <div style={{ display: 'flex', gap: 14 }}>
          {Object.entries(moeStyle).map(([moe, cfg]) => (
            <div key={moe} style={{ display: 'flex', alignItems: 'center', gap: 5 }}>
              <span style={{ width: 20, height: 3, background: cfg.stroke, display: 'inline-block', borderRadius: 2 }}/>
              <span style={{ fontFamily: T.body, fontSize: 11, color: T.fg2 }}>{cfg.label}</span>
            </div>
          ))}
        </div>
      </div>

      {/* SVG map */}
      <svg width="100%" viewBox="0 0 600 490" style={{ flex: 1, display: 'block', background: T.mapBg }}>
        <defs>
          {/* Subtle grid */}
          <pattern id="tvgrid" width="40" height="40" patternUnits="userSpaceOnUse">
            <path d="M 40 0 L 0 0 0 40" fill="none" stroke="rgba(99,130,200,0.06)" strokeWidth="0.5"/>
          </pattern>
          {/* Congestion glow filter */}
          <filter id="congGlow" x="-50%" y="-50%" width="200%" height="200%">
            <feGaussianBlur stdDeviation="4" result="blur"/>
            <feComposite in="SourceGraphic" in2="blur" operator="over"/>
          </filter>
        </defs>

        {/* Grid background */}
        <rect width="600" height="490" fill="url(#tvgrid)"/>

        {/* NTUT campus block */}
        <rect x="193" y="203" width="62" height="72" rx="5"
          fill="rgba(59,130,246,0.06)" stroke="rgba(59,130,246,0.20)" strokeWidth="1"/>
        <text x="224" y="233" textAnchor="middle" fill={T.blue400} fontSize="9.5"
          fontFamily="Noto Sans TC" fontWeight="500">北科大</text>
        <text x="224" y="247" textAnchor="middle" fill={T.blue400} fontSize="8"
          fontFamily="JetBrains Mono" opacity="0.7">NTUT</text>

        {/* Roads */}
        {mapRoads.map(road => {
          const s = moeStyle[road.moe];
          const isHov = hovered === road.id;
          const isSel = selectedRoad === road.id;
          const active = isHov || isSel;
          return (
            <g key={road.id} onClick={() => onSelectRoad && onSelectRoad(road)} style={{ cursor: 'pointer' }}>
              {/* Glow layer */}
              {active && (
                <line x1={road.x1} y1={road.y1} x2={road.x2} y2={road.y2}
                  stroke={s.stroke} strokeWidth="9" strokeOpacity="0.15" strokeLinecap="round"/>
              )}
              {/* Road line */}
              <line
                x1={road.x1} y1={road.y1} x2={road.x2} y2={road.y2}
                stroke={active ? s.stroke : s.dim}
                strokeWidth={active ? 4.5 : 3}
                strokeLinecap="round"
                style={{ transition: 'stroke 120ms, stroke-width 120ms' }}
                onMouseEnter={() => setHovered(road.id)}
                onMouseLeave={() => setHovered(null)}
              />
              {/* Hit area */}
              <line x1={road.x1} y1={road.y1} x2={road.x2} y2={road.y2}
                stroke="transparent" strokeWidth={16}
                onMouseEnter={() => setHovered(road.id)}
                onMouseLeave={() => setHovered(null)}/>
              {/* Label */}
              <text
                x={road.v ? road.x1 + 7 : road.x1 + 8}
                y={road.v ? road.y1 + 16 : road.y1 - 7}
                fill={active ? s.labelColor : T.fg3}
                fontSize="9.5" fontFamily="Noto Sans TC"
                style={{ transition: 'fill 150ms', userSelect: 'none', pointerEvents: 'none' }}
              >{road.name}</text>
            </g>
          );
        })}

        {/* Intersections */}
        {intersections.map(([x, y], i) => (
          <circle key={i} cx={x} cy={y} r={4.5}
            fill={T.bgElev} stroke="rgba(99,130,200,0.40)" strokeWidth={1.5}/>
        ))}

        {/* Congested pulse nodes */}
        {congestedNodes.map(([x, y], i) => (
          <g key={i}>
            <circle cx={x} cy={y} r={10}
              fill="rgba(239,68,68,0.12)" stroke={T.red500} strokeWidth="1.5">
              <animate attributeName="r" values="8;14;8" dur={`${1.8 + i * 0.4}s`} repeatCount="indefinite"/>
              <animate attributeName="opacity" values="0.9;0.2;0.9" dur={`${1.8 + i * 0.4}s`} repeatCount="indefinite"/>
            </circle>
            <circle cx={x} cy={y} r={4.5} fill={T.red500} opacity="0.9"/>
          </g>
        ))}
      </svg>

    </div>
  );
}

Object.assign(window, { TrafficMap });
