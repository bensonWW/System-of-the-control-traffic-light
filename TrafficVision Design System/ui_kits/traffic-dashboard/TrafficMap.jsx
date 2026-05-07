// TrafficMap.jsx — Road network map visualization (schematic, no external tiles)
const mapRoads = [
  // Horizontal roads
  { id: 'zhongxiao', name: '忠孝東路', x1: 50, y1: 195, x2: 550, y2: 195, moe: 1 },
  { id: 'bade', name: '八德路', x1: 50, y1: 280, x2: 550, y2: 280, moe: 2 },
  { id: 'shimin', name: '市民大道', x1: 50, y1: 360, x2: 550, y2: 360, moe: 1 },
  // Vertical roads
  { id: 'jianguo-n', name: '建國北路', x1: 260, y1: 80, x2: 260, y2: 420, moe: 2, vertical: true },
  { id: 'jianguo-s', name: '建國南路', x1: 320, y1: 80, x2: 320, y2: 420, moe: 0, vertical: true },
  { id: 'xinsheng', name: '新生南路', x1: 170, y1: 80, x2: 170, y2: 420, moe: 0, vertical: true },
  { id: 'songjiang', name: '松江路', x1: 380, y1: 80, x2: 380, y2: 420, moe: 1, vertical: true },
];

const intersections = [
  { x: 170, y: 195 }, { x: 260, y: 195 }, { x: 320, y: 195 }, { x: 380, y: 195 },
  { x: 170, y: 280 }, { x: 260, y: 280 }, { x: 320, y: 280 }, { x: 380, y: 280 },
  { x: 170, y: 360 }, { x: 260, y: 360 }, { x: 320, y: 360 }, { x: 380, y: 360 },
];

const moeColors = { 0: '#22c55e', 1: '#f59e0b', 2: '#ef4444' };

function TrafficMap({ selectedRoad, onSelectRoad }) {
  const [hovered, setHovered] = React.useState(null);

  return (
    <div style={mapStyles.wrap}>
      <div style={mapStyles.header}>
        <span style={mapStyles.title}>路網示意圖</span>
        <div style={mapStyles.legend}>
          {[[0,'暢通'],[1,'普通'],[2,'壅塞']].map(([moe, label]) => (
            <div key={moe} style={mapStyles.legendItem}>
              <span style={{width:20,height:3,background:moeColors[moe],display:'inline-block',borderRadius:2}}></span>
              <span style={mapStyles.legendLabel}>{label}</span>
            </div>
          ))}
        </div>
      </div>
      <svg width="100%" viewBox="0 0 600 480" style={mapStyles.svg}>
        {/* Grid background */}
        <defs>
          <pattern id="grid" width="40" height="40" patternUnits="userSpaceOnUse">
            <path d="M 40 0 L 0 0 0 40" fill="none" stroke="rgba(42,53,85,0.3)" strokeWidth="0.5"/>
          </pattern>
        </defs>
        <rect width="600" height="480" fill="url(#grid)"/>
        {/* NTUT label */}
        <rect x="195" y="200" width="60" height="75" rx="4" fill="rgba(59,130,246,0.10)" stroke="rgba(59,130,246,0.3)" strokeWidth="1"/>
        <text x="225" y="232" textAnchor="middle" fill="#60a5fa" fontSize="9" fontFamily="Noto Sans TC">北科大</text>
        <text x="225" y="246" textAnchor="middle" fill="#60a5fa" fontSize="7" fontFamily="JetBrains Mono">NTUT</text>

        {/* Roads */}
        {mapRoads.map(road => (
          <g key={road.id} onClick={() => onSelectRoad && onSelectRoad(road)} style={{cursor:'pointer'}}>
            <line
              x1={road.x1} y1={road.y1} x2={road.x2} y2={road.y2}
              stroke={hovered === road.id || selectedRoad === road.id ? moeColors[road.moe] : `${moeColors[road.moe]}99`}
              strokeWidth={hovered === road.id || selectedRoad === road.id ? 5 : 3}
              strokeLinecap="round"
              onMouseEnter={() => setHovered(road.id)}
              onMouseLeave={() => setHovered(null)}
            />
            <line
              x1={road.x1} y1={road.y1} x2={road.x2} y2={road.y2}
              stroke="transparent" strokeWidth={14}
              onMouseEnter={() => setHovered(road.id)}
              onMouseLeave={() => setHovered(null)}
            />
            <text
              x={road.vertical ? road.x1 + 6 : road.x1 + 6}
              y={road.vertical ? road.y1 + 14 : road.y1 - 6}
              fill="#94a3b8" fontSize="9" fontFamily="Noto Sans TC"
            >{road.name}</text>
          </g>
        ))}

        {/* Intersections */}
        {intersections.map((pt, i) => (
          <circle key={i} cx={pt.x} cy={pt.y} r={4} fill="#1a2236" stroke="#2a3555" strokeWidth={1.5}/>
        ))}

        {/* Congested marker */}
        <circle cx={260} cy={195} r={8} fill="rgba(239,68,68,0.25)" stroke="#ef4444" strokeWidth={1.5}>
          <animate attributeName="r" values="8;12;8" dur="2s" repeatCount="indefinite"/>
          <animate attributeName="opacity" values="1;0.4;1" dur="2s" repeatCount="indefinite"/>
        </circle>
        <circle cx={260} cy={280} r={8} fill="rgba(239,68,68,0.25)" stroke="#ef4444" strokeWidth={1.5}>
          <animate attributeName="r" values="8;12;8" dur="2.4s" repeatCount="indefinite"/>
          <animate attributeName="opacity" values="1;0.4;1" dur="2.4s" repeatCount="indefinite"/>
        </circle>
      </svg>
    </div>
  );
}

const mapStyles = {
  wrap: { background: '#0f1825', border: '1px solid #2a3555', borderRadius: 8, overflow: 'hidden', display: 'flex', flexDirection: 'column' },
  header: { display: 'flex', alignItems: 'center', justifyContent: 'space-between', padding: '10px 14px', borderBottom: '1px solid #2a3555' },
  title: { fontFamily: "'Space Grotesk',sans-serif", fontSize: 13, fontWeight: 600, color: '#e2e8f0' },
  legend: { display: 'flex', gap: 12 },
  legendItem: { display: 'flex', alignItems: 'center', gap: 5 },
  legendLabel: { fontFamily: "'Noto Sans TC',sans-serif", fontSize: 11, color: '#94a3b8' },
  svg: { flex: 1 },
};

Object.assign(window, { TrafficMap });
