# Traffic Dashboard UI Kit

Interactive dashboard prototype for TrafficVision — NTUT 交通監控系統.

## Design Width
1280px (responsive fluid layout)

## Screens
- **總覽儀表板** — KPI metrics, road table, map, signal control, LLM chat
- **路段監控** — Full road segment data table
- **號誌控制** — Signal timing panel + map view
- **AI 數據助理** — Full-height LLM chat interface
- **流量預測 / 匯出** — Placeholder screens

## Components
| File | Description |
|---|---|
| `Sidebar.jsx` | Left navigation with logo, nav items, system status |
| `TopBar.jsx` | Top bar with page title, timestamp, refresh & optimize buttons |
| `MetricsPanel.jsx` | 6-column KPI cards (speed, volume, occupancy, status counts) |
| `RoadTable.jsx` | Sortable road segment table with MOE badge and hover state |
| `TrafficMap.jsx` | SVG schematic road network map with color-coded MOE segments |
| `SignalControl.jsx` | Signal timing list + phase bar visualization |
| `LLMChat.jsx` | AI assistant chat panel with suggested prompts |

## Usage
Open `index.html` directly in a browser. Toggle **Tweaks** in the toolbar to show/hide map, chat, and change accent color.

## Design Tokens
Import `../../colors_and_type.css` for all CSS custom properties.
