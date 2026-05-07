# TrafficVision 交通視覺化系統 — Design System

## Overview

**TrafficVision** is a web-based traffic data monitoring and intelligent signal control system focused on the area surrounding **National Taipei University of Technology (NTUT / 台北科技大學)**. The system:

1. **Fetches real-time traffic data** from the Taipei City Open Data API (GetVD endpoint), capturing speed, volume, occupancy, and MOE (Measure of Effectiveness) levels for road segments.
2. **Simulates traffic flow** using SUMO (Simulation of Urban MObility) to model vehicle routing and signal timing.
3. **Predicts future traffic** using a GRU (Gated Recurrent Unit) deep learning model (`gru_traffic_model.pth`).
4. **Optimizes traffic signals** — automatically selecting and applying the best signal timing strategy based on predicted congestion scores.
5. **Presents data to users** through a web dashboard designed to make complex traffic metrics easy to understand, enhanced by an LLM assistant for natural language data interpretation.

### Primary Domain
Roads around NTUT campus in central Taipei: 忠孝東路、八德路、建國南/北路、市民大道、新生南路.

### Key Data Fields
| Field | Chinese | Meaning |
|---|---|---|
| `AvgSpd` | 平均車速 | Average speed (km/h) |
| `TotalVol` | 總車流量 | Total vehicle volume |
| `AvgOcc` | 平均佔有率 | Average road occupancy (%) |
| `MOELevel` | 服務水準 | 0 = Free flow, 1 = Moderate, 2 = Congested |

---

## Sources

- **Local codebase**: `System-of-the-control-traffic-light-kaven/System-of-the-control-traffic-light-kaven/`
- **GitHub repo**: `https://github.com/bensonWW/System-of-the-control-traffic-light` (branch: `main`)
- No Figma file provided.

---

## CONTENT FUNDAMENTALS

### Language & Tone
- **Primary language**: Traditional Chinese (繁體中文) for all UI labels, tooltips, and messages.
- **Secondary**: English for technical identifiers (road IDs, API fields, timestamps).
- **Tone**: Professional, data-driven, neutral. This is a control-room-adjacent tool — no casual language.
- **Casing (English)**: Sentence case for labels, ALL CAPS for status indicators (e.g. `CONGESTED`, `FREE FLOW`).
- **Numbers**: Always show units — `34.6 km/h`, `61 輛`, `2.1%`.
- **Timestamps**: `YYYY-MM-DD HH:mm:ss` format, always shown in local time (Asia/Taipei).
- **No emoji** in data displays; traffic signal icons use dedicated SVG/icon assets only.
- **Pronouns**: System speaks to operators as 您 (formal you); LLM responses use 您 for respectful user-facing copy.

### Copy Examples
- Status labels: `暢通` (free flow), `壅塞` (congested), `普通` (moderate)
- Actions: `重新整理`, `匯出報告`, `查看詳情`, `啟動優化`
- LLM prompts: `請問目前建國南路的車流狀況如何？`
- Empty states: `暫無數據`, `載入中...`
- Errors: `資料擷取失敗，請稍後再試`

---

## VISUAL FOUNDATIONS

### Color System
Dark-mode-first. Designed for extended monitoring sessions in dimly-lit control environments.

**Backgrounds**: Deep navy-black base (`#0a0e1a`), layered with dark surface (`#111827`) and card (`#1a2236`).
**Borders**: Subtle blue-grey (`#2a3555`), near-invisible at rest; slightly brighter on hover.
**Primary accent**: Electric blue (`#3b82f6`) — used for interactive elements, map overlays, active states.
**Traffic status colors**:
- 🟢 Free flow (MOE 0 / 暢通): `#22c55e`
- 🟡 Moderate (MOE 1 / 普通): `#f59e0b`
- 🔴 Congested (MOE 2 / 壅塞): `#ef4444`
**Text**: `#e2e8f0` (primary), `#94a3b8` (secondary/label), `#475569` (muted/disabled).

### Typography
- **Display / Headings**: `Space Grotesk` — used for large numbers, section titles, metric values. Bold weight.
- **Body / UI**: `Noto Sans TC` — used for all Chinese text, table content, labels.
- **Mono / Data**: `JetBrains Mono` — road IDs, timestamps, raw values, code.
- Type scale: 12 / 13 / 14 / 16 / 20 / 24 / 32 / 48px.

### Spacing
4px base grid. Common tokens: 4, 8, 12, 16, 24, 32, 48, 64px.

### Backgrounds & Texture
- No gradients except subtle scrim overlays on map (dark linear gradient from bottom).
- No background images or illustrations in the dashboard.
- Card surfaces use a very slight inner border (`1px solid #2a3555`) with no drop shadow — flat, crisp.
- Map tiles: dark-mode tile style (mapbox dark or equivalent).

### Animation
- Transitions: `150ms ease-out` for hover states; `250ms ease` for panel reveals.
- Data refresh: number counters animate with brief fade-in.
- No bouncy or spring animations — all easing is standard.
- Traffic light status changes: smooth color cross-fade `200ms`.

### Hover & Press States
- Interactive rows: background lightens to `rgba(59,130,246,0.08)` on hover.
- Buttons: opacity drops to 0.85 on hover; scale `0.97` on press.
- Icon buttons: circular ripple fill `rgba(255,255,255,0.06)`.

### Cards & Containers
- Rounded corners: `8px` for cards, `6px` for inputs, `4px` for badges/chips.
- No drop shadows; separation achieved via border and background contrast.
- Card padding: `16px` standard, `12px` compact (data tables).

### Borders
- Standard: `1px solid #2a3555`
- Active/Focus: `1px solid #3b82f6`
- Dividers: `1px solid rgba(42,53,85,0.6)`

### Transparency & Blur
- Modal overlays: `backdrop-filter: blur(8px)` with `rgba(10,14,26,0.8)` tint.
- Sidebar: no blur — solid background.
- Tooltip: `rgba(26,34,54,0.96)` near-opaque.

### Corner Radii
- Cards: `8px`
- Buttons: `6px`
- Badges/chips: `4px`
- Input fields: `6px`
- Map pins: `50%` (circle)

### Imagery
- No stock photography.
- Map as primary "image" — SUMO network visualization or Mapbox dark tile.
- Status visualization: color-coded road segments overlaid on map.

### Layout
- Sidebar navigation: 240px fixed left; collapses to 56px icon rail.
- Main content: fluid, min 800px.
- Dashboard grid: 12-column, 24px gutter.
- Top bar: 56px fixed.

---

## ICONOGRAPHY

No existing icon font or SVG sprite was found in the codebase. The system uses **no inline emoji** for data displays.

**Recommended**: Use [Lucide Icons](https://lucide.dev/) via CDN — clean, 24px stroke-based SVGs. Matches the clean technical aesthetic. Weight: `1.5px` stroke.

Key icons used in this system:
- `traffic-cone` / `map-pin` — road segment markers
- `activity` — traffic flow charts
- `signal` / `radio` — signal strength / data status
- `clock` — timing / simulation time
- `zap` — optimization / signal control
- `message-square` — LLM assistant chat
- `refresh-cw` — data refresh
- `download` — export
- `alert-triangle` — congestion warning
- `check-circle` — free flow status
- `bar-chart-2` — metrics panel

**Usage**: Always 20×20px or 16×16px in UI; 24×24px in empty states. `currentColor` fill. Never scale above 32px.

---

## VISUAL FOUNDATIONS SUMMARY

| Property | Value |
|---|---|
| Background base | `#0a0e1a` |
| Surface | `#111827` |
| Card | `#1a2236` |
| Border | `#2a3555` |
| Accent blue | `#3b82f6` |
| Free flow green | `#22c55e` |
| Moderate amber | `#f59e0b` |
| Congested red | `#ef4444` |
| Text primary | `#e2e8f0` |
| Text secondary | `#94a3b8` |
| Text muted | `#475569` |
| Radius card | `8px` |
| Radius button | `6px` |
| Font display | Space Grotesk |
| Font body | Noto Sans TC |
| Font mono | JetBrains Mono |
| Transition | `150ms ease-out` |

---

## File Index

```
README.md                          This file
SKILL.md                           Agent skill definition
colors_and_type.css                CSS custom properties for colors + typography
assets/                            Logos, icons, visual assets
preview/                           Design system card previews (registered in Design System tab)
  colors-base.html                 Base color palette
  colors-traffic-status.html       Traffic status color system
  colors-semantic.html             Semantic color tokens
  type-scale.html                  Typography scale
  type-specimens.html              Type specimens (display, body, mono)
  spacing-tokens.html              Spacing scale tokens
  spacing-radii.html               Border radius + border tokens
  components-buttons.html          Button variants
  components-badges.html           Status badges
  components-data-card.html        Metric data card
  components-table-row.html        Data table rows
  components-signal-light.html     Traffic signal indicator
  components-input.html            Form inputs
  components-llm-chat.html         LLM chat bubble
ui_kits/
  traffic-dashboard/               Main web dashboard UI kit
    index.html                     Interactive dashboard prototype
    Sidebar.jsx                    Navigation sidebar component
    TopBar.jsx                     Top navigation bar
    TrafficMap.jsx                 Road network map panel
    MetricsPanel.jsx               KPI metrics cards
    RoadTable.jsx                  Road segment data table
    SignalControl.jsx              Traffic signal timing panel
    LLMChat.jsx                    LLM assistant chat panel
```
