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

## VISUAL FOUNDATIONS — v2.0

> Redesigned with ui-ux-pro-max: **Dark Mode OLED** style + **Glassmorphism** elevation.
> Priority: Accessibility (WCAG AA+), operational readability, data density.

### Color System

Dark-mode-only. OLED-optimised with near-black base to maximise contrast and reduce eye strain in dimly-lit control environments.

**Backgrounds** (deepened from v1 for true OLED contrast):
- Base: `#020617` — OLED near-black
- Surface: `#0a0f1e` — page surface layer
- Card: `#0f1629` — content containers
- Elevated: `#162036` — modals, popovers
- Glass: `rgba(15,22,41,0.72)` — backdrop-blur surfaces

**Borders** (changed from opaque blue-grey → alpha-based for robustness):
- Default: `rgba(99,130,200,0.15)` — near-invisible at rest
- Medium: `rgba(99,130,200,0.25)` — card/panel edges
- Strong: `rgba(99,130,200,0.40)` — active states
- Focus: `#3b82f6` — keyboard focus ring

**Primary accent**: Electric blue `#3b82f6` — interactive elements, map overlays, active states. Hover: `#60a5fa`. Glow: `rgba(59,130,246,0.30)`.

**Traffic status colors** (unchanged semantic meaning, added glow + border variants):
- Free flow (MOE 0 / 暢通): `#22c55e` — border `rgba(34,197,94,0.25)`, glow `rgba(34,197,94,0.20)`
- Moderate (MOE 1 / 普通): `#f59e0b` — border `rgba(245,158,11,0.25)`, glow `rgba(245,158,11,0.20)`
- Congested (MOE 2 / 壅塞): `#ef4444` — border `rgba(239,68,68,0.25)`, glow `rgba(239,68,68,0.20)`

**Prediction / ML color** (new in v2): Violet `#8b5cf6` — used for GRU prediction overlays and forecast badges.

**Text** (primary brightened for WCAG AA compliance):
- Primary: `#f1f5f9` (was `#e2e8f0`, +contrast)
- Secondary: `#94a3b8`
- Muted: `#4e6080`
- Disabled: `#2d3d55`

### Typography

Upgraded from `Space Grotesk` → **`Inter`** for display/headings.

Rationale: Inter has superior tabular numeral rendering, tighter Latin/CJK co-existence with Noto Sans TC, and is the industry standard for professional data dashboards (fintech, operations, analytics).

| Role | Font | Usage |
|---|---|---|
| **Display / Headings** | `Inter` 600–700 | Section titles, KPI metric labels |
| **Metrics (large numbers)** | `Inter` 700, `tabular-nums`, `tracking-tightest` | KPI card values (48px) |
| **Body / UI** | `Noto Sans TC` 400–500 | All Chinese text, table content, labels |
| **Mono / Data** | `JetBrains Mono` 400–500, `tabular-nums` | Road IDs, timestamps, raw values |

Type scale: 11 / 12 / 13 / 14 / 16 / 20 / 24 / 32 / 48px.

All numeric data columns use `font-variant-numeric: tabular-nums` to prevent layout shift during updates.

### Spacing
4px base grid. Tokens: 2, 4, 6, 8, 12, 16, 20, 24, 32, 40, 48, 64, 80, 96px.

### Elevation System (v2 — Glass Border + Ambient)

v1 used border-only separation with no shadows. v2 adds a subtle 3-layer elevation system:

| Level | Token | Shadow |
|---|---|---|
| Card | `--shadow-card` | 1px glass border + 1px ambient lift |
| Elevated | `--shadow-elevated` | border + 8px ambient + 2px inner lift |
| Modal | `--shadow-modal` | border-strong + 24px deep shadow |
| Focus | `--shadow-focus` | 2px base ring + 4px blue ring |
| Status glow | `--shadow-glow-*` | 12px status-colour radial glow |

Glassmorphism applies to panels using `backdrop-filter: blur(12px)` + `--color-bg-glass`.

### Animation (v2 — Expressive but purposeful)

All animations follow the ui-ux-pro-max motion rules: 150–300ms, ease-out for entry, ease-in for exit, exit ~70% of enter duration.

| Token | Value | Use |
|---|---|---|
| `--duration-fast` | 150ms | Hover states, colour changes |
| `--duration-base` | 250ms | Toggles, panel opens |
| `--duration-enter` | 300ms | Modal / drawer open |
| `--duration-exit` | 200ms | Modal / drawer close |
| `--ease-out` | `cubic-bezier(0,0,0.2,1)` | All entering elements |
| `--ease-spring` | `cubic-bezier(0.16,1,0.3,1)` | Expressive reveals |

Status changes: colour cross-fade `150ms ease-out` + glow fade.
`prefers-reduced-motion: reduce` disables all transitions/animations.

### Hover & Press States
- Interactive rows: `rgba(59,130,246,0.08)` background on hover
- Buttons: scale `0.97` on `:active`; transition 75ms
- Focus: 2px base ring + 4px blue ring (WCAG 2.1 §2.4.11)
- Nav active: left 3px bar indicator with blue glow

### Cards & Containers
- Card radius: `12px` (was 8px — softer, more modern)
- Button radius: `6px`
- Badge radius: `4px`
- Input/tooltip radius: `6px` / `4px`
- Card padding: `16px` standard, `12px` compact

### Borders (alpha-based in v2)
- Default: `1px solid rgba(99,130,200,0.15)` — near invisible
- Card edge: `1px solid rgba(99,130,200,0.25)`
- Focus: `1px solid #3b82f6` + shadow ring
- Row divider: `1px solid rgba(99,130,200,0.08)`

### Transparency & Blur
- Modal/glass panels: `backdrop-filter: blur(12px)` + `--color-bg-glass`
- Overlay scrim: `rgba(2,6,23,0.85)` — 85% opacity for legibility
- Sidebar: solid `--color-bg-surface` — no blur (stability)
- Tooltip: `--color-bg-elevated` near-opaque

### Layout (tokens added in v2)
- Sidebar: `220px` fixed / `56px` collapsed (was 240px)
- Top bar: `52px` fixed (was 56px — tighter for density)
- Content max-width: `1440px`
- Panel width: `360px` (slide-over panels)

---

## ICONOGRAPHY

Use **Lucide Icons** via CDN. All icons: SVG, `1.5px` stroke, `currentColor`. Never use emoji as icons.

| Icon | Usage |
|---|---|
| `traffic-cone` / `map-pin` | Road segment markers |
| `activity` | Traffic flow charts |
| `zap` | Signal optimisation |
| `clock` | Timing / simulation time |
| `message-square` | LLM assistant |
| `refresh-cw` | Data refresh |
| `download` | Export |
| `alert-triangle` | Congestion warning |
| `check-circle` | Free-flow confirmation |
| `bar-chart-2` | Metrics panel |
| `brain-circuit` | ML prediction |
| `wifi` | Real-time connection status |

Sizes: 16×16px (compact), 20×20px (default), 24×24px (empty states).

---

## VISUAL FOUNDATIONS SUMMARY — v2.0

| Property | v1 | v2 |
|---|---|---|
| Background base | `#0a0e1a` | `#020617` (OLED deeper) |
| Surface | `#111827` | `#0a0f1e` |
| Card | `#1a2236` | `#0f1629` |
| Border | `#2a3555` (opaque) | `rgba(99,130,200,0.15)` (alpha) |
| Accent blue | `#3b82f6` | `#3b82f6` + glow token |
| Free flow | `#22c55e` | `#22c55e` + border + glow |
| Moderate | `#f59e0b` | `#f59e0b` + border + glow |
| Congested | `#ef4444` | `#ef4444` + border + glow |
| Prediction | — | `#8b5cf6` (new) |
| Text primary | `#e2e8f0` | `#f1f5f9` (brighter) |
| Text secondary | `#94a3b8` | `#94a3b8` |
| Text muted | `#475569` | `#4e6080` |
| Radius card | `8px` | `12px` |
| Radius button | `6px` | `6px` |
| Font display | Space Grotesk | **Inter** |
| Font body | Noto Sans TC | Noto Sans TC |
| Font mono | JetBrains Mono | JetBrains Mono |
| Elevation | border-only | glass border + ambient shadow |
| Animation easing | `ease` | `cubic-bezier(0,0,0.2,1)` |
| Transition fast | `150ms ease-out` | `150ms cubic-bezier(0,0,0.2,1)` |

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
