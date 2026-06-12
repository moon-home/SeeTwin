# SeeTwin — UI Style Guide

## Design intent

Surgical tool, not a creative playground. The user is doing precise technical work
(mask painting, classifier confirmation, texture review, poly budget checks). The UI
stays out of the way: neutral surfaces, one strong accent for navigation and primary
actions, semantic color only for status feedback.

---

## Color tokens

| Role | Hex | CSS var fallback | Usage |
|------|-----|-----------------|-------|
| Brand accent | `#534AB7` | — | Active stage, primary buttons, active tab underline, slider |
| Brand hover | `#3C3489` | — | Hover on primary elements |
| Brand light fill | `#EEEDFE` | — | Subtle active background tint |
| Success | `#1D9E75` | — | Stage complete, foreground brush, live mocap indicator |
| Success light | `#9FE1CB` | — | Upload slot border (filled state) |
| Warning | `#EF9F27` | — | Low-confidence classification, atlas coverage warnings |
| Warning light | `#FAEEDA` | — | Warning pill background |
| Error / remove | `#E24B4A` | — | Background brush mode, quality check failures |
| Error light | `#FCEBEB` | — | Error pill background |
| Surface primary | — | `var(--color-background-primary)` | Canvas, main panel |
| Surface secondary | — | `var(--color-background-secondary)` | Sidebar, status bar, tool column |
| Border default | — | `var(--color-border-tertiary)` 0.5px | All panel dividers |
| Border emphasis | — | `var(--color-border-secondary)` 0.5px | Hover borders, button borders |
| Text primary | — | `var(--color-text-primary)` | Stage titles, confirmed labels |
| Text secondary | — | `var(--color-text-secondary)` | Stage names, descriptions |
| Text tertiary | — | `var(--color-text-tertiary)` | Hints, metadata, sub-labels |

**Rule:** No new hex colors. All colors trace to this table.

---

## Typography

Single typeface: Anthropic Sans via `var(--font-sans)`.

| Role | Size | Weight | Usage |
|------|------|--------|-------|
| Stage title | 14px | 500 | Topbar |
| Stage label | 12px | 400 | Sidebar stage names |
| Tool section header | 10px | 500 | Uppercase, 0.5px letter-spacing |
| Body / meta | 11–12px | 400 | Descriptions, hints, status |
| Canvas label | 10px | 400 | Below previews |

Two weights only: 400 regular, 500 medium. Never 700.
Sentence case everywhere except tool section headers (uppercase).

---

## Layout system

```
┌──────────────────────────────────────────────────────┐
│  Topbar: stage title · meta · secondary CTA · primary CTA  │
├──────────────────────────────────────────────────────┤
│  Tab bar (Original · Mask · Preview · All angles)          │
├────────────┬──────────────────────────┬──────────────┤
│  Input col │       Canvas / main      │  Tool col    │
│  (220px)   │       (flex 1)           │  (180px)     │
├──────────────────────────────────────────────────────┤
│  Status bar: model · session · warnings · readiness        │
└──────────────────────────────────────────────────────┘
```

Stages without a three-column layout (Stage 6 Blender checklist, Stage 8 webcam,
Stage 9 pose analysis) collapse the tool column and expand the canvas region.

---

## Components

### Sidebar stage item
- Height: ~34px, padding: 7px 16px
- Active: `border-left: 2px solid #534AB7` + surface-primary background
- Done: step bubble → `#1D9E75` fill, white check
- Pending: step bubble → `border-tertiary` fill, tertiary text number
- Manual step (Stage 6): step bubble → amber fill

### Primary button
- Background: `#534AB7`, white text, border-radius: `var(--border-radius-md)`
- Hover: `#3C3489`
- Padding: 5px 12px, font: 12px 400

### Secondary button
- Background: surface-primary, `border-secondary` 0.5px
- Hover: surface-secondary

### Confirmation card (Stage 3 specific)
- Full-width card in main canvas area
- Shows: part thumbnail + classifier result + confidence bar + top 3 alternatives
- Actions: "Confirm" (primary) + "Choose alternative" (secondary)
- Confidence bar: green > 85%, amber 60–85%, red < 60%

### Quality warning card (Stage 4 specific)
- Amber border-left accent
- Shows: check name + measured value + threshold + suggestion
- User must dismiss each warning before proceeding

### Upload slot
- Empty: dashed `border-secondary`, surface-secondary background
- Hover: dashed `#7F77DD`
- Filled: solid `#9FE1CB`

### Status pill
- Success: `#EAF3DE` bg, `#3B6D11` text
- Warning: `#FAEEDA` bg, `#854F0B` text
- Error: `#FCEBEB` bg, `#A32D2D` text
- Border-radius: 20px

### Brush mode buttons
- Foreground active: `#EAF3DE` bg, `#3B6D11` text, `#97C459` border
- Background active: `#FCEBEB` bg, `#A32D2D` text, `#E24B4A` border

---

## Canvas area

- Background: checkerboard (`repeating-conic-gradient`) = transparency indicator
- Brush cursor: 24px circle, rgba green fill + `#1D9E75` border (fg) / red (bg)
- Paint strokes on canvas: 18% opacity fill matching active brush mode color

---

## Stage-specific accent usage

| Stage | Accent color | Rationale |
|-------|-------------|-----------|
| 1 BG removal | Green / Red for brush modes | Universal mask editing convention |
| 2 Body shape | Blue progress ring | Processing / information |
| 3 Classification | Amber confidence bar | Caution-scaled feedback |
| 4 Texture | Amber / Red quality warnings | Warning → error severity scale |
| 5 Assembly | Teal poly budget bar | Progress bar convention |
| 6 Blender | Gray checklist items → green when checked | Neutral manual step |
| 7 Rigging | Blue external link | Information |
| 8 Mocap | Green live dot | Live status universal convention |
| 9 Pose analysis | Green / Amber / Red per-joint overlay | Performance feedback scale |

---

## Spacing rhythm

| Scale | Value | Usage |
|-------|-------|-------|
| xs | 4px | Tightly related elements |
| sm | 8px | Button padding, icon gap |
| md | 12px | Default section gap |
| lg | 16px | Column padding |
| xl | 20px | Panel separation, major headers |

---

## Iconography

Tabler icons, outline only. 16px inline with text, 18px standalone.

| Icon | Stage / usage |
|------|--------------|
| `ti-check` | Completed stage bubble |
| `ti-upload` | Upload actions |
| `ti-download` | Export / download |
| `ti-refresh` | Re-run model |
| `ti-player-play` | Start capture / analysis |
| `ti-camera` | Webcam stage |
| `ti-adjustments` | Settings / parameters |
| `ti-layers` | 3D / mesh related |
| `ti-alert-triangle` | Quality warnings |
| `ti-pencil` | Edit / correct classification |
