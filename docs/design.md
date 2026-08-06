# Interface design

Resume Tailor uses a dark, forest-inspired visual system intended to feel
distinctive and grounded without obscuring a safety-critical workflow. The
interface combines deep moss and pine foundations, translucent green panels,
soft olive borders, sage active states, and restrained warm-amber atmosphere
that evokes filtered canopy light.

The runtime UI uses no remote fonts or decorative assets. Body text and activity
logs use the system sans-serif stack; Georgia is reserved for selected headings.
Canopy mist and soft leaf gradients are pure CSS; the mark is a locally
authored SVG.

Core tokens:

| Role | Value |
| --- | --- |
| Canvas | `#0a120c` |
| Strong panel | `#13241a` |
| Primary text | `#eef4ea` |
| Secondary text | `#b8c9b0` |
| Muted text | `#7a8f75` |
| Sage accent | `#8fbc8f` |
| Forest steel | `#3d5c45` |
| Warm highlight | `#d4b86a` |
| Focus | `#a8d4a0` |

The original moonlit redesign was directed and implemented with Codex as
implementation owner and Antigravity as a read-only visual critic. The forest
theme is a subsequent visual evolution that preserves the same layout, spacing,
accessibility, and interaction model.

Accessibility checks covered contrast, visible focus, touch targets, semantic
headings, status announcements, reduced motion, forced colors, text selection,
keyboard navigation, 200% reflow, and horizontal overflow at desktop and mobile
widths.
