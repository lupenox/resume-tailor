# Interface design

Resume Tailor uses a dark, moonlit visual system intended to feel distinctive
without obscuring a safety-critical workflow. The interface combines near-black
and midnight-navy foundations, translucent blue panels, steel borders, icy-cyan
active states, and restrained purple atmosphere.

The runtime UI uses no remote fonts or decorative assets. Body text and activity
logs use the system sans-serif stack; Georgia is reserved for selected headings.
Moonlight and mist are CSS gradients, and the crescent/wolf mark is a locally
authored SVG.

Core tokens:

| Role | Value |
| --- | --- |
| Canvas | `#050914` |
| Strong panel | `#0c182b` |
| Primary text | `#f2f6fb` |
| Secondary text | `#b4c7d9` |
| Muted text | `#8599b2` |
| Icy cyan | `#96d9ef` |
| Steel blue | `#385b82` |
| Purple accent | `#8063ad` |
| Focus | `#a9e5f5` |

The redesign was directed and implemented with Codex as implementation owner
and Antigravity as a read-only visual critic. Three bounded review rounds covered
design direction, implementation scoring, and final high-impact critique. Only
synthetic states entered that loop; raw agent transcripts and hidden reasoning
are not retained.

Accessibility checks covered contrast, visible focus, touch targets, semantic
headings, status announcements, reduced motion, forced colors, text selection,
keyboard navigation, 200% reflow, and horizontal overflow at desktop and mobile
widths.
