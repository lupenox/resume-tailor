# Interface design

Resume Tailor uses a dark “Study in the Woods” visual system: quiet, scholarly,
and grounded. The interface combines deep warm charcoal-green foundations,
softly elevated panels, parchment-toned text, muted lichen accents, and restrained
brass highlights that suggest lamplight under a canopy.

The runtime UI uses no remote fonts or decorative assets. Body text and activity
logs use the system sans-serif stack; Georgia is reserved for selected headings.
Subtle gradients and texture are pure CSS; the mark is a locally authored SVG.

Core tokens:

| Role | Value |
| --- | --- |
| Canvas | `#0c100d` |
| Strong panel | `#1a221c` |
| Primary text | `#e8e4d9` |
| Secondary text | `#b5b3a4` |
| Muted text | `#8a887a` |
| Lichen accent | `#a8b48a` |
| Brass highlight | `#c4a574` |
| Forest steel | `#3d4a3c` |
| Focus | `#c4a574` |

This direction evolved from an earlier moonlit system and a pure forest pass.
It preserves layout, spacing, accessibility, and the safety-critical interaction
model while shifting the mood toward a calm, precise workspace.

Accessibility checks covered contrast, visible focus, touch targets, semantic
headings, status announcements, reduced motion, forced colors, text selection,
keyboard navigation, 200% reflow, and horizontal overflow at desktop and mobile
widths.
