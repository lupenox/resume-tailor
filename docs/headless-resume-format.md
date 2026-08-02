# Deterministic Headless résumé format

New Qwen runs render the approved structured content into a deterministic
single-column résumé influenced by the supplied Headless template and résumé
guide. The reference documents inform layout and writing constraints; they are
not copied into the repository or sent to Qwen at runtime.

## Layout contract

- US Letter, one section, one column, and no tables.
- Arial throughout; 14-point bold centered name, 9-point centered contact line,
  and 10.5-point section/body text.
- Black body text with restrained blue underlined contact hyperlinks.
- Half-inch top/bottom margins and three-quarter-inch left/right margins.
- Compact, readable spacing with section divider rules.
- Real Word list paragraphs for bullets.
- Reverse-chronological work-first ordering, followed by projects and the
  open-source entry.
- Italic entry headings, with experience dates on a right-aligned tab stop.

The four rendered sections are `SUMMARY`, `EDUCATION & CERTIFICATES`,
`TECHNICAL SKILLS`, and `WORK HISTORY & PROJECTS`. Every field in the canonical
Resume Tailor content object remains present: summary; degree, coursework, and
certifications; three skill groups; experience; three projects; and the
open-source contribution.

## Writing guidance supplied to Qwen

- Apply only authenticated approved edits and preserve every other value.
- Make the first bullet for an entry understandable to a nontechnical reader.
- Prefer a supported WHAT + HOW + RESULT/REASON structure, but omit a result
  when the master provides no evidence for one.
- Use concise past-tense language for completed work, no first-person pronouns,
  and no more than one terminal period per bullet.
- Place supported role keywords early when natural; never force an unsupported
  keyword into a claim.
- Never invent metrics, credentials, technologies, leadership, citizenship,
  availability, dates, employers, project identity, or customer impact.

These are authoring preferences inside the stronger local evidence boundary.
They never authorize facts and never override content budgets.

## Deterministic validation

After saving, Python reopens the DOCX and verifies:

- exactly one section and zero tables;
- the original name and complete contact line;
- every original contact hyperlink text and target in order;
- all four required section headings;
- every approved canonical content string;
- Arial on every direct text run;
- an unchanged source-resume SHA-256.

LibreOffice and Poppler then require exactly one Letter PDF page, extractable
required text, no replacement glyphs, in-page non-overlapping text bounding
boxes, and a nonempty PNG preview. Failure preserves the generation-specific
artifacts and prevents stable publication.

Selecting `--writer-provider antigravity` uses the historical master-template
copy-and-replace renderer instead. Resume Tailor never silently switches layout
or writer providers during a run.
