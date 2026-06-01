# Dennis's Personal Knowledge Base

A persistent, compounding knowledge base maintained by LLM. The human curates sources, directs analysis, and asks questions. The LLM does the summarizing, cross-referencing, filing, and maintenance.

## Architecture

```
raw/          # Immutable source documents. Never modify these.
raw/assets/   # Downloaded images and attachments
wiki/         # LLM-generated and maintained markdown pages
wiki/index.md # Content catalog — read this first when answering queries
wiki/log.md   # Chronological record of all operations
```

## Source Types

Raw sources may be:
- **Web articles** — markdown files clipped from the web (via Obsidian Web Clipper or manual copy)
- **Documents** — PDFs, papers, reports, notes
- **Images** — screenshots, diagrams, photos (stored in `raw/assets/`)
- **Tables/Data** — CSV, JSON, or markdown tables
- **Links** — a markdown file containing a URL and minimal metadata when the full content isn't saved locally

## Naming Convention

- Raw sources: `raw/YYYY-MM-DD_short-descriptive-slug.md` (or appropriate extension)
- Raw assets: `raw/assets/` with descriptive filenames
- Wiki pages: `wiki/` with kebab-case names, e.g. `wiki/machine-learning.md`, `wiki/book-atomic-habits.md`

## Wiki Page Format

Every wiki page should have YAML frontmatter:

```yaml
---
title: Page Title
type: entity | concept | source-summary | comparison | synthesis | question
tags: [tag1, tag2]
sources: [source-filename-1, source-filename-2]
created: YYYY-MM-DD
updated: YYYY-MM-DD
---
```

Page types:
- **entity** — a person, place, organization, tool, book, etc.
- **concept** — an idea, framework, methodology, principle
- **source-summary** — summary of a single raw source
- **comparison** — side-by-side analysis of multiple things
- **synthesis** — higher-order page connecting multiple concepts/entities
- **question** — an answered question that's worth preserving

Use `[[wiki-links]]` for cross-references between wiki pages.

## Operations

### Ingest

When the user adds a new source to `raw/`:

1. Read the source thoroughly
2. Discuss key takeaways with the user
3. Create a source-summary page in `wiki/`
4. Update `wiki/index.md` with the new page
5. Create or update relevant entity/concept pages across the wiki
6. Add cross-references (`[[links]]`) between related pages
7. Append an entry to `wiki/log.md`

A single source may touch 5-15 wiki pages. Always keep the index and log current.

### Query

When the user asks a question:

1. Read `wiki/index.md` to find relevant pages
2. Read the relevant wiki pages
3. Synthesize an answer with citations to wiki pages and raw sources
4. If the answer is substantive and reusable, offer to file it as a new wiki page (type: question or synthesis)

### Lint

When asked to health-check the wiki:

- Flag contradictions between pages
- Identify stale claims superseded by newer sources
- Find orphan pages with no inbound links
- Note important concepts mentioned but lacking their own page
- Suggest missing cross-references
- Recommend new sources or questions to investigate

## Guidelines

- The wiki is the LLM's responsibility. Write clearly, concisely, and maintain it rigorously.
- Always update cross-references when adding or modifying pages.
- When new information contradicts existing wiki content, flag the contradiction explicitly and update the affected pages.
- Prefer updating existing pages over creating new ones when the topic already exists.
- Keep `wiki/index.md` as the single navigational entry point — it should be browsable and well-organized.
- The user reads the wiki in Obsidian. Use standard markdown and wikilinks for best compatibility.
