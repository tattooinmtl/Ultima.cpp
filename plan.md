# Current task plan

## Coder IDE — Decision 18

**Requested:** A coding IDE in the testpad: multi-file tabbed editor, full file
operations (read/write/rename/edit/delete), and a code session that builds
whole apps by working in phases.
**Started:** 2026-09-16
**Design:** roadmap01.md Decision 18. Builds out Decisions 16 + 17, which were
designed but never implemented.

One phase is one task. A phase is done when it produces something the user can
look at. Subtasks for a phase live in `.ultima/todos.json`, not here.

- [x] Phase 1: Workspace file API — `/api/fs/*` behind one path-resolution
      guard. Provable by curl: create, read, edit, rename, delete a file, and
      get a 400 for every escape attempt.
      **Done 2026-09-16.** 22 unit tests + verified live over HTTP: all seven
      endpoints work; ../, encoded ../, backslash ../, drive letters, UNC,
      /etc/passwd and NUL all refused 400 with a named reason; nothing written
      outside the root.
- [x] Phase 2: File tool layer — the same six operations declared to the model,
      plus the tool-call loop. Provable by asking a loaded model to create a
      file and seeing it appear on disk.
      **Done 2026-09-16.** Qwen2.5-Coder-7B creates a file on disk through
      /api/agent/chat, reproducibly. 37 unit tests. Live runs drove three
      robustness fixes (see Notes). Known limitation: the 7B stops after a
      failed tool call instead of retrying — Phase 4 is the structural answer.
- [x] Phase 3: Coder tab — file tree, editor tabs, syntax highlighting, save.
      Provable by opening the testpad and editing a file end to end.
      **Done 2026-09-17.** Verified in a browser: tree lists the workspace,
      clicking a file opens a tab, editing marks it dirty, Ctrl+S writes to
      disk, and an agent run creates a file that appears as a focused tab.
      Highlighter has its own regression test (scripts/test_testpad_highlight.mjs).
- [x] Phase 4: Phased planning — `.ultima/plan.md` + `todos.json`, the agent
      contract, and the Plan panel that renders phases with nested subtasks.
      Provable by giving the agent a project-sized ask and getting a plan back
      with no code.
      **Done 2026-09-17.** Asked the 7B for a coffee-shop landing page in plan
      mode: it wrote .ultima/plan.md with four phases and nothing else - the
      workspace still held only that file and the pre-existing README. The Plan
      panel renders it, checking a phase off advances the active one and appends
      to .ultima/CHANGELOG.md. 210 unit tests.
- [x] Phase 5: Permissions — three modes (auto / ask / yolo), an always-ask set
      (destructive, escaping the project, plan edits, installs, execution), a
      reason on every prompt, and reads never gated. See PERMISSIONS.md.
      Provable by: a delete asks even in auto mode, an ordinary write does not,
      denial reaches the model as a tool result, and yolo skips both.
      **Done 2026-09-18.** All four verified live against the 7B. 230 unit
      tests. The live run caught a gate bypass - see Notes.
- [ ] Phase 6: Full-stack session — project scaffolding templates, the sandbox
      run tools from Decision 16 §16.3, and a live preview pane. Provable by
      "build me a todo app" producing a running page.

## Notes

- The editor is written in-tree. `ui/testpad/index.html` is bundled into the
  binary at configure time and references nothing external, so Monaco and every
  other CDN editor is out. See Decision 18 §18.1.
- Reference studied: `C:\GameForgerAI` — `server/controllers/gameSimController.js`
  and `client/src/pages/GameSimCreator.jsx`. Its `<!-- FILE: x -->` marker
  format and one-tab-per-file UI carry over; its whole-file regeneration model
  does not, because it cannot express rename or partial edit.
- Phase 1 is the security-sensitive one. Every path in the whole feature funnels
  through a single `resolve_in_workspace()`; that function is where the tests go.
- Phase 2, learned from live runs rather than guessed. Qwen2.5-Coder-7B called
  `write_project-file` with a hyphen and passed `/hello.js` with a leading
  slash; later it invented `read_project`. So the tool layer now canonicalizes
  names (separator/case drift, aliases, and a leading-verb fallback), strips a
  leading slash from paths, and makes every error actionable by naming the
  retry. None of this loosens `resolve()` — the normalization happens above it.
- Phase 3 editor is a textarea with transparent text over a highlighted
  <pre>. The two must keep identical font metrics and padding or the caret
  drifts from the glyphs; both rules carry a comment saying so. The
  highlighter is one combined regex per language, and `\x60` is how a
  backtick gets into a String.raw pattern.
- Phase 4 found the cause of the empty turns seen in Phase 2. A round that
  produced neither prose nor a tool call now emits a `warning` event carrying
  the raw head of the generation, and that showed the 7B writing typographic
  quotes - the JSON was unparseable, so the whole call was silently dropped.
  Parsing now retries with the quotes normalized, but only after a strict parse
  has already failed, so content that legitimately contains curly quotes is
  untouched.
- Phase 5 caught a real gate bypass, and only because it was tested live. The
  model called `delete_project`; the dispatcher resolves that to `delete_file`
  by its leading verb, but `classify()` was comparing the RAW name, saw an
  ordinary write, and deleted the file with no prompt. Anything that reasons
  about a call must judge the tool that will actually run. The same class of
  mismatch bit twice more in one session: the permission layer normalized a
  leading slash before the dispatcher did, and accepted `file_path` when the
  dispatcher did not. Rule of thumb: the gate and the dispatcher must agree
  about every argument, or the gate is decoration.
- Open, and the honest state of Phase 2: the loop, parsing, dispatch and
  events are solid, but a 7B given a two-step task ("read it, then edit it")
  frequently stops after one failed call rather than recovering. That is a
  model-capability limit, and the reason Phase 4 exists: one phase at a time,
  each small enough to be a single confident step.
