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
- [ ] Phase 2: File tool layer — the same six operations declared to the model,
      plus the tool-call loop. Provable by asking a loaded model to create a
      file and seeing it appear on disk.
- [ ] Phase 3: Coder tab — file tree, editor tabs, syntax highlighting, save.
      Provable by opening the testpad and editing a file end to end.
- [ ] Phase 4: Phased planning — `.ultima/plan.md` + `todos.json`, the agent
      contract, and the Plan panel that renders phases with nested subtasks.
      Provable by giving the agent a project-sized ask and getting a plan back
      with no code.
- [ ] Phase 5: Full-stack session — project scaffolding templates, the sandbox
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
