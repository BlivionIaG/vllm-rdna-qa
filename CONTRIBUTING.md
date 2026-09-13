# Contributing

This repo is **agent skills**, not a playbook. Each skill is one folder
and one short `SKILL.md`. Dest code lands on
[`opengfx1030/vllm-rdna`](https://github.com/opengfx1030/vllm-rdna)
`rdna_extras`. Tiles land in
[`BlivionIaG/hippihx`](https://github.com/BlivionIaG/hippihx). Silicon
notes stay in
[`BlivionIaG/rdna-hip-wiki`](https://github.com/BlivionIaG/rdna-hip-wiki).

## Add a skill

1. Create `skills/<id>/` — kebab-case, `rdna-` prefix for dest gates.
2. Add `SKILL.md` with YAML frontmatter:

   ```yaml
   ---
   name: <id>
   description: use this when <the task that should load this skill>.
   ---
   ```

   `description` is **one line** and **starts with** `use this when`.
3. Body: a short recipe. Hard rules, a table if needed, links out.
   Point at the wiki or `arch-research/…/BRIEF.md`. Do not paste wikis
   or papers.
4. List the skill and its description in [README.md](README.md).
5. Keep it short. If it needs a chapter, it is the wrong shape.

Do not add `docs/00-…` chapter stubs. Do not treat tok/s as dest.
Never collapse the three residuals. Never land TP≤2 `can_implement` or
Triton as dest.

## Attribution

Dest extras HIP is **BlivionIaG**. Foreign HIP keeps the source Author
and Committer. Skills do not re-author kernels.
