# Knowledge corpus

The Markdown cards under `.claude/knowledge/` are optional technical
references. Search them directly when useful:

```bash
rg -n -i "query terms" .claude/knowledge
```

The scripts under `scripts/kb/` can provide ranked search results, but no hook
runs them automatically and consulting the corpus is never a prerequisite for
editing or testing code.

The SOL and campaign cards remain available through the explicit
`.claude/skills/run-sol-campaign/SKILL.md` entry point. That skill routes to the
full retained stack without making knowledge lookup or legacy artifact checks
mandatory for unrelated work.
