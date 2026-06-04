export const meta = {
  name: 'review-example',
  description: 'Run physics and code reviews in parallel for a qdot example script, then synthesise findings',
  phases: [
    { title: 'Review', detail: 'Physics and code reviewers run in parallel' },
    { title: 'Synthesise', detail: 'Combine and rank findings' },
  ],
}

// args: filename string, e.g. "efg_species_comparison.py"
const filename = args

phase('Review')
const [physicsResult, codeResult] = await parallel([
  () => agent(
    `Review examples/${filename} for physical correctness. Read the script, run it to produce output, then assess all seven checklist points in your system prompt.`,
    { label: 'physics', agentType: 'physics-reviewer' }
  ),
  () => agent(
    `Review examples/${filename} for code quality. Read the script, run it to check for errors, then assess all eight checklist points in your system prompt.`,
    { label: 'code', agentType: 'code-reviewer' }
  ),
])

phase('Synthesise')
const synthesis = await agent(
  `Two independent reviews of examples/${filename} are below. Synthesise them into a single actionable report.

## Physics review
${physicsResult}

## Code review
${codeResult}

Structure the output as:
- Overall verdict: PASS / FAIL / WARNING (fail if either reviewer fails; warning if either warns)
- FAIL items (any, from either reviewer) — listed first with reviewer tag [physics] or [code]
- WARNING items — same
- What passed cleanly

Where a physics issue and a code issue concern the same line or value, group them. Keep it concise — the reader wants to know what to fix, in priority order.`,
  { label: 'synthesise' }
)

return synthesis
