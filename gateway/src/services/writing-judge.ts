/**
 * AuthorClaw Writing Judge
 *
 * Modify-evaluate-retry loop for chapter writing, ported from AutoNovel's
 * dual-immune-system pattern. Every chapter draft passes through:
 *
 *   1. Mechanical screen — regex-based checks for cliches, AI-tell words,
 *      adverb density, filter words, passive voice, weak verbs. Cheap,
 *      deterministic, runs locally.
 *
 *   2. LLM judge — single AI call that scores 6 craft dimensions (voice,
 *      show-vs-tell, pacing, dialogue, sensory, emotional truth) on 1-10
 *      and surfaces 1-3 specific issues per dimension.
 *
 *   3. Combined score with weighting — overall 1-10. Below threshold
 *      triggers a retry with the judge's feedback as steering input.
 *
 * Why both layers?
 *   - Mechanical screens catch formulaic patterns LLMs miss (or generate
 *     themselves — "delve", "tapestry", "testament" all rate fine in
 *     LLM judges because LLMs wrote them in the first place)
 *   - LLM judges catch coherence and voice issues mechanical screens
 *     can't see (a chapter with great word stats but a flat emotional arc)
 *
 * Cost discipline:
 *   - Mechanical screen is free (no AI call)
 *   - LLM judge: 1 call per chapter
 *   - Retry: 1 additional draft call + 1 additional judge call
 *   - Default cap: 1 retry, so each chapter costs at most 3 AI calls
 *     (1 draft + 1 judge + 1 retry-with-feedback).
 *   - User can disable retries via project config to keep cost = 2 AI calls.
 */

// ═══════════════════════════════════════════════════════════
// Types
// ═══════════════════════════════════════════════════════════

export interface MechanicalIssue {
  category: 'cliche' | 'ai_tell' | 'filter_word' | 'adverb_density' |
            'passive_voice' | 'weak_verb' | 'banned_phrase' | 'hedge_word' |
            'started_to' | 'suddenly';
  severity: 'info' | 'warning' | 'error';
  description: string;
  examples: string[];
  count: number;
}

export interface MechanicalReport {
  wordCount: number;
  issues: MechanicalIssue[];
  /** Composite mechanical score 0-100. 100 = clean, 0 = riddled. */
  score: number;
}

export interface JudgeDimension {
  name: string;
  score: number;          // 1-10
  issues: string[];       // 1-3 specific problems
}

export interface JudgeReport {
  dimensions: JudgeDimension[];
  /** Average of dimension scores 1-10. */
  overall: number;
  /** Top 3 most actionable issues across all dimensions. */
  topIssues: string[];
}

export interface QualityVerdict {
  /** Combined score 0-100. */
  score: number;
  /** True if the chapter should be retried. */
  retry: boolean;
  /** The mechanical screen result. */
  mechanical: MechanicalReport;
  /** The LLM judge result (null if judge wasn't run). */
  judge: JudgeReport | null;
  /** Human-readable summary for logs. */
  summary: string;
  /** Steering text to pass to the AI on retry. */
  retryFeedback: string;
}

export type AICompleteFn = (req: {
  provider: string;
  system: string;
  messages: Array<{ role: 'user' | 'assistant'; content: string }>;
  maxTokens?: number;
  temperature?: number;
}) => Promise<{ text: string }>;

export type AISelectProviderFn = (taskType: string) => { id: string };

// ═══════════════════════════════════════════════════════════
// Mechanical screen lexicons
// ═══════════════════════════════════════════════════════════

// AI-tell words — phrases that flag LLM-generated prose to readers.
const AI_TELL_PATTERNS = [
  /\bdelve(s|d|ing)? into\b/gi,
  /\btapestry of\b/gi,
  /\btestament to\b/gi,
  /\bin the realm of\b/gi,
  /\bnavigate(s|d|ing)? (?:the|this|these)\b/gi,
  /\bmultifaceted\b/gi,
  /\bmyriad of\b/gi,
  /\bnuanced\b/gi,
  /\bvisceral\b/gi,
  /\bresonate(s|d)? (?:with|deeply)\b/gi,
  /\bparadigm\b/gi,
  /\bbeacon of\b/gi,
  /\bin (?:today|this)['\s]+\w+\s+(?:landscape|world)\b/gi,
  /\bit['\s]s worth (?:noting|mentioning) that\b/gi,
  /\bunderscores? (?:the|that)\b/gi,
  /\bjourney of self.discovery\b/gi,
];

// Banned cliches — phrases that signal lazy prose.
const BANNED_PHRASES = [
  /\bat the end of the day\b/gi,
  /\btip of the iceberg\b/gi,
  /\bin the blink of an eye\b/gi,
  /\bavoid like the plague\b/gi,
  /\bbutterflies in (?:her|his|their|my) stomach\b/gi,
  /\bheart skipped a beat\b/gi,
  /\bno stone unturned\b/gi,
  /\bcalm before the storm\b/gi,
  /\bonly time will tell\b/gi,
  /\bdeafening silence\b/gi,
  /\bfor what (?:felt|seemed) like (?:hours|an eternity)\b/gi,
  /\beyes (?:bored|drilled) into\b/gi,
];

// Filter words — they put a narrator between the reader and the POV character.
const FILTER_WORDS = new Set([
  'saw', 'heard', 'felt', 'smelled', 'tasted', 'noticed', 'watched',
  'thought', 'realized', 'wondered', 'decided', 'knew', 'understood',
  'seemed', 'appeared', 'observed',
]);

// Hedge words — sap urgency when overused.
const HEDGE_WORDS = new Set([
  'perhaps', 'maybe', 'might', 'possibly', 'probably', 'apparently',
  'somewhat', 'rather', 'quite',
]);

// Passive voice: "was/were/is/are/be/been/being + past participle"
const PASSIVE_VOICE_RE = /\b(?:was|were|is|are|be|been|being)\s+\w+ed\b/gi;

// Adverb detection — word ending in "ly" excluding common false positives.
const NON_ADVERB_LY = new Set([
  'family', 'supply', 'only', 'reply', 'apply', 'holy', 'lovely', 'early',
  'ugly', 'silly', 'jolly', 'belly', 'bully', 'really',
]);

// Weak verbs that should usually be replaced with stronger ones.
const WEAK_VERBS_RE = /\b(?:was|were|had been|got|gotten|went|came|put|got)\b/gi;

// "Started to" / "began to" — usually droppable.
const STARTED_TO_RE = /\b(?:started|begun|began) to\s+\w+/gi;

// "Suddenly" — almost always cuttable.
const SUDDENLY_RE = /\bsuddenly\b/gi;

// ═══════════════════════════════════════════════════════════
// LLM judge prompt
// ═══════════════════════════════════════════════════════════

const JUDGE_SYSTEM_PROMPT = `You are a developmental editor evaluating a chapter of fiction. Score the prose on six dimensions, 1-10 each:

1. **voice_consistency** — Is the narrator's voice distinctive and consistent throughout?
2. **show_vs_tell** — Does the chapter show emotion through gesture/action/sensation, or tell it via labels?
3. **pacing** — Does tension rise and fall purposefully? Is there a satisfying arc within the chapter?
4. **dialogue_authenticity** — Do characters sound distinct? Is there subtext, or is dialogue purely informational?
5. **sensory_grounding** — Are scenes anchored in physical detail across multiple senses, or floating in abstraction?
6. **emotional_truth** — Do emotional beats feel earned? Does the reader feel something specific?

For each dimension provide 1-3 SPECIFIC issues (concrete, actionable — never generic). If the dimension is genuinely strong, write a 1-issue note saying so.

Return ONLY valid JSON in this exact format:
{
  "dimensions": [
    {"name": "voice_consistency", "score": 7, "issues": ["The narrator slips into a clinical register in paragraph 5 ('it could be observed that...')"]},
    {"name": "show_vs_tell", "score": 6, "issues": ["Sarah 'felt scared' three times in the opening — replace with physiology", "The grief on page 2 is told ('she was sad') instead of shown"]},
    ...
  ]
}

No commentary outside the JSON. No markdown code fences.`;

// ═══════════════════════════════════════════════════════════
// Service
// ═══════════════════════════════════════════════════════════

export class WritingJudgeService {
  /** Run the deterministic mechanical screen. Cheap; safe to run on every draft. */
  mechanicalScreen(text: string): MechanicalReport {
    const issues: MechanicalIssue[] = [];
    const wordCount = text.split(/\s+/).filter(Boolean).length || 1;
    const wordsLower = text.toLowerCase().split(/\s+/).map(w => w.replace(/[^a-z'\-]/g, ''));

    // ── AI-tell words ──
    {
      let count = 0;
      const examples: string[] = [];
      for (const re of AI_TELL_PATTERNS) {
        const matches = text.match(re);
        if (matches) {
          count += matches.length;
          examples.push(...matches.slice(0, 2));
        }
      }
      if (count > 0) {
        issues.push({
          category: 'ai_tell',
          severity: count > 3 ? 'error' : 'warning',
          description: `${count} AI-tell phrases found. These signal LLM prose to careful readers.`,
          examples: Array.from(new Set(examples)).slice(0, 5),
          count,
        });
      }
    }

    // ── Banned cliches ──
    {
      let count = 0;
      const examples: string[] = [];
      for (const re of BANNED_PHRASES) {
        const matches = text.match(re);
        if (matches) {
          count += matches.length;
          examples.push(...matches.slice(0, 1));
        }
      }
      if (count > 0) {
        issues.push({
          category: 'banned_phrase',
          severity: count > 2 ? 'error' : 'warning',
          description: `${count} cliché phrase${count === 1 ? '' : 's'} that should be replaced.`,
          examples: Array.from(new Set(examples)).slice(0, 5),
          count,
        });
      }
    }

    // ── Filter words ──
    {
      let count = 0;
      for (const w of wordsLower) if (FILTER_WORDS.has(w)) count++;
      const rate = (count / wordCount) * 1000;
      if (rate > 8) {
        issues.push({
          category: 'filter_word',
          severity: rate > 14 ? 'error' : 'warning',
          description: `${count} filter words (${rate.toFixed(1)}/1000) — saw, heard, felt, noticed, realized. They distance the reader from the POV character.`,
          examples: [],
          count,
        });
      }
    }

    // ── Adverb density ──
    {
      let adverbs = 0;
      for (const w of wordsLower) {
        if (w.length > 3 && w.endsWith('ly') && !NON_ADVERB_LY.has(w)) adverbs++;
      }
      const rate = (adverbs / wordCount) * 1000;
      if (rate > 12) {
        issues.push({
          category: 'adverb_density',
          severity: rate > 20 ? 'error' : 'warning',
          description: `${adverbs} -ly adverbs (${rate.toFixed(1)}/1000). Strong verbs beat adverb crutches.`,
          examples: [],
          count: adverbs,
        });
      }
    }

    // ── Passive voice ──
    {
      const matches = text.match(PASSIVE_VOICE_RE) || [];
      const rate = (matches.length / wordCount) * 1000;
      if (rate > 8) {
        issues.push({
          category: 'passive_voice',
          severity: rate > 14 ? 'error' : 'warning',
          description: `${matches.length} passive constructions (${rate.toFixed(1)}/1000). Active voice tightens prose.`,
          examples: matches.slice(0, 3),
          count: matches.length,
        });
      }
    }

    // ── Weak verbs ──
    {
      const matches = text.match(WEAK_VERBS_RE) || [];
      const rate = (matches.length / wordCount) * 1000;
      if (rate > 30) {
        issues.push({
          category: 'weak_verb',
          severity: rate > 50 ? 'warning' : 'info',
          description: `${matches.length} weak verbs (was/were/had been/got/went) per 1000 words: ${rate.toFixed(1)}.`,
          examples: [],
          count: matches.length,
        });
      }
    }

    // ── Started-to / began-to ──
    {
      const matches = text.match(STARTED_TO_RE) || [];
      if (matches.length >= 3) {
        issues.push({
          category: 'started_to',
          severity: matches.length > 6 ? 'warning' : 'info',
          description: `${matches.length} "started to" / "began to" constructions. Usually drop the auxiliary.`,
          examples: matches.slice(0, 3),
          count: matches.length,
        });
      }
    }

    // ── Suddenly ──
    {
      const matches = text.match(SUDDENLY_RE) || [];
      if (matches.length >= 2) {
        issues.push({
          category: 'suddenly',
          severity: matches.length > 4 ? 'warning' : 'info',
          description: `"Suddenly" appears ${matches.length}× — almost always cuttable; the action itself implies the sudden.`,
          examples: [],
          count: matches.length,
        });
      }
    }

    // ── Hedge density ──
    {
      let count = 0;
      for (const w of wordsLower) if (HEDGE_WORDS.has(w)) count++;
      const rate = (count / wordCount) * 1000;
      if (rate > 6) {
        issues.push({
          category: 'hedge_word',
          severity: rate > 12 ? 'warning' : 'info',
          description: `${count} hedge words (${rate.toFixed(1)}/1000) — perhaps, maybe, might, somewhat, rather. Sap urgency when overused.`,
          examples: [],
          count,
        });
      }
    }

    // Composite score: 100 minus weighted penalties.
    let score = 100;
    for (const issue of issues) {
      const weight = issue.severity === 'error' ? 18
                   : issue.severity === 'warning' ? 8
                   : 3;
      score -= weight;
    }
    score = Math.max(0, Math.min(100, score));

    return { wordCount, issues, score };
  }

  /** Run the LLM judge — one AI call, structured JSON output. */
  async llmJudge(
    text: string,
    aiComplete: AICompleteFn,
    aiSelectProvider: AISelectProviderFn,
  ): Promise<JudgeReport | null> {
    const provider = aiSelectProvider('revision');
    // Cap input — judge doesn't need the entire chapter to score it; first 8K
    // chars is plenty for an editor's read.
    const sample = text.length > 8000
      ? text.slice(0, 4000) + '\n\n[...middle truncated for evaluation...]\n\n' + text.slice(-3000)
      : text;

    let raw = '';
    try {
      const response = await aiComplete({
        provider: provider.id,
        system: JUDGE_SYSTEM_PROMPT,
        messages: [{ role: 'user', content: sample }],
        maxTokens: 1500,
        temperature: 0.3,
      });
      raw = response.text || '';
    } catch (err) {
      console.warn('  [writing-judge] LLM call failed:', (err as Error)?.message || err);
      return null;
    }

    // Parse JSON defensively (same approach as ContextEngine).
    const cleaned = raw.replace(/```json\s*/gi, '').replace(/```\s*/gi, '').trim();
    const start = cleaned.indexOf('{');
    const end = cleaned.lastIndexOf('}');
    if (start < 0 || end <= start) {
      console.warn('  [writing-judge] judge returned non-JSON; skipping LLM scoring');
      return null;
    }
    let parsed: any;
    try {
      parsed = JSON.parse(cleaned.substring(start, end + 1));
    } catch {
      try {
        parsed = JSON.parse(cleaned.substring(start, end + 1).replace(/,\s*([}\]])/g, '$1'));
      } catch {
        return null;
      }
    }

    const dims: JudgeDimension[] = Array.isArray(parsed?.dimensions)
      ? parsed.dimensions.filter((d: any) =>
          typeof d?.name === 'string' && typeof d.score === 'number' && Array.isArray(d.issues))
      : [];
    if (dims.length === 0) return null;

    const overall = dims.reduce((sum, d) => sum + Math.max(1, Math.min(10, d.score)), 0) / dims.length;
    // Top issues = lowest-scoring dimensions' first issue, capped at 3.
    const topIssues = [...dims]
      .sort((a, b) => a.score - b.score)
      .slice(0, 3)
      .map(d => `[${d.name} ${d.score}/10] ${d.issues[0] || ''}`)
      .filter(s => s.length > 5);

    return {
      dimensions: dims.map(d => ({
        name: d.name,
        score: Math.round(Math.max(1, Math.min(10, d.score)) * 10) / 10,
        issues: (d.issues || []).slice(0, 3),
      })),
      overall: Math.round(overall * 10) / 10,
      topIssues,
    };
  }

  /**
   * Combined evaluation. Returns a verdict including whether to retry.
   *
   * Default scoring weights:
   *   - mechanical 30%, judge 70% (judge catches things mechanical can't,
   *     mechanical catches things judge writes itself)
   * Threshold default: 70/100. Below = retry.
   */
  async evaluate(
    text: string,
    opts: {
      aiComplete?: AICompleteFn;
      aiSelectProvider?: AISelectProviderFn;
      threshold?: number;        // 0-100; default 70
      mechanicalWeight?: number; // 0-1; default 0.3
      runLLMJudge?: boolean;     // default true if AI fns provided
    } = {}
  ): Promise<QualityVerdict> {
    const mechanical = this.mechanicalScreen(text);

    let judge: JudgeReport | null = null;
    if (opts.runLLMJudge !== false && opts.aiComplete && opts.aiSelectProvider) {
      judge = await this.llmJudge(text, opts.aiComplete, opts.aiSelectProvider);
    }

    const mechWeight = opts.mechanicalWeight ?? 0.3;
    const judgeWeight = 1 - mechWeight;

    // Normalize judge to 0-100 scale.
    const judgeScore100 = judge ? judge.overall * 10 : null;
    let combined: number;
    if (judgeScore100 !== null) {
      combined = mechanical.score * mechWeight + judgeScore100 * judgeWeight;
    } else {
      combined = mechanical.score; // Mechanical-only fallback.
    }
    combined = Math.round(combined * 10) / 10;

    const threshold = opts.threshold ?? 70;
    const retry = combined < threshold;

    // Build retry feedback — concise actionable steering for the next draft.
    const feedbackLines: string[] = [];
    if (judge && judge.topIssues.length > 0) {
      feedbackLines.push('## Top issues to fix on rewrite');
      for (const issue of judge.topIssues) feedbackLines.push(`- ${issue}`);
    }
    if (mechanical.issues.length > 0) {
      const errors = mechanical.issues.filter(i => i.severity === 'error');
      const warnings = mechanical.issues.filter(i => i.severity === 'warning');
      if (errors.length > 0) {
        feedbackLines.push('\n## Mechanical errors');
        for (const e of errors) {
          feedbackLines.push(`- ${e.description}${e.examples.length > 0 ? ` (e.g., "${e.examples[0]}")` : ''}`);
        }
      }
      if (warnings.length > 0) {
        feedbackLines.push('\n## Mechanical warnings');
        for (const w of warnings.slice(0, 4)) {
          feedbackLines.push(`- ${w.description}`);
        }
      }
    }
    const retryFeedback = feedbackLines.join('\n');

    const summary = judge
      ? `Score ${combined}/100 (mechanical ${mechanical.score}, judge ${judge.overall}/10). ${retry ? '↻ retry' : '✓ pass'}.`
      : `Score ${combined}/100 (mechanical-only). ${retry ? '↻ retry' : '✓ pass'}.`;

    return { score: combined, retry, mechanical, judge, summary, retryFeedback };
  }
}
