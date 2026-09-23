export type Verdict = "entailed" | "neutral" | "contradicted" | "unsupported"
export type LanguageType = "hindi" | "english" | "code_mixed"
export type Length = "short" | "medium" | "long"
export type Style = "paragraph" | "bullets"

export interface Claim {
  claim_text: string
  claim_en: string
  verdict: Verdict
  nli_confidence: number
  evidence_chunk_id: number | null
  evidence_text: string
  evidence_start: number | null
  evidence_end: number | null
  rerank_score: number
}

export interface Mismatch {
  text: string
  type: "number" | "date" | "entity" | "script"
  reason: string
}

export interface Stats {
  words_in: number
  words_out: number
  compression_ratio: number
  inference_time_ms: number
}

export interface CompareResult {
  standard: {
    summary: string
    factuality_score: number
    claim_score: number
    entity_score: number
    claims: Claim[]
    entity_mismatches: Mismatch[]
  }
  factuality_aware: { summary: string; factuality_score: number; first_attempt_score: number }
}

export interface SummaryResult {
  doc_id: string
  attempt_id: string
  summary: string
  prompt_type: string
  factuality_score: number
  claim_score: number
  entity_score: number
  low_confidence: boolean
  claims: Claim[]
  entity_mismatches: Mismatch[]
  attempts: number
  language_type: LanguageType
  source_text: string
  stats: Stats
  compare?: CompareResult
  options?: { length: Length; style: Style; compare: boolean }
  feedback?: "up" | "down" | null
}

export interface HistoryItem {
  id: string
  title: string
  created_at: string
  language_type: LanguageType
  final_score: number
  low_confidence: boolean
}

export interface AppStats {
  total_documents: number
  avg_factuality_score: number | null
  avg_inference_time_ms: number | null
  code_mixed_documents: number
}

export interface Health {
  status: "ok" | "degraded"
  llm: { model: string; configured: boolean }
  factuality_threshold: number
}
