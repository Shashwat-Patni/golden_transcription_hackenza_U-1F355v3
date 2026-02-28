export interface AudioSample {
  id: string;
  audio_id: string;
  language: string;
  audio_url: string;
  duration?: number;
  created_at: string;
}

export interface Transcription {
  id: string;
  audio_sample_id: string;
  option_number: number;
  transcript_text: string;
  asr_model?: string;
  is_golden: boolean;
  created_at: string;
}

export interface ScoringResult {
  id: string;
  transcription_id: string;
  alignment_score?: number;
  majority_vote_score?: number;
  similarity_score?: number;
  composite_score: number;
  wer?: number;
  substitutions: number;
  deletions: number;
  insertions: number;
  created_at: string;
}

export interface SystemConfig {
  id: string;
  config_name: string;
  alignment_weight: number;
  voting_weight: number;
  similarity_weight: number;
  baseline_asr: string;
  is_active: boolean;
  created_at: string;
  updated_at: string;
}

export interface TranscriptionWithScore extends Transcription {
  scoring_result?: ScoringResult;
}
