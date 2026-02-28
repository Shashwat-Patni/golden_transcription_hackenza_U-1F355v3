/*
  # ASR Transcription Evaluation System Schema

  1. New Tables
    - `audio_samples`
      - `id` (uuid, primary key)
      - `audio_id` (text, unique identifier from dataset)
      - `language` (text, language of the audio)
      - `audio_url` (text, URL to the audio file)
      - `duration` (numeric, duration in seconds)
      - `created_at` (timestamptz)
    
    - `transcriptions`
      - `id` (uuid, primary key)
      - `audio_sample_id` (uuid, foreign key to audio_samples)
      - `option_number` (int, 1-5 for the transcript options)
      - `transcript_text` (text, the actual transcription)
      - `asr_model` (text, which ASR model generated this)
      - `is_golden` (boolean, whether this is the selected golden transcript)
      - `created_at` (timestamptz)
    
    - `scoring_results`
      - `id` (uuid, primary key)
      - `transcription_id` (uuid, foreign key to transcriptions)
      - `alignment_score` (numeric, acoustic alignment score)
      - `majority_vote_score` (numeric, linguistic voting score)
      - `similarity_score` (numeric, acoustic-text similarity)
      - `composite_score` (numeric, final weighted score)
      - `wer` (numeric, word error rate vs golden)
      - `substitutions` (int, substitution errors)
      - `deletions` (int, deletion errors)
      - `insertions` (int, insertion errors)
      - `created_at` (timestamptz)
    
    - `system_config`
      - `id` (uuid, primary key)
      - `config_name` (text, unique name for the configuration)
      - `alignment_weight` (numeric, weight for alignment score)
      - `voting_weight` (numeric, weight for majority voting)
      - `similarity_weight` (numeric, weight for similarity score)
      - `baseline_asr` (text, which ASR to use for ground truth)
      - `is_active` (boolean, whether this is the active config)
      - `created_at` (timestamptz)
      - `updated_at` (timestamptz)

  2. Security
    - Enable RLS on all tables
    - Add policies for authenticated users to read and write data
*/

CREATE TABLE IF NOT EXISTS audio_samples (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  audio_id text UNIQUE NOT NULL,
  language text NOT NULL,
  audio_url text NOT NULL,
  duration numeric,
  created_at timestamptz DEFAULT now()
);

CREATE TABLE IF NOT EXISTS transcriptions (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  audio_sample_id uuid REFERENCES audio_samples(id) ON DELETE CASCADE,
  option_number int NOT NULL CHECK (option_number BETWEEN 1 AND 5),
  transcript_text text NOT NULL,
  asr_model text,
  is_golden boolean DEFAULT false,
  created_at timestamptz DEFAULT now(),
  UNIQUE(audio_sample_id, option_number)
);

CREATE TABLE IF NOT EXISTS scoring_results (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  transcription_id uuid REFERENCES transcriptions(id) ON DELETE CASCADE UNIQUE,
  alignment_score numeric,
  majority_vote_score numeric,
  similarity_score numeric,
  composite_score numeric NOT NULL,
  wer numeric,
  substitutions int DEFAULT 0,
  deletions int DEFAULT 0,
  insertions int DEFAULT 0,
  created_at timestamptz DEFAULT now()
);

CREATE TABLE IF NOT EXISTS system_config (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  config_name text UNIQUE NOT NULL,
  alignment_weight numeric NOT NULL DEFAULT 0.33 CHECK (alignment_weight BETWEEN 0 AND 1),
  voting_weight numeric NOT NULL DEFAULT 0.33 CHECK (voting_weight BETWEEN 0 AND 1),
  similarity_weight numeric NOT NULL DEFAULT 0.34 CHECK (similarity_weight BETWEEN 0 AND 1),
  baseline_asr text NOT NULL DEFAULT 'Whisper-Large-v3',
  is_active boolean DEFAULT false,
  created_at timestamptz DEFAULT now(),
  updated_at timestamptz DEFAULT now()
);

ALTER TABLE audio_samples ENABLE ROW LEVEL SECURITY;
ALTER TABLE transcriptions ENABLE ROW LEVEL SECURITY;
ALTER TABLE scoring_results ENABLE ROW LEVEL SECURITY;
ALTER TABLE system_config ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Anyone can read audio samples"
  ON audio_samples FOR SELECT
  USING (true);

CREATE POLICY "Anyone can insert audio samples"
  ON audio_samples FOR INSERT
  WITH CHECK (true);

CREATE POLICY "Anyone can read transcriptions"
  ON transcriptions FOR SELECT
  USING (true);

CREATE POLICY "Anyone can insert transcriptions"
  ON transcriptions FOR INSERT
  WITH CHECK (true);

CREATE POLICY "Anyone can update transcriptions"
  ON transcriptions FOR UPDATE
  USING (true)
  WITH CHECK (true);

CREATE POLICY "Anyone can read scoring results"
  ON scoring_results FOR SELECT
  USING (true);

CREATE POLICY "Anyone can insert scoring results"
  ON scoring_results FOR INSERT
  WITH CHECK (true);

CREATE POLICY "Anyone can read system config"
  ON system_config FOR SELECT
  USING (true);

CREATE POLICY "Anyone can insert system config"
  ON system_config FOR INSERT
  WITH CHECK (true);

CREATE POLICY "Anyone can update system config"
  ON system_config FOR UPDATE
  USING (true)
  WITH CHECK (true);

INSERT INTO system_config (config_name, alignment_weight, voting_weight, similarity_weight, baseline_asr, is_active)
VALUES ('Default Configuration', 0.33, 0.33, 0.34, 'Whisper-Large-v3', true)
ON CONFLICT (config_name) DO NOTHING;