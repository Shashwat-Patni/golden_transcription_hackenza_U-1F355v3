import { useState } from 'react';
import { Upload, Database } from 'lucide-react';
import { supabase } from '../lib/supabase';

export function SampleDataLoader() {
  const [loading, setLoading] = useState(false);

  const loadSampleData = async () => {
    setLoading(true);
    try {
      const sampleAudio = {
        audio_id: 'sample_001',
        language: 'English',
        audio_url: 'https://example.com/sample.mp3',
        duration: 15.5,
      };

      const { data: audioData, error: audioError } = await supabase
        .from('audio_samples')
        .upsert(sampleAudio, { onConflict: 'audio_id' })
        .select()
        .single();

      if (audioError) throw audioError;

      const sampleTranscriptions = [
        {
          audio_sample_id: audioData.id,
          option_number: 1,
          transcript_text:
            'The quick brown fox jumps over the lazy dog in the bright morning sun.',
          asr_model: 'Whisper-Large-v3',
          is_golden: true,
        },
        {
          audio_sample_id: audioData.id,
          option_number: 2,
          transcript_text:
            'The quick brown fox jumped over the lazy dog in the bright morning sun.',
          asr_model: 'Wav2Vec2-Large',
          is_golden: false,
        },
        {
          audio_sample_id: audioData.id,
          option_number: 3,
          transcript_text: 'The quick brown fox jumps over a lazy dog in bright morning sun.',
          asr_model: 'HuBERT-Large',
          is_golden: false,
        },
        {
          audio_sample_id: audioData.id,
          option_number: 4,
          transcript_text:
            'A quick brown fox jumps over the lazy dog in the bright morning sunshine.',
          asr_model: 'Qwen2-Audio',
          is_golden: false,
        },
        {
          audio_sample_id: audioData.id,
          option_number: 5,
          transcript_text: 'The quick brown fox jump over the lazy dogs in bright morning.',
          asr_model: 'Whisper-Medium',
          is_golden: false,
        },
      ];

      const { data: transcriptionData, error: transcriptionError } = await supabase
        .from('transcriptions')
        .upsert(sampleTranscriptions, { onConflict: 'audio_sample_id,option_number' })
        .select();

      if (transcriptionError) throw transcriptionError;

      const scoringResults = [
        {
          transcription_id: transcriptionData[0].id,
          alignment_score: 0.92,
          majority_vote_score: 0.88,
          similarity_score: 0.91,
          composite_score: 0.903,
          wer: null,
          substitutions: 0,
          deletions: 0,
          insertions: 0,
        },
        {
          transcription_id: transcriptionData[1].id,
          alignment_score: 0.89,
          majority_vote_score: 0.85,
          similarity_score: 0.87,
          composite_score: 0.870,
          wer: 0.071,
          substitutions: 1,
          deletions: 0,
          insertions: 0,
        },
        {
          transcription_id: transcriptionData[2].id,
          alignment_score: 0.85,
          majority_vote_score: 0.82,
          similarity_score: 0.84,
          composite_score: 0.837,
          wer: 0.143,
          substitutions: 1,
          deletions: 1,
          insertions: 0,
        },
        {
          transcription_id: transcriptionData[3].id,
          alignment_score: 0.81,
          majority_vote_score: 0.79,
          similarity_score: 0.80,
          composite_score: 0.800,
          wer: 0.214,
          substitutions: 2,
          deletions: 0,
          insertions: 1,
        },
        {
          transcription_id: transcriptionData[4].id,
          alignment_score: 0.76,
          majority_vote_score: 0.73,
          similarity_score: 0.75,
          composite_score: 0.747,
          wer: 0.286,
          substitutions: 2,
          deletions: 2,
          insertions: 1,
        },
      ];

      await supabase.from('scoring_results').upsert(scoringResults, {
        onConflict: 'transcription_id',
      });

      window.location.reload();
    } catch (error) {
      console.error('Error loading sample data:', error);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
      <div className="text-center">
        <Database className="w-12 h-12 text-blue-600 mx-auto mb-4" />
        <h3 className="text-lg font-semibold text-gray-800 mb-2">Load Sample Data</h3>
        <p className="text-sm text-gray-600 mb-4">
          Load demonstration data to see the system in action
        </p>
        <button
          onClick={loadSampleData}
          disabled={loading}
          className="flex items-center gap-2 px-6 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:bg-gray-400 disabled:cursor-not-allowed transition-colors mx-auto"
        >
          <Upload className="w-4 h-4" />
          {loading ? 'Loading...' : 'Load Sample Data'}
        </button>
      </div>
    </div>
  );
}
