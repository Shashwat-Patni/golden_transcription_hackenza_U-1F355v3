import { useEffect, useState } from 'react';
import { AudioLines, ChevronDown, ChevronUp } from 'lucide-react';
import { supabase } from './lib/supabase';
import { AudioSample, TranscriptionWithScore } from './types';
import { ConfigurationPanel } from './components/ConfigurationPanel';
import { AudioSampleViewer } from './components/AudioSampleViewer';
import { ScoringVisualization } from './components/ScoringVisualization';
import { ComparisonTable } from './components/ComparisonTable';
import { SampleDataLoader } from './components/SampleDataLoader';

function App() {
  const [audioSamples, setAudioSamples] = useState<AudioSample[]>([]);
  const [selectedSampleId, setSelectedSampleId] = useState<string | null>(null);
  const [transcriptions, setTranscriptions] = useState<TranscriptionWithScore[]>([]);
  const [loading, setLoading] = useState(true);
  const [configExpanded, setConfigExpanded] = useState(true);

  useEffect(() => {
    loadAudioSamples();
  }, []);

  useEffect(() => {
    if (selectedSampleId) {
      loadTranscriptions(selectedSampleId);
    }
  }, [selectedSampleId]);

  const loadAudioSamples = async () => {
    setLoading(true);
    const { data } = await supabase
      .from('audio_samples')
      .select('*')
      .order('created_at', { ascending: false });

    if (data && data.length > 0) {
      setAudioSamples(data);
      setSelectedSampleId(data[0].id);
    }
    setLoading(false);
  };

  const loadTranscriptions = async (sampleId: string) => {
    const { data: transData } = await supabase
      .from('transcriptions')
      .select('*')
      .eq('audio_sample_id', sampleId)
      .order('option_number');

    if (transData) {
      const transcriptionsWithScores: TranscriptionWithScore[] = await Promise.all(
        transData.map(async (t) => {
          const { data: scoreData } = await supabase
            .from('scoring_results')
            .select('*')
            .eq('transcription_id', t.id)
            .maybeSingle();

          return {
            ...t,
            scoring_result: scoreData || undefined,
          };
        })
      );
      setTranscriptions(transcriptionsWithScores);
    }
  };

  const selectedSample = audioSamples.find((s) => s.id === selectedSampleId);

  if (loading) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <div className="text-center">
          <AudioLines className="w-12 h-12 text-blue-600 animate-pulse mx-auto mb-4" />
          <p className="text-gray-600">Loading ASR Evaluation System...</p>
        </div>
      </div>
    );
  }

  if (audioSamples.length === 0) {
    return (
      <div className="min-h-screen bg-gray-50">
        <header className="bg-white shadow-sm border-b border-gray-200">
          <div className="max-w-7xl mx-auto px-6 py-6">
            <div className="flex items-center gap-3">
              <AudioLines className="w-8 h-8 text-blue-600" />
              <div>
                <h1 className="text-2xl font-bold text-gray-900">
                  ASR Transcription Evaluation System
                </h1>
                <p className="text-sm text-gray-600 mt-1">
                  Golden transcript selection and quality scoring
                </p>
              </div>
            </div>
          </div>
        </header>

        <main className="max-w-7xl mx-auto px-6 py-8">
          <SampleDataLoader />
        </main>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-50">
      <header className="bg-white shadow-sm border-b border-gray-200 sticky top-0 z-10">
        <div className="max-w-7xl mx-auto px-6 py-6">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <AudioLines className="w-8 h-8 text-blue-600" />
              <div>
                <h1 className="text-2xl font-bold text-gray-900">
                  ASR Transcription Evaluation System
                </h1>
                <p className="text-sm text-gray-600 mt-1">
                  Golden transcript selection and quality scoring
                </p>
              </div>
            </div>
            {audioSamples.length > 1 && (
              <select
                value={selectedSampleId || ''}
                onChange={(e) => setSelectedSampleId(e.target.value)}
                className="px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              >
                {audioSamples.map((sample) => (
                  <option key={sample.id} value={sample.id}>
                    {sample.audio_id} ({sample.language})
                  </option>
                ))}
              </select>
            )}
          </div>
        </div>
      </header>

      <main className="max-w-7xl mx-auto px-6 py-8 space-y-6">
        <div>
          <button
            onClick={() => setConfigExpanded(!configExpanded)}
            className="flex items-center gap-2 text-sm text-gray-600 hover:text-gray-900 mb-4"
          >
            {configExpanded ? (
              <ChevronUp className="w-4 h-4" />
            ) : (
              <ChevronDown className="w-4 h-4" />
            )}
            {configExpanded ? 'Hide' : 'Show'} Configuration
          </button>
          {configExpanded && <ConfigurationPanel />}
        </div>

        {selectedSample && transcriptions.length > 0 && (
          <>
            <AudioSampleViewer sample={selectedSample} transcriptions={transcriptions} />

            <div className="grid lg:grid-cols-2 gap-6">
              <ScoringVisualization transcriptions={transcriptions} />
              <div className="space-y-6">
                <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
                  <h3 className="text-lg font-semibold text-gray-800 mb-4">System Overview</h3>
                  <div className="space-y-3 text-sm text-gray-600">
                    <div className="flex justify-between">
                      <span>Total Transcriptions:</span>
                      <span className="font-medium text-gray-900">{transcriptions.length}</span>
                    </div>
                    <div className="flex justify-between">
                      <span>Golden Transcript:</span>
                      <span className="font-medium text-gray-900">
                        Option {transcriptions.find((t) => t.is_golden)?.option_number}
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span>Average WER:</span>
                      <span className="font-medium text-gray-900">
                        {(
                          (transcriptions
                            .filter((t) => !t.is_golden && t.scoring_result?.wer)
                            .reduce((sum, t) => sum + (t.scoring_result?.wer || 0), 0) /
                            transcriptions.filter((t) => !t.is_golden).length) *
                          100
                        ).toFixed(1)}
                        %
                      </span>
                    </div>
                  </div>
                </div>

                <div className="bg-blue-50 rounded-lg border border-blue-200 p-6">
                  <h4 className="text-sm font-semibold text-blue-900 mb-3">
                    Scoring Methodology
                  </h4>
                  <ul className="space-y-2 text-sm text-blue-800">
                    <li>
                      <span className="font-medium">Alignment Score:</span> Acoustic log-likelihood
                      from forced alignment
                    </li>
                    <li>
                      <span className="font-medium">Majority Voting:</span> LLM-as-a-Judge ensemble
                      evaluation
                    </li>
                    <li>
                      <span className="font-medium">Similarity Score:</span> Cosine similarity in
                      joint embedding space
                    </li>
                  </ul>
                </div>
              </div>
            </div>

            <ComparisonTable transcriptions={transcriptions} />
          </>
        )}
      </main>
    </div>
  );
}

export default App;
