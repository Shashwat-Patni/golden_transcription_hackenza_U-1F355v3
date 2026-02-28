import { useState, useEffect } from 'react';
import { AudioSample, TranscriptionWithScore } from '../types';
import { Music, Globe, Clock } from 'lucide-react';

interface Props {
  sample: AudioSample;
  transcriptions: TranscriptionWithScore[];
}

export function AudioSampleViewer({ sample, transcriptions }: Props) {
  const [selectedOption, setSelectedOption] = useState<number | null>(null);
  const goldenTranscription = transcriptions.find((t) => t.is_golden);

  useEffect(() => {
    if (goldenTranscription) {
      setSelectedOption(goldenTranscription.option_number);
    }
  }, [goldenTranscription]);

  const selectedTranscription = transcriptions.find(
    (t) => t.option_number === selectedOption
  );

  return (
    <div className="bg-white rounded-lg shadow-sm border border-gray-200 overflow-hidden">
      <div className="px-6 py-4 bg-gradient-to-r from-blue-600 to-blue-700 text-white">
        <h3 className="text-lg font-semibold mb-3">Audio Sample: {sample.audio_id}</h3>
        <div className="flex items-center gap-6 text-sm">
          <div className="flex items-center gap-2">
            <Globe className="w-4 h-4" />
            <span>{sample.language}</span>
          </div>
          {sample.duration && (
            <div className="flex items-center gap-2">
              <Clock className="w-4 h-4" />
              <span>{sample.duration.toFixed(1)}s</span>
            </div>
          )}
        </div>
      </div>

      <div className="p-6">
        <div className="mb-6">
          <div className="flex items-center gap-3 mb-3">
            <Music className="w-5 h-5 text-gray-600" />
            <h4 className="text-sm font-medium text-gray-700">Audio Player</h4>
          </div>
          <audio controls className="w-full" src={sample.audio_url}>
            Your browser does not support the audio element.
          </audio>
        </div>

        <div className="mb-6">
          <h4 className="text-sm font-medium text-gray-700 mb-3">Select Transcription Option</h4>
          <div className="flex flex-wrap gap-2">
            {transcriptions.map((t) => (
              <button
                key={t.id}
                onClick={() => setSelectedOption(t.option_number)}
                className={`px-4 py-2 rounded-lg border-2 transition-all ${
                  selectedOption === t.option_number
                    ? t.is_golden
                      ? 'bg-amber-50 border-amber-400 text-amber-900'
                      : 'bg-blue-50 border-blue-400 text-blue-900'
                    : 'bg-white border-gray-200 text-gray-700 hover:border-gray-300'
                }`}
              >
                <span className="font-medium">Option {t.option_number}</span>
                {t.is_golden && (
                  <span className="ml-2 text-xs bg-amber-200 text-amber-800 px-2 py-0.5 rounded-full">
                    Golden
                  </span>
                )}
              </button>
            ))}
          </div>
        </div>

        {selectedTranscription && (
          <div className="space-y-4">
            <div
              className={`p-4 rounded-lg border-2 ${
                selectedTranscription.is_golden
                  ? 'bg-amber-50 border-amber-300'
                  : 'bg-gray-50 border-gray-200'
              }`}
            >
              <div className="flex items-center justify-between mb-2">
                <h4 className="text-sm font-medium text-gray-700">Transcription Text</h4>
                {selectedTranscription.asr_model && (
                  <span className="text-xs text-gray-500">
                    Model: {selectedTranscription.asr_model}
                  </span>
                )}
              </div>
              <p className="text-gray-800 leading-relaxed">
                {selectedTranscription.transcript_text}
              </p>
            </div>

            {selectedTranscription.scoring_result && (
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                <div className="bg-blue-50 p-3 rounded-lg">
                  <div className="text-xs text-blue-600 font-medium mb-1">Composite Score</div>
                  <div className="text-lg font-bold text-blue-900">
                    {selectedTranscription.scoring_result.composite_score.toFixed(3)}
                  </div>
                </div>
                {selectedTranscription.scoring_result.alignment_score !== null &&
                  selectedTranscription.scoring_result.alignment_score !== undefined && (
                    <div className="bg-green-50 p-3 rounded-lg">
                      <div className="text-xs text-green-600 font-medium mb-1">Alignment</div>
                      <div className="text-lg font-bold text-green-900">
                        {selectedTranscription.scoring_result.alignment_score.toFixed(3)}
                      </div>
                    </div>
                  )}
                {selectedTranscription.scoring_result.majority_vote_score !== null &&
                  selectedTranscription.scoring_result.majority_vote_score !== undefined && (
                    <div className="bg-violet-50 p-3 rounded-lg">
                      <div className="text-xs text-violet-600 font-medium mb-1">Voting</div>
                      <div className="text-lg font-bold text-violet-900">
                        {selectedTranscription.scoring_result.majority_vote_score.toFixed(3)}
                      </div>
                    </div>
                  )}
                {!selectedTranscription.is_golden &&
                  selectedTranscription.scoring_result.wer !== null &&
                  selectedTranscription.scoring_result.wer !== undefined && (
                    <div className="bg-red-50 p-3 rounded-lg">
                      <div className="text-xs text-red-600 font-medium mb-1">WER</div>
                      <div className="text-lg font-bold text-red-900">
                        {(selectedTranscription.scoring_result.wer * 100).toFixed(1)}%
                      </div>
                    </div>
                  )}
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
