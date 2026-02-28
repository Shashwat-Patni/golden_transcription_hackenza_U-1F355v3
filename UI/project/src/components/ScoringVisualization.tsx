import { TranscriptionWithScore } from '../types';
import { Award, TrendingUp } from 'lucide-react';

interface Props {
  transcriptions: TranscriptionWithScore[];
}

export function ScoringVisualization({ transcriptions }: Props) {
  const goldenTranscription = transcriptions.find((t) => t.is_golden);

  return (
    <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
      <h3 className="text-lg font-semibold text-gray-800 mb-4 flex items-center gap-2">
        <TrendingUp className="w-5 h-5 text-blue-600" />
        Scoring Breakdown
      </h3>

      <div className="space-y-4">
        {transcriptions.map((transcription) => {
          const score = transcription.scoring_result;
          if (!score) return null;

          return (
            <div
              key={transcription.id}
              className={`p-4 rounded-lg border-2 ${
                transcription.is_golden
                  ? 'bg-amber-50 border-amber-400'
                  : 'bg-gray-50 border-gray-200'
              }`}
            >
              <div className="flex items-center justify-between mb-3">
                <div className="flex items-center gap-2">
                  <span className="text-sm font-medium text-gray-700">
                    Option {transcription.option_number}
                    {transcription.asr_model && (
                      <span className="text-gray-500 ml-1">({transcription.asr_model})</span>
                    )}
                  </span>
                  {transcription.is_golden && (
                    <span className="flex items-center gap-1 text-xs font-medium text-amber-700 bg-amber-200 px-2 py-1 rounded-full">
                      <Award className="w-3 h-3" />
                      Golden
                    </span>
                  )}
                </div>
                <span className="text-lg font-bold text-gray-800">
                  {score.composite_score.toFixed(3)}
                </span>
              </div>

              <div className="space-y-2">
                {score.alignment_score !== null && score.alignment_score !== undefined && (
                  <div>
                    <div className="flex justify-between text-xs text-gray-600 mb-1">
                      <span>Alignment</span>
                      <span>{score.alignment_score.toFixed(3)}</span>
                    </div>
                    <div className="h-2 bg-gray-200 rounded-full overflow-hidden">
                      <div
                        className="h-full bg-blue-500"
                        style={{ width: `${score.alignment_score * 100}%` }}
                      />
                    </div>
                  </div>
                )}

                {score.majority_vote_score !== null && score.majority_vote_score !== undefined && (
                  <div>
                    <div className="flex justify-between text-xs text-gray-600 mb-1">
                      <span>Majority Vote</span>
                      <span>{score.majority_vote_score.toFixed(3)}</span>
                    </div>
                    <div className="h-2 bg-gray-200 rounded-full overflow-hidden">
                      <div
                        className="h-full bg-green-500"
                        style={{ width: `${score.majority_vote_score * 100}%` }}
                      />
                    </div>
                  </div>
                )}

                {score.similarity_score !== null && score.similarity_score !== undefined && (
                  <div>
                    <div className="flex justify-between text-xs text-gray-600 mb-1">
                      <span>Similarity</span>
                      <span>{score.similarity_score.toFixed(3)}</span>
                    </div>
                    <div className="h-2 bg-gray-200 rounded-full overflow-hidden">
                      <div
                        className="h-full bg-violet-500"
                        style={{ width: `${score.similarity_score * 100}%` }}
                      />
                    </div>
                  </div>
                )}
              </div>

              {!transcription.is_golden && score.wer !== null && score.wer !== undefined && (
                <div className="mt-3 pt-3 border-t border-gray-200">
                  <div className="flex justify-between items-center">
                    <span className="text-xs text-gray-600">WER vs Golden:</span>
                    <span className="text-sm font-semibold text-red-600">
                      {(score.wer * 100).toFixed(1)}%
                    </span>
                  </div>
                </div>
              )}
            </div>
          );
        })}
      </div>

      {goldenTranscription && (
        <div className="mt-6 p-4 bg-blue-50 border border-blue-200 rounded-lg">
          <h4 className="text-sm font-semibold text-blue-900 mb-2">Why this is the Golden Transcript</h4>
          <p className="text-sm text-blue-800">
            Option {goldenTranscription.option_number} achieved the highest composite quality score
            of {goldenTranscription.scoring_result?.composite_score.toFixed(3)}, combining optimal
            acoustic alignment, linguistic plausibility, and similarity to the audio signal.
          </p>
        </div>
      )}
    </div>
  );
}
