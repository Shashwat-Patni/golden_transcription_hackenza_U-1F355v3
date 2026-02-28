import { TranscriptionWithScore } from '../types';
import { ArrowDown, ArrowUp, Minus } from 'lucide-react';

interface Props {
  transcriptions: TranscriptionWithScore[];
}

export function ComparisonTable({ transcriptions }: Props) {
  const goldenTranscription = transcriptions.find((t) => t.is_golden);
  const sortedTranscriptions = [...transcriptions].sort(
    (a, b) => (b.scoring_result?.composite_score || 0) - (a.scoring_result?.composite_score || 0)
  );

  return (
    <div className="bg-white rounded-lg shadow-sm border border-gray-200 overflow-hidden">
      <div className="px-6 py-4 border-b border-gray-200">
        <h3 className="text-lg font-semibold text-gray-800">Detailed Comparison</h3>
      </div>

      <div className="overflow-x-auto">
        <table className="w-full">
          <thead className="bg-gray-50">
            <tr>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                Option
              </th>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                ASR Model
              </th>
              <th className="px-6 py-3 text-center text-xs font-medium text-gray-500 uppercase tracking-wider">
                Composite Score
              </th>
              <th className="px-6 py-3 text-center text-xs font-medium text-gray-500 uppercase tracking-wider">
                WER
              </th>
              <th className="px-6 py-3 text-center text-xs font-medium text-gray-500 uppercase tracking-wider">
                Errors
              </th>
              <th className="px-6 py-3 text-center text-xs font-medium text-gray-500 uppercase tracking-wider">
                Status
              </th>
            </tr>
          </thead>
          <tbody className="bg-white divide-y divide-gray-200">
            {sortedTranscriptions.map((transcription, index) => {
              const score = transcription.scoring_result;
              if (!score) return null;

              return (
                <tr
                  key={transcription.id}
                  className={transcription.is_golden ? 'bg-amber-50' : ''}
                >
                  <td className="px-6 py-4 whitespace-nowrap">
                    <div className="flex items-center gap-2">
                      <span className="text-sm font-medium text-gray-900">
                        #{transcription.option_number}
                      </span>
                      {index === 0 && !transcription.is_golden && (
                        <ArrowUp className="w-4 h-4 text-green-600" />
                      )}
                      {index === sortedTranscriptions.length - 1 && !transcription.is_golden && (
                        <ArrowDown className="w-4 h-4 text-red-600" />
                      )}
                    </div>
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-600">
                    {transcription.asr_model || 'Unknown'}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-center">
                    <span className="text-sm font-semibold text-gray-900">
                      {score.composite_score.toFixed(4)}
                    </span>
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-center">
                    {transcription.is_golden ? (
                      <span className="text-sm text-gray-500">
                        <Minus className="w-4 h-4 inline" />
                      </span>
                    ) : (
                      <span
                        className={`text-sm font-semibold ${
                          score.wer && score.wer < 0.1
                            ? 'text-green-600'
                            : score.wer && score.wer < 0.3
                            ? 'text-yellow-600'
                            : 'text-red-600'
                        }`}
                      >
                        {score.wer ? (score.wer * 100).toFixed(1) : '0.0'}%
                      </span>
                    )}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-center">
                    {transcription.is_golden ? (
                      <span className="text-sm text-gray-500">Reference</span>
                    ) : (
                      <div className="text-xs text-gray-600">
                        <span className="text-red-600">S:{score.substitutions}</span>
                        {' / '}
                        <span className="text-orange-600">D:{score.deletions}</span>
                        {' / '}
                        <span className="text-blue-600">I:{score.insertions}</span>
                      </div>
                    )}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-center">
                    {transcription.is_golden ? (
                      <span className="inline-flex items-center px-3 py-1 rounded-full text-xs font-medium bg-amber-100 text-amber-800">
                        Golden
                      </span>
                    ) : (
                      <span
                        className={`inline-flex items-center px-3 py-1 rounded-full text-xs font-medium ${
                          score.wer && score.wer < 0.1
                            ? 'bg-green-100 text-green-800'
                            : score.wer && score.wer < 0.3
                            ? 'bg-yellow-100 text-yellow-800'
                            : 'bg-red-100 text-red-800'
                        }`}
                      >
                        {score.wer && score.wer < 0.1
                          ? 'Excellent'
                          : score.wer && score.wer < 0.3
                          ? 'Good'
                          : 'Poor'}
                      </span>
                    )}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      <div className="px-6 py-4 bg-gray-50 border-t border-gray-200">
        <div className="flex items-center gap-6 text-xs text-gray-600">
          <div>
            <span className="font-medium">Legend:</span>
          </div>
          <div>
            <span className="font-medium">S</span> = Substitutions
          </div>
          <div>
            <span className="font-medium">D</span> = Deletions
          </div>
          <div>
            <span className="font-medium">I</span> = Insertions
          </div>
          <div>
            <span className="font-medium">WER</span> = Word Error Rate
          </div>
        </div>
      </div>
    </div>
  );
}
