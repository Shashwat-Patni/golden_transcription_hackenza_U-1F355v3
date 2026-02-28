import { useState, useEffect } from 'react';
import { Settings, Save } from 'lucide-react';
import { supabase } from '../lib/supabase';
import { SystemConfig } from '../types';

const ASR_OPTIONS = [
  'Whisper-Large-v3',
  'Whisper-Medium',
  'Wav2Vec2-Large',
  'HuBERT-Large',
  'Qwen2-Audio',
];

export function ConfigurationPanel() {
  const [config, setConfig] = useState<SystemConfig | null>(null);
  const [isEditing, setIsEditing] = useState(false);
  const [saving, setSaving] = useState(false);

  useEffect(() => {
    loadConfig();
  }, []);

  const loadConfig = async () => {
    const { data } = await supabase
      .from('system_config')
      .select('*')
      .eq('is_active', true)
      .maybeSingle();

    if (data) {
      setConfig(data);
    }
  };

  const handleSave = async () => {
    if (!config) return;

    setSaving(true);
    const { error } = await supabase
      .from('system_config')
      .update({
        alignment_weight: config.alignment_weight,
        voting_weight: config.voting_weight,
        similarity_weight: config.similarity_weight,
        baseline_asr: config.baseline_asr,
        updated_at: new Date().toISOString(),
      })
      .eq('id', config.id);

    if (!error) {
      setIsEditing(false);
    }
    setSaving(false);
  };

  const updateWeight = (field: keyof SystemConfig, value: number) => {
    if (!config) return;
    setConfig({ ...config, [field]: value });
  };

  const totalWeight = config
    ? config.alignment_weight + config.voting_weight + config.similarity_weight
    : 0;

  if (!config) {
    return <div className="text-gray-500">Loading configuration...</div>;
  }

  return (
    <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
      <div className="flex items-center justify-between mb-6">
        <div className="flex items-center gap-3">
          <Settings className="w-6 h-6 text-blue-600" />
          <h2 className="text-xl font-semibold text-gray-800">System Configuration</h2>
        </div>
        <button
          onClick={() => (isEditing ? handleSave() : setIsEditing(true))}
          disabled={saving || (isEditing && Math.abs(totalWeight - 1) > 0.001)}
          className="flex items-center gap-2 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:bg-gray-400 disabled:cursor-not-allowed transition-colors"
        >
          <Save className="w-4 h-4" />
          {saving ? 'Saving...' : isEditing ? 'Save Changes' : 'Edit'}
        </button>
      </div>

      <div className="space-y-6">
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">
            Baseline ASR Model
          </label>
          <select
            value={config.baseline_asr}
            onChange={(e) => setConfig({ ...config, baseline_asr: e.target.value })}
            disabled={!isEditing}
            className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent disabled:bg-gray-50 disabled:cursor-not-allowed"
          >
            {ASR_OPTIONS.map((asr) => (
              <option key={asr} value={asr}>
                {asr}
              </option>
            ))}
          </select>
          <p className="mt-1 text-sm text-gray-500">
            Used for generating the ground truth baseline transcription
          </p>
        </div>

        <div>
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-sm font-medium text-gray-700">Scoring Weights</h3>
            {isEditing && (
              <span
                className={`text-sm font-medium ${
                  Math.abs(totalWeight - 1) < 0.001 ? 'text-green-600' : 'text-red-600'
                }`}
              >
                Total: {totalWeight.toFixed(2)} {Math.abs(totalWeight - 1) < 0.001 ? '✓' : '(must equal 1.00)'}
              </span>
            )}
          </div>

          <div className="space-y-4">
            <div>
              <div className="flex items-center justify-between mb-2">
                <label className="text-sm text-gray-600">Alignment Score Weight</label>
                <span className="text-sm font-medium text-gray-800">
                  {config.alignment_weight.toFixed(2)}
                </span>
              </div>
              <input
                type="range"
                min="0"
                max="1"
                step="0.01"
                value={config.alignment_weight}
                onChange={(e) => updateWeight('alignment_weight', parseFloat(e.target.value))}
                disabled={!isEditing}
                className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer accent-blue-600 disabled:cursor-not-allowed"
              />
              <p className="mt-1 text-xs text-gray-500">
                Acoustic log-likelihood score from forced alignment
              </p>
            </div>

            <div>
              <div className="flex items-center justify-between mb-2">
                <label className="text-sm text-gray-600">Majority Voting Weight</label>
                <span className="text-sm font-medium text-gray-800">
                  {config.voting_weight.toFixed(2)}
                </span>
              </div>
              <input
                type="range"
                min="0"
                max="1"
                step="0.01"
                value={config.voting_weight}
                onChange={(e) => updateWeight('voting_weight', parseFloat(e.target.value))}
                disabled={!isEditing}
                className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer accent-blue-600 disabled:cursor-not-allowed"
              />
              <p className="mt-1 text-xs text-gray-500">
                LLM-as-a-Judge ensemble voting score
              </p>
            </div>

            <div>
              <div className="flex items-center justify-between mb-2">
                <label className="text-sm text-gray-600">Acoustic-Text Similarity Weight</label>
                <span className="text-sm font-medium text-gray-800">
                  {config.similarity_weight.toFixed(2)}
                </span>
              </div>
              <input
                type="range"
                min="0"
                max="1"
                step="0.01"
                value={config.similarity_weight}
                onChange={(e) => updateWeight('similarity_weight', parseFloat(e.target.value))}
                disabled={!isEditing}
                className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer accent-blue-600 disabled:cursor-not-allowed"
              />
              <p className="mt-1 text-xs text-gray-500">
                Cosine similarity in joint embedding space
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
