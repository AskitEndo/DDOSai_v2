import React from "react";
import { DetectionResult } from "../types";
import { Brain, Info } from "lucide-react";

interface XAIPanelProps {
  detection: DetectionResult | null;
  loading?: boolean;
}

const XAIPanel: React.FC<XAIPanelProps> = ({ detection, loading = false }) => {
  if (loading) {
    return (
      <div className="bg-gray-800 border border-gray-700 rounded-lg p-6">
        <h2 className="text-xl font-semibold text-white mb-4 flex items-center">
          <Brain className="w-5 h-5 mr-2" />
          AI Explanation
        </h2>
        <div className="animate-pulse space-y-3">
          {[...Array(3)].map((_, i) => (
            <div key={i} className="bg-gray-700 h-8 rounded"></div>
          ))}
        </div>
      </div>
    );
  }

  return (
    <div className="bg-gray-800 border border-gray-700 rounded-lg p-6">
      <h2 className="text-xl font-semibold text-white mb-4 flex items-center">
        <Brain className="w-5 h-5 mr-2" />
        AI Explanation
      </h2>

      {detection ? (
        <div className="space-y-4">
          <div className="bg-gray-700/50 rounded p-3">
            <h3 className="text-white font-medium mb-2">Detection Method</h3>
            <p className="text-gray-300 text-sm">
              {detection.detection_method}
            </p>
          </div>

          <div className="bg-gray-700/50 rounded p-3">
            <h3 className="text-white font-medium mb-2">Confidence Score</h3>
            <div className="flex items-center">
              <div className="flex-1 bg-gray-600 rounded-full h-2 mr-3">
                <div
                  className="bg-blue-500 h-2 rounded-full"
                  style={{ width: `${detection.confidence * 100}%` }}
                ></div>
              </div>
              <span className="text-white text-sm">
                {(detection.confidence * 100).toFixed(1)}%
              </span>
            </div>
          </div>

          <div className="bg-gray-700/50 rounded p-3">
            <h3 className="text-white font-medium mb-2">Model Version</h3>
            <p className="text-gray-300 text-sm">{detection.model_version}</p>
          </div>

          {detection.explanation && (
            <div className="bg-gray-700/50 rounded p-3">
              <h3 className="text-white font-medium mb-2">
                Additional Details
              </h3>
              <pre className="text-gray-300 text-xs overflow-auto">
                {JSON.stringify(detection.explanation, null, 2)}
              </pre>
            </div>
          )}
        </div>
      ) : (
        <div className="text-center text-gray-400 py-8">
          <Info className="w-12 h-12 mx-auto mb-2 opacity-50" />
          <p>Select a detection to view AI explanation</p>
        </div>
      )}
    </div>
  );
};

export default XAIPanel;
