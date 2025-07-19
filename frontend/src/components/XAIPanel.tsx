import React from "react";
import { DetectionResult } from "../types";
import {
  Brain,
  Info,
  AlertTriangle,
  CheckCircle,
  BarChart2,
  Code,
  Zap,
} from "lucide-react";

interface XAIPanelProps {
  detection: DetectionResult | null;
  loading?: boolean;
}

const XAIPanel: React.FC<XAIPanelProps> = ({ detection, loading = false }) => {
  const getConfidenceColor = (confidence: number) => {
    const percent = confidence * 100;
    if (percent >= 90) return "bg-red-500";
    if (percent >= 70) return "bg-orange-500";
    if (percent >= 50) return "bg-yellow-500";
    return "bg-blue-500";
  };

  const getConfidenceTextColor = (confidence: number) => {
    const percent = confidence * 100;
    if (percent >= 90) return "text-red-400";
    if (percent >= 70) return "text-orange-400";
    if (percent >= 50) return "text-yellow-400";
    return "text-blue-400";
  };

  if (loading) {
    return (
      <div className="bg-gray-800/70 border border-gray-700 rounded-xl p-6 shadow-lg">
        <h2 className="text-xl font-semibold text-white mb-4 flex items-center">
          <Brain className="w-5 h-5 mr-2 text-purple-400" />
          AI Explanation
        </h2>
        <div className="animate-pulse space-y-3">
          {[...Array(3)].map((_, i) => (
            <div key={i} className="bg-gray-700/70 h-12 rounded-lg"></div>
          ))}
        </div>
      </div>
    );
  }

  return (
    <div className="bg-gray-800/70 border border-gray-700 rounded-xl p-6 shadow-lg hover:shadow-xl transition-all duration-300 hover:border-purple-700/50">
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-xl font-semibold text-white flex items-center">
          <Brain className="w-5 h-5 mr-2 text-purple-400" />
          AI Explanation
        </h2>
        {detection && (
          <div
            className={`px-2 py-1 rounded-full text-xs font-medium flex items-center ${
              detection.is_malicious
                ? "bg-red-900/50 text-red-300"
                : "bg-green-900/50 text-green-300"
            }`}
          >
            {detection.is_malicious ? (
              <>
                <AlertTriangle className="w-3 h-3 mr-1" />
                Malicious
              </>
            ) : (
              <>
                <CheckCircle className="w-3 h-3 mr-1" />
                Benign
              </>
            )}
          </div>
        )}
      </div>

      {detection ? (
        <div className="space-y-4">
          <div className="bg-gray-800/80 p-4 rounded-lg border border-gray-700/50">
            <div className="flex items-center mb-2">
              <div className="p-2 bg-blue-900/30 rounded-lg mr-3">
                <Zap className="w-4 h-4 text-blue-400" />
              </div>
              <h3 className="text-white font-medium">Detection Method</h3>
            </div>
            <p className="text-gray-300 text-sm ml-11">
              {detection.detection_method || "Multiple AI Models"}
            </p>
          </div>

          <div className="bg-gray-800/80 p-4 rounded-lg border border-gray-700/50">
            <div className="flex items-center mb-3">
              <div className="p-2 bg-purple-900/30 rounded-lg mr-3">
                <BarChart2 className="w-4 h-4 text-purple-400" />
              </div>
              <h3 className="text-white font-medium">Confidence Score</h3>
            </div>
            <div className="flex items-center ml-11">
              <div className="flex-1 bg-gray-700/70 rounded-full h-3 mr-3">
                <div
                  className={`${getConfidenceColor(
                    detection.confidence || 0
                  )} h-3 rounded-full transition-all duration-500`}
                  style={{ width: `${(detection.confidence || 0) * 100}%` }}
                ></div>
              </div>
              <span
                className={`${getConfidenceTextColor(
                  detection.confidence || 0
                )} text-sm font-bold`}
              >
                {((detection.confidence || 0) * 100).toFixed(1)}%
              </span>
            </div>
          </div>

          <div className="bg-gray-800/80 p-4 rounded-lg border border-gray-700/50">
            <div className="flex items-center mb-2">
              <div className="p-2 bg-green-900/30 rounded-lg mr-3">
                <Code className="w-4 h-4 text-green-400" />
              </div>
              <h3 className="text-white font-medium">Model Information</h3>
            </div>
            <div className="ml-11 grid grid-cols-2 gap-2 text-sm">
              <div className="text-gray-400">Version:</div>
              <div className="text-gray-300 font-mono">
                {detection.model_version || "v1.0.0"}
              </div>

              <div className="text-gray-400">Attack Type:</div>
              <div className="text-gray-300 font-mono">
                {detection.attack_type || "Unknown"}
              </div>

              <div className="text-gray-400">Protocol:</div>
              <div className="text-gray-300 font-mono">
                {detection.protocol || "TCP/IP"}
              </div>
            </div>
          </div>

          {detection.explanation && (
            <div className="bg-gray-800/80 p-4 rounded-lg border border-gray-700/50">
              <div className="flex items-center mb-2">
                <div className="p-2 bg-yellow-900/30 rounded-lg mr-3">
                  <Info className="w-4 h-4 text-yellow-400" />
                </div>
                <h3 className="text-white font-medium">Feature Importance</h3>
              </div>
              <div className="ml-11 mt-3 bg-gray-900/50 p-3 rounded-lg border border-gray-800 overflow-auto max-h-40">
                <pre className="text-gray-300 text-xs font-mono">
                  {JSON.stringify(detection.explanation, null, 2)}
                </pre>
              </div>
            </div>
          )}
        </div>
      ) : (
        <div className="text-center py-10 bg-gray-800/50 rounded-xl border border-gray-700">
          <Brain className="w-16 h-16 mx-auto mb-4 text-purple-500/30" />
          <p className="text-gray-300 font-medium">
            Select a detection to view AI explanation
          </p>
          <p className="text-gray-500 text-sm mt-2">
            Click on any threat in the panel above
          </p>
        </div>
      )}
    </div>
  );
};

export default XAIPanel;
