import React from "react";
import { DetectionResult } from "../types";
import { X, Shield, Clock, Target } from "lucide-react";
import { formatTimestamp } from "../utils";

interface DetectionDetailsProps {
  detection: DetectionResult;
  onClose: () => void;
}

const DetectionDetails: React.FC<DetectionDetailsProps> = ({
  detection,
  onClose,
}) => {
  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4">
      <div className="bg-gray-800 border border-gray-700 rounded-lg max-w-2xl w-full max-h-[90vh] overflow-auto">
        <div className="flex items-center justify-between p-6 border-b border-gray-700">
          <h2 className="text-xl font-semibold text-white flex items-center">
            <Shield className="w-5 h-5 mr-2" />
            Detection Details
          </h2>
          <button
            onClick={onClose}
            className="text-gray-400 hover:text-white transition-colors"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        <div className="p-6 space-y-6">
          {/* Basic Information */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="bg-gray-700/50 rounded p-4">
              <h3 className="text-white font-medium mb-2 flex items-center">
                <Target className="w-4 h-4 mr-2" />
                Source
              </h3>
              <p className="text-gray-300">{detection.src_ip}</p>
              {detection.src_port && (
                <p className="text-gray-400 text-sm">
                  Port: {detection.src_port}
                </p>
              )}
            </div>

            <div className="bg-gray-700/50 rounded p-4">
              <h3 className="text-white font-medium mb-2 flex items-center">
                <Target className="w-4 h-4 mr-2" />
                Destination
              </h3>
              <p className="text-gray-300">{detection.dst_ip}</p>
              {detection.dst_port && (
                <p className="text-gray-400 text-sm">
                  Port: {detection.dst_port}
                </p>
              )}
            </div>
          </div>

          {/* Detection Information */}
          <div className="bg-gray-700/50 rounded p-4">
            <h3 className="text-white font-medium mb-3 flex items-center">
              <Clock className="w-4 h-4 mr-2" />
              Detection Information
            </h3>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div>
                <p className="text-gray-400 text-sm">Timestamp</p>
                <p className="text-white">
                  {formatTimestamp(detection.timestamp)}
                </p>
              </div>
              <div>
                <p className="text-gray-400 text-sm">Attack Type</p>
                <p className="text-white">{detection.attack_type}</p>
              </div>
              <div>
                <p className="text-gray-400 text-sm">Threat Score</p>
                <p className="text-white">{detection.threat_score}/100</p>
              </div>
              <div>
                <p className="text-gray-400 text-sm">Confidence</p>
                <p className="text-white">
                  {(detection.confidence * 100).toFixed(1)}%
                </p>
              </div>
              <div>
                <p className="text-gray-400 text-sm">Detection Method</p>
                <p className="text-white">{detection.detection_method}</p>
              </div>
              <div>
                <p className="text-gray-400 text-sm">Model Version</p>
                <p className="text-white">{detection.model_version}</p>
              </div>
            </div>
          </div>

          {/* Protocol Information */}
          {detection.protocol && (
            <div className="bg-gray-700/50 rounded p-4">
              <h3 className="text-white font-medium mb-2">
                Protocol Information
              </h3>
              <p className="text-gray-300">{detection.protocol}</p>
            </div>
          )}

          {/* Explanation */}
          {detection.explanation && (
            <div className="bg-gray-700/50 rounded p-4">
              <h3 className="text-white font-medium mb-2">Technical Details</h3>
              <pre className="text-gray-300 text-sm overflow-auto bg-gray-800 p-3 rounded">
                {JSON.stringify(detection.explanation, null, 2)}
              </pre>
            </div>
          )}
        </div>

        <div className="flex justify-end p-6 border-t border-gray-700">
          <button
            onClick={onClose}
            className="px-4 py-2 bg-gray-600 text-white rounded hover:bg-gray-700 transition-colors"
          >
            Close
          </button>
        </div>
      </div>
    </div>
  );
};

export default DetectionDetails;
