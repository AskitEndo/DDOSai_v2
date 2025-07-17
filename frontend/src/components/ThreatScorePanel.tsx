import React from "react";
import { DetectionResult } from "../types";
import { Shield, AlertTriangle } from "lucide-react";
import { formatTimestamp } from "../utils";

interface ThreatScorePanelProps {
  detections: DetectionResult[];
  loading?: boolean;
  onSelectDetection?: (detection: DetectionResult) => void;
}

const ThreatScorePanel: React.FC<ThreatScorePanelProps> = ({
  detections,
  loading = false,
  onSelectDetection,
}) => {
  const maliciousDetections = detections
    .filter((d) => d.is_malicious)
    .slice(0, 10);

  return (
    <div className="bg-gray-800 border border-gray-700 rounded-lg p-6">
      <h2 className="text-xl font-semibold text-white mb-4 flex items-center">
        <Shield className="w-5 h-5 mr-2" />
        Threat Detection
      </h2>

      {loading ? (
        <div className="animate-pulse space-y-3">
          {[...Array(5)].map((_, i) => (
            <div key={i} className="bg-gray-700 h-12 rounded"></div>
          ))}
        </div>
      ) : (
        <div className="space-y-2 max-h-80 overflow-y-auto">
          {maliciousDetections.length > 0 ? (
            maliciousDetections.map((detection, index) => (
              <div
                key={`${detection.packet_id}-${index}`}
                className="bg-red-900/20 border border-red-500/30 rounded-md p-3 cursor-pointer hover:bg-red-900/30 transition-colors"
                onClick={() => onSelectDetection?.(detection)}
              >
                <div className="flex items-center justify-between">
                  <div className="flex items-center">
                    <AlertTriangle className="w-4 h-4 text-red-400 mr-2" />
                    <span className="text-white font-medium">
                      {detection.src_ip} → {detection.dst_ip}
                    </span>
                  </div>
                  <div className="text-right">
                    <div className="text-red-400 font-bold">
                      {detection.threat_score}/100
                    </div>
                    <div className="text-xs text-gray-400">
                      {formatTimestamp(detection.timestamp)}
                    </div>
                  </div>
                </div>
                <div className="mt-2 text-sm text-gray-300">
                  Attack Type: {detection.attack_type}
                </div>
              </div>
            ))
          ) : (
            <div className="text-center text-gray-400 py-8">
              <Shield className="w-12 h-12 mx-auto mb-2 opacity-50" />
              <p>No threats detected</p>
            </div>
          )}
        </div>
      )}
    </div>
  );
};

export default ThreatScorePanel;
