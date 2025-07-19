import React from "react";
import { DetectionResult } from "../types";
import { Shield, AlertTriangle, Zap, Eye, ChevronRight } from "lucide-react";
import { formatTimestamp } from "../utils";

interface ThreatScorePanelProps {
  detections: DetectionResult[];
  loading?: boolean;
  onSelectDetection?: (detection: DetectionResult) => void;
}

const ThreatScorePanel: React.FC<ThreatScorePanelProps> = ({
  detections = [],
  loading = false,
  onSelectDetection,
}) => {
  const maliciousDetections = Array.isArray(detections)
    ? detections.filter((d) => d.is_malicious).slice(0, 10)
    : [];

  const getThreatColor = (score: number) => {
    if (score >= 80) return "text-red-400";
    if (score >= 60) return "text-orange-400";
    if (score >= 40) return "text-yellow-400";
    return "text-blue-400";
  };

  const getThreatBgColor = (score: number) => {
    if (score >= 80) return "bg-red-900/30 border-red-700/40";
    if (score >= 60) return "bg-orange-900/30 border-orange-700/40";
    if (score >= 40) return "bg-yellow-900/30 border-yellow-700/40";
    return "bg-blue-900/30 border-blue-700/40";
  };

  const getAttackIcon = (attackType: string) => {
    switch (attackType?.toLowerCase()) {
      case "syn_flood":
        return <Zap className="w-3 h-3 mr-1" />;
      case "udp_flood":
        return <Zap className="w-3 h-3 mr-1" />;
      case "http_flood":
        return <Zap className="w-3 h-3 mr-1" />;
      default:
        return <AlertTriangle className="w-3 h-3 mr-1" />;
    }
  };

  return (
    <div className="bg-gray-800/70 border border-gray-700 rounded-xl p-6 shadow-lg hover:shadow-xl transition-all duration-300 hover:border-blue-700/50">
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-xl font-semibold text-white flex items-center">
          <Shield className="w-5 h-5 mr-2 text-red-400" />
          Threat Detection
        </h2>
        <div className="text-xs bg-gray-700 text-gray-300 px-2 py-1 rounded-full">
          {maliciousDetections.length}{" "}
          {maliciousDetections.length === 1 ? "threat" : "threats"} detected
        </div>
      </div>

      {loading ? (
        <div className="animate-pulse space-y-3">
          {[...Array(5)].map((_, i) => (
            <div key={i} className="bg-gray-700 h-16 rounded-lg"></div>
          ))}
        </div>
      ) : (
        <div className="space-y-3 max-h-80 overflow-y-auto pr-1">
          {maliciousDetections.length > 0 ? (
            maliciousDetections.map((detection, index) => (
              <div
                key={`${detection.packet_id || index}-${index}`}
                className={`${getThreatBgColor(
                  detection.threat_score || 0
                )} border rounded-lg p-4 cursor-pointer hover:bg-opacity-50 transition-all duration-200 transform hover:-translate-y-1`}
                onClick={() => onSelectDetection?.(detection)}
              >
                <div className="flex items-center justify-between">
                  <div className="flex items-center">
                    <div className="w-2 h-2 bg-red-500 rounded-full animate-pulse mr-2"></div>
                    <span className="text-white font-medium font-mono">
                      {detection.src_ip}
                      <ChevronRight className="w-3 h-3 inline mx-1 text-gray-500" />
                      {detection.dst_ip}
                    </span>
                  </div>
                  <div className="text-right">
                    <div
                      className={`${getThreatColor(
                        detection.threat_score || 0
                      )} font-bold text-lg`}
                    >
                      {detection.threat_score || 0}/100
                    </div>
                  </div>
                </div>

                <div className="flex items-center justify-between mt-3">
                  <div className="flex items-center">
                    <span className="bg-gray-800/70 text-xs px-2 py-1 rounded-full flex items-center">
                      {getAttackIcon(detection.attack_type || "")}
                      {detection.attack_type || "Unknown Attack"}
                    </span>
                  </div>
                  <div className="flex items-center">
                    <div className="text-xs text-gray-400 mr-2">
                      {formatTimestamp(detection.timestamp)}
                    </div>
                    <button
                      className="p-1 bg-gray-700/50 hover:bg-gray-700 rounded-full text-gray-400 hover:text-blue-400 transition-colors"
                      onClick={(e) => {
                        e.stopPropagation();
                        onSelectDetection?.(detection);
                      }}
                    >
                      <Eye className="w-3 h-3" />
                    </button>
                  </div>
                </div>
              </div>
            ))
          ) : (
            <div className="text-center py-10 bg-gray-800/50 rounded-xl border border-gray-700">
              <Shield className="w-16 h-16 mx-auto mb-4 text-green-500/50" />
              <p className="text-gray-300 font-medium">No threats detected</p>
              <p className="text-gray-500 text-sm mt-2">
                Your network is currently secure
              </p>
            </div>
          )}
        </div>
      )}
    </div>
  );
};

export default ThreatScorePanel;
