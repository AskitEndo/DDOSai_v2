import React, { useRef, useEffect, useState } from "react";
import { DetectionResult } from "../types";
import { formatTimestamp } from "../utils";
import {
  Activity,
  ArrowRight,
  Shield,
  AlertTriangle,
  RefreshCw,
} from "lucide-react";

interface LiveTrafficMonitorProps {
  detections: DetectionResult[];
  maxItems?: number;
}

const LiveTrafficMonitor: React.FC<LiveTrafficMonitorProps> = ({
  detections = [],
  maxItems = 50,
}) => {
  const containerRef = useRef<HTMLDivElement>(null);
  const [visibleDetections, setVisibleDetections] = useState<DetectionResult[]>(
    []
  );
  const [isLoading, setIsLoading] = useState(false);

  // Update visible detections when new detections come in
  useEffect(() => {
    // Always set loading state first to show activity
    setIsLoading(true);

    // Handle all cases - null, undefined, empty array, or populated array
    const newDetections = Array.isArray(detections)
      ? [...detections.slice(0, maxItems)]
      : [];

    setVisibleDetections(newDetections);

    // Short timeout to ensure UI updates and shows loading state
    setTimeout(() => {
      setIsLoading(false);
    }, 300);
  }, [detections, maxItems]);

  const handleRefresh = () => {
    setIsLoading(true);
    setTimeout(() => setIsLoading(false), 500);
  };

  return (
    <div className="bg-gray-800/70 border border-gray-700 rounded-xl p-6 shadow-lg hover:shadow-xl transition-all duration-300 hover:border-blue-700/50">
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-xl font-semibold text-white flex items-center">
          <Activity className="w-5 h-5 mr-2 text-blue-400" />
          Live Traffic
        </h2>
        <button
          onClick={handleRefresh}
          className="p-2 rounded-full bg-gray-700/50 hover:bg-gray-700 text-gray-400 hover:text-blue-400 transition-colors"
        >
          <RefreshCw className={`w-4 h-4 ${isLoading ? "animate-spin" : ""}`} />
        </button>
      </div>

      <div
        ref={containerRef}
        className="h-[300px] overflow-y-auto pr-2 scrollbar-thin scrollbar-thumb-gray-700 scrollbar-track-gray-800/30"
      >
        {Array.isArray(visibleDetections) && visibleDetections.length > 0 ? (
          <div className="space-y-2">
            {visibleDetections.map((detection, index) => (
              <div
                key={`${detection.packet_id || index}-${index}`}
                className={`traffic-item flex items-center p-3 rounded-md transition-all duration-300 ${
                  detection.is_malicious
                    ? "bg-red-900/20 border border-red-700/30 backdrop-blur-sm"
                    : "bg-green-900/10 border border-green-700/20"
                }`}
                style={{
                  opacity: 1,
                  transform: "translateY(0)",
                  transition: "all 0.3s ease",
                  animation: index === 0 ? "fadeIn 0.5s ease" : "",
                }}
              >
                <div
                  className={`w-3 h-3 rounded-full ${
                    detection.is_malicious
                      ? "bg-red-500 animate-pulse shadow-[0_0_5px_rgba(239,68,68,0.5)]"
                      : "bg-green-500"
                  }`}
                />
                <div className="ml-3 text-sm text-gray-400">
                  {formatTimestamp(detection.timestamp)}
                </div>
                <div className="ml-3 text-white font-mono">
                  {detection.src_ip}
                </div>
                <ArrowRight className="mx-2 w-3 h-3 text-gray-500" />
                <div className="text-white font-mono">{detection.dst_ip}</div>
                <div className="ml-auto">
                  <span
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
                        <Shield className="w-3 h-3 mr-1" />
                        Benign
                      </>
                    )}
                  </span>
                </div>
              </div>
            ))}
          </div>
        ) : (
          <div className="flex flex-col items-center justify-center h-full">
            <div className="bg-gray-800/80 p-6 rounded-xl border border-gray-700 text-center">
              <Activity className="w-12 h-12 mx-auto mb-4 text-gray-500" />
              <p className="text-gray-400 mb-2">No traffic data available</p>
              <p className="text-gray-500 text-sm">
                Traffic will appear here when detected
              </p>
            </div>
          </div>
        )}
      </div>

      <style>{`
        @keyframes fadeIn {
          from {
            opacity: 0;
            transform: translateY(-10px);
          }
          to {
            opacity: 1;
            transform: translateY(0);
          }
        }
      `}</style>
    </div>
  );
};

export default LiveTrafficMonitor;
