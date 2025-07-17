import React, { useRef, useEffect, useState } from "react";
import { DetectionResult } from "../types";
import { formatTimestamp } from "../utils";
import { Activity, ArrowRight } from "lucide-react";

interface LiveTrafficMonitorProps {
  detections: DetectionResult[];
  maxItems?: number;
}

const LiveTrafficMonitor: React.FC<LiveTrafficMonitorProps> = ({
  detections,
  maxItems = 50,
}) => {
  const containerRef = useRef<HTMLDivElement>(null);
  const [visibleDetections, setVisibleDetections] = useState<DetectionResult[]>(
    []
  );

  // Update visible detections when new detections come in
  useEffect(() => {
    if (detections.length > 0) {
      setVisibleDetections((prev) => {
        // Add new detections to the beginning
        const newDetections = [...detections.slice(0, maxItems)];

        // Animate out old detections
        setTimeout(() => {
          if (containerRef.current) {
            const items =
              containerRef.current.querySelectorAll(".traffic-item");
            items.forEach((item, index) => {
              if (index >= maxItems) {
                (item as HTMLElement).style.opacity = "0";
                (item as HTMLElement).style.transform = "translateY(10px)";
              }
            });
          }
        }, 100);

        return newDetections;
      });
    }
  }, [detections, maxItems]);

  return (
    <div className="bg-dark-surface border border-dark-border rounded-lg p-6">
      <h2 className="text-xl font-semibold text-white mb-4 flex items-center">
        <Activity className="w-5 h-5 mr-2" />
        Live Traffic
      </h2>

      <div ref={containerRef} className="h-[300px] overflow-y-auto pr-2">
        {visibleDetections.length > 0 ? (
          <div className="space-y-2">
            {visibleDetections.map((detection, index) => (
              <div
                key={`${detection.packet_id}-${index}`}
                className={`traffic-item flex items-center p-2 rounded-md transition-all duration-300 ${
                  detection.is_malicious
                    ? "bg-danger/10 border border-danger/30"
                    : "bg-gray-800/50"
                }`}
                style={{
                  opacity: 1,
                  transform: "translateY(0)",
                  transition: "opacity 0.3s ease, transform 0.3s ease",
                }}
              >
                <div
                  className={`w-2 h-2 rounded-full ${
                    detection.is_malicious ? "bg-danger" : "bg-success"
                  }`}
                />
                <div className="ml-3 text-sm text-gray-400">
                  {formatTimestamp(detection.timestamp)}
                </div>
                <div className="ml-3 text-white">{detection.src_ip}</div>
                <ArrowRight className="mx-2 w-3 h-3 text-gray-500" />
                <div className="text-white">{detection.dst_ip}</div>
                <div className="ml-auto text-sm">
                  <span
                    className={`${
                      detection.is_malicious ? "text-danger" : "text-success"
                    }`}
                  >
                    {detection.is_malicious ? "Malicious" : "Benign"}
                  </span>
                </div>
              </div>
            ))}
          </div>
        ) : (
          <div className="flex items-center justify-center h-full text-gray-400">
            <p>No traffic data available</p>
          </div>
        )}
      </div>
    </div>
  );
};

export default LiveTrafficMonitor;
