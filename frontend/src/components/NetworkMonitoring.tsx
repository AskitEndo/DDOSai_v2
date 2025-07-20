import React, { useState, useEffect, useCallback } from "react";
import {
  Monitor,
  Activity,
  AlertTriangle,
  Shield,
  Play,
  Pause,
  RefreshCw,
  Wifi,
  WifiOff,
} from "lucide-react";
import api from "../services/api";
import webSocketService from "../services/websocket";

interface NetworkStats {
  bytes_sent: number;
  bytes_recv: number;
  packets_sent: number;
  packets_recv: number;
  errin: number;
  errout: number;
  dropin: number;
  dropout: number;
}

interface DetectedAttack {
  timestamp: string;
  source_ip: string;
  destination_ip: string;
  attack_type: string;
  severity: string;
  is_malicious: boolean;
  confidence: number;
  protocol: string;
  packet_size: number;
  flags: string[];
}

interface NetworkMonitoringData {
  monitoring_active: boolean;
  detected_attacks: DetectedAttack[];
  network_stats: NetworkStats | null;
  total_monitored_packets: number;
  active_monitoring_duration: string;
  timestamp: string;
  message?: string;
}

interface NetworkMonitoringProps {
  className?: string;
}

const NetworkMonitoring: React.FC<NetworkMonitoringProps> = ({
  className = "",
}) => {
  const [monitoringData, setMonitoringData] =
    useState<NetworkMonitoringData | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [autoRefresh, setAutoRefresh] = useState(false);
  const [refreshInterval, setRefreshInterval] = useState<number | null>(null);

  const fetchMonitoringData = useCallback(async () => {
    if (isLoading) return;

    setIsLoading(true);
    setError(null);

    try {
      const response = await api.getNetworkMonitoringData();
      if (response.status === 200) {
        setMonitoringData(response.data);
      } else {
        setError("Failed to fetch monitoring data");
      }
    } catch (err: any) {
      setError(err.message || "Failed to fetch monitoring data");
    } finally {
      setIsLoading(false);
    }
  }, [isLoading]);

  const startMonitoring = useCallback(async () => {
    setIsLoading(true);
    try {
      const response = await api.startNetworkMonitoring();
      if (response.status === 200) {
        await fetchMonitoringData();
        setAutoRefresh(true);
      } else {
        setError(response.data?.message || "Failed to start monitoring");
      }
    } catch (err: any) {
      setError(err.message || "Failed to start monitoring");
    } finally {
      setIsLoading(false);
    }
  }, [fetchMonitoringData]);

  const stopMonitoring = useCallback(async () => {
    setIsLoading(true);
    try {
      const response = await api.stopNetworkMonitoring();
      if (response.status === 200) {
        await fetchMonitoringData();
        setAutoRefresh(false);
      } else {
        setError(response.data?.message || "Failed to stop monitoring");
      }
    } catch (err: any) {
      setError(err.message || "Failed to stop monitoring");
    } finally {
      setIsLoading(false);
    }
  }, [fetchMonitoringData]);

  const formatBytes = (bytes: number): string => {
    if (bytes === 0) return "0 B";
    const k = 1024;
    const sizes = ["B", "KB", "MB", "GB", "TB"];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + " " + sizes[i];
  };

  const getSeverityColor = (severity: string): string => {
    switch (severity.toLowerCase()) {
      case "critical":
        return "text-red-400 bg-red-900/20";
      case "high":
        return "text-orange-400 bg-orange-900/20";
      case "medium":
        return "text-yellow-400 bg-yellow-900/20";
      case "low":
        return "text-blue-400 bg-blue-900/20";
      default:
        return "text-gray-400 bg-gray-900/20";
    }
  };

  // Auto-refresh effect and WebSocket integration
  useEffect(() => {
    // WebSocket listeners for real-time updates
    const handleNetworkMonitoringUpdate = (data: NetworkMonitoringData) => {
      console.log("Received real-time monitoring update:", data);
      setMonitoringData(data);
      setError(null);
    };

    const handleSimulationAttackDetected = (data: any) => {
      console.log("Received simulation attack detection:", data);
      // Force refresh monitoring data to show new detection
      fetchMonitoringData();
    };

    const handleSimulationStarted = (data: any) => {
      console.log("Simulation started:", data);
      setAutoRefresh(true); // Auto-enable refresh during simulation
      fetchMonitoringData();
    };

    const handleSimulationCompleted = (data: any) => {
      console.log("Simulation completed:", data);
      fetchMonitoringData(); // Final refresh
    };

    // Subscribe to WebSocket events
    webSocketService.on(
      "network_monitoring_update",
      handleNetworkMonitoringUpdate
    );
    webSocketService.on(
      "simulation_attack_detected",
      handleSimulationAttackDetected
    );
    webSocketService.on("simulation_started", handleSimulationStarted);
    webSocketService.on("simulation_completed", handleSimulationCompleted);

    // Regular auto-refresh for non-WebSocket updates
    if (autoRefresh && monitoringData?.monitoring_active) {
      const interval = setInterval(fetchMonitoringData, 5000); // Refresh every 5 seconds
      setRefreshInterval(interval);
      return () => {
        clearInterval(interval);
        // Clean up WebSocket listeners
        webSocketService.off(
          "network_monitoring_update",
          handleNetworkMonitoringUpdate
        );
        webSocketService.off(
          "simulation_attack_detected",
          handleSimulationAttackDetected
        );
        webSocketService.off("simulation_started", handleSimulationStarted);
        webSocketService.off("simulation_completed", handleSimulationCompleted);
      };
    } else if (refreshInterval) {
      clearInterval(refreshInterval);
      setRefreshInterval(null);
    }

    // Clean up WebSocket listeners when component unmounts
    return () => {
      webSocketService.off(
        "network_monitoring_update",
        handleNetworkMonitoringUpdate
      );
      webSocketService.off(
        "simulation_attack_detected",
        handleSimulationAttackDetected
      );
      webSocketService.off("simulation_started", handleSimulationStarted);
      webSocketService.off("simulation_completed", handleSimulationCompleted);
    };
  }, [autoRefresh, monitoringData?.monitoring_active, fetchMonitoringData]);

  // Initial data fetch
  useEffect(() => {
    fetchMonitoringData();
  }, []);

  return (
    <div
      className={`bg-gray-900/50 backdrop-blur-sm rounded-lg border border-gray-700/50 ${className}`}
    >
      {/* Header */}
      <div className="p-4 border-b border-gray-700/50">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-2">
            <Monitor className="w-5 h-5 text-blue-400" />
            <h3 className="text-lg font-semibold text-white">
              Network Monitoring
            </h3>
            {monitoringData?.monitoring_active && (
              <div className="flex items-center text-green-400 text-sm">
                <div className="w-2 h-2 bg-green-400 rounded-full mr-1 animate-pulse"></div>
                Active
              </div>
            )}
          </div>

          <div className="flex items-center space-x-2">
            <button
              onClick={fetchMonitoringData}
              disabled={isLoading}
              className="p-2 text-gray-400 hover:text-white hover:bg-gray-700/50 rounded-md transition-colors"
              title="Refresh data"
            >
              <RefreshCw
                className={`w-4 h-4 ${isLoading ? "animate-spin" : ""}`}
              />
            </button>

            {monitoringData?.monitoring_active ? (
              <button
                onClick={stopMonitoring}
                disabled={isLoading}
                className="flex items-center px-3 py-1.5 bg-red-900/30 hover:bg-red-800/50 text-red-300 rounded-md text-sm transition-colors"
              >
                <Pause className="w-3 h-3 mr-1" />
                Stop
              </button>
            ) : (
              <button
                onClick={startMonitoring}
                disabled={isLoading}
                className="flex items-center px-3 py-1.5 bg-green-900/30 hover:bg-green-800/50 text-green-300 rounded-md text-sm transition-colors"
              >
                <Play className="w-3 h-3 mr-1" />
                Start
              </button>
            )}
          </div>
        </div>
      </div>

      {/* Content */}
      <div className="p-4">
        {error && (
          <div className="mb-4 p-3 bg-red-900/20 border border-red-700/50 rounded-md text-red-300 text-sm">
            {error}
          </div>
        )}

        {!monitoringData ? (
          <div className="text-center py-8 text-gray-400">
            <Monitor className="w-12 h-12 mx-auto mb-2 opacity-50" />
            <p>Loading network monitoring data...</p>
          </div>
        ) : !monitoringData.monitoring_active ? (
          <div className="text-center py-8 text-gray-400">
            <WifiOff className="w-12 h-12 mx-auto mb-2 opacity-50" />
            <p className="mb-2">Network monitoring is not active</p>
            <p className="text-sm">
              Start monitoring to detect real cross-device attacks
            </p>
            {monitoringData.message && (
              <p className="text-xs mt-2 text-gray-500">
                {monitoringData.message}
              </p>
            )}
          </div>
        ) : (
          <div className="space-y-4">
            {/* Network Statistics */}
            {monitoringData.network_stats && (
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                <div className="bg-gray-800/50 p-3 rounded-md">
                  <div className="text-xs text-gray-400 mb-1">
                    Bytes Received
                  </div>
                  <div className="text-sm font-mono text-white">
                    {formatBytes(monitoringData.network_stats.bytes_recv)}
                  </div>
                </div>
                <div className="bg-gray-800/50 p-3 rounded-md">
                  <div className="text-xs text-gray-400 mb-1">Bytes Sent</div>
                  <div className="text-sm font-mono text-white">
                    {formatBytes(monitoringData.network_stats.bytes_sent)}
                  </div>
                </div>
                <div className="bg-gray-800/50 p-3 rounded-md">
                  <div className="text-xs text-gray-400 mb-1">
                    Packets Received
                  </div>
                  <div className="text-sm font-mono text-white">
                    {monitoringData.network_stats.packets_recv.toLocaleString()}
                  </div>
                </div>
                <div className="bg-gray-800/50 p-3 rounded-md">
                  <div className="text-xs text-gray-400 mb-1">Packets Sent</div>
                  <div className="text-sm font-mono text-white">
                    {monitoringData.network_stats.packets_sent.toLocaleString()}
                  </div>
                </div>
              </div>
            )}

            {/* Monitoring Stats */}
            <div className="flex items-center justify-between text-sm text-gray-400">
              <div className="flex items-center space-x-4">
                <span>
                  Total Monitored: {monitoringData.total_monitored_packets}
                </span>
                <span>
                  Duration: {monitoringData.active_monitoring_duration}
                </span>
              </div>
              <span>
                Last Update:{" "}
                {new Date(monitoringData.timestamp).toLocaleTimeString()}
              </span>
            </div>

            {/* Detected Attacks */}
            <div>
              <div className="flex items-center space-x-2 mb-3">
                <AlertTriangle className="w-4 h-4 text-yellow-400" />
                <h4 className="text-sm font-medium text-white">
                  Detected Attacks
                </h4>
                <span className="text-xs text-gray-400">
                  ({monitoringData.detected_attacks.length})
                </span>
              </div>

              {monitoringData.detected_attacks.length === 0 ? (
                <div className="text-center py-6 text-gray-500">
                  <Shield className="w-8 h-8 mx-auto mb-2 opacity-50" />
                  <p className="text-sm">No attacks detected</p>
                  <p className="text-xs">Network appears secure</p>
                </div>
              ) : (
                <div className="space-y-2 max-h-60 overflow-y-auto">
                  {monitoringData.detected_attacks.map((attack, index) => (
                    <div
                      key={index}
                      className="bg-gray-800/30 border border-gray-700/50 rounded-md p-3"
                    >
                      <div className="flex items-center justify-between mb-2">
                        <div className="flex items-center space-x-2">
                          <Activity className="w-3 h-3 text-red-400" />
                          <span className="text-sm font-medium text-white">
                            {attack.attack_type}
                          </span>
                          <span
                            className={`text-xs px-2 py-0.5 rounded ${getSeverityColor(
                              attack.severity
                            )}`}
                          >
                            {attack.severity}
                          </span>
                        </div>
                        <span className="text-xs text-gray-400">
                          {new Date(attack.timestamp).toLocaleTimeString()}
                        </span>
                      </div>

                      <div className="grid grid-cols-2 gap-2 text-xs">
                        <div>
                          <span className="text-gray-400">From:</span>
                          <span className="ml-1 font-mono text-white">
                            {attack.source_ip}
                          </span>
                        </div>
                        <div>
                          <span className="text-gray-400">To:</span>
                          <span className="ml-1 font-mono text-white">
                            {attack.destination_ip}
                          </span>
                        </div>
                        <div>
                          <span className="text-gray-400">Protocol:</span>
                          <span className="ml-1 text-white">
                            {attack.protocol}
                          </span>
                        </div>
                        <div>
                          <span className="text-gray-400">Confidence:</span>
                          <span className="ml-1 text-white">
                            {(attack.confidence * 100).toFixed(1)}%
                          </span>
                        </div>
                      </div>

                      {attack.flags.length > 0 && (
                        <div className="mt-2">
                          <span className="text-xs text-gray-400">Flags:</span>
                          <div className="flex flex-wrap gap-1 mt-1">
                            {attack.flags.map((flag, i) => (
                              <span
                                key={i}
                                className="text-xs px-1.5 py-0.5 bg-gray-700/50 text-gray-300 rounded"
                              >
                                {flag}
                              </span>
                            ))}
                          </div>
                        </div>
                      )}
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default NetworkMonitoring;
