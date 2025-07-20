import React, { useState, useEffect } from "react";
import { useSimulation } from "../context/SimulationContext";
import webSocketService from "../services/websocket";
import {
  Play,
  Square,
  Wifi,
  WifiOff,
  AlertCircle,
  Clock,
  Activity,
  Target,
  Users,
  Zap,
  Shield,
  Globe,
  RefreshCw,
} from "lucide-react";

interface AttackType {
  id: string;
  name: string;
  description: string;
  icon: React.ReactNode;
  severity: "low" | "medium" | "high" | "critical";
}

const attackTypes: AttackType[] = [
  {
    id: "syn_flood",
    name: "SYN Flood",
    description: "TCP SYN flood attack overwhelming the target",
    icon: <Zap className="w-4 h-4" />,
    severity: "high",
  },
  {
    id: "udp_flood",
    name: "UDP Flood",
    description: "UDP packet flood targeting random ports",
    icon: <Activity className="w-4 h-4" />,
    severity: "medium",
  },
  {
    id: "http_flood",
    name: "HTTP Flood",
    description: "HTTP GET/POST request flood",
    icon: <Globe className="w-4 h-4" />,
    severity: "high",
  },
  {
    id: "ping_flood",
    name: "Ping Flood",
    description: "ICMP ping flood attack",
    icon: <Target className="w-4 h-4" />,
    severity: "low",
  },
  {
    id: "slowloris",
    name: "Slowloris",
    description: "Slow HTTP connection attack",
    icon: <Clock className="w-4 h-4" />,
    severity: "critical",
  },
];

const getSeverityColor = (severity: string) => {
  switch (severity) {
    case "low":
      return "text-green-400 bg-green-900/20 border-green-700";
    case "medium":
      return "text-yellow-400 bg-yellow-900/20 border-yellow-700";
    case "high":
      return "text-orange-400 bg-orange-900/20 border-orange-700";
    case "critical":
      return "text-red-400 bg-red-900/20 border-red-700";
    default:
      return "text-gray-400 bg-gray-900/20 border-gray-700";
  }
};

const Simulation: React.FC = () => {
  const {
    state,
    startSimulation,
    stopSimulation,
    forceStopSimulation,
    getUserIP,
    refreshSimulationStatus,
  } = useSimulation();

  const [simulationConfig, setSimulationConfig] = useState({
    target_ip: "",
    attack_type: "syn_flood",
    target_port: 80,
    duration: 30,
    packet_rate: 1000,
    packet_size: 64,
    num_threads: 10,
  });

  const [isStarting, setIsStarting] = useState(false);
  const [isStopping, setIsStopping] = useState(false);
  const [connectionStatus, setConnectionStatus] = useState<
    "checking" | "connected" | "disconnected"
  >("checking");
  const [isDetectingIP, setIsDetectingIP] = useState(false);
  const [simulationMessage, setSimulationMessage] = useState<string>("");

  // Auto-detect user IP on component mount
  useEffect(() => {
    const detectIP = async () => {
      setIsDetectingIP(true);
      try {
        const ip = await getUserIP();
        if (ip && !simulationConfig.target_ip) {
          setSimulationConfig((prev) => ({ ...prev, target_ip: ip }));
        }
      } catch (error) {
        console.error("Failed to detect IP:", error);
      } finally {
        setIsDetectingIP(false);
      }
    };

    detectIP();
  }, [getUserIP, simulationConfig.target_ip]);

  // Check connection status and refresh simulation state
  useEffect(() => {
    const checkConnection = async () => {
      try {
        // Import API service to check connection
        const { api } = await import("../services/api");
        const response = await api.health();
        setConnectionStatus(
          response.status === 200 ? "connected" : "disconnected"
        );

        // If connected, refresh simulation status to sync with backend
        if (response.status === 200) {
          await refreshSimulationStatus();
        }
      } catch (error) {
        setConnectionStatus("disconnected");
        // If disconnected, we can't trust simulation state
        console.log("Backend disconnected, simulation state may be stale");
      }
    };

    checkConnection();
    const interval = setInterval(checkConnection, 30000); // Check every 30s
    return () => clearInterval(interval);
  }, [refreshSimulationStatus]);

  // WebSocket listeners for real-time simulation feedback
  useEffect(() => {
    const handleSimulationStarted = (data: any) => {
      console.log("Real-time simulation started:", data);
      // Update simulation state to show it's really running
      setSimulationMessage(
        `REAL ${data.attack_type} simulation started against ${data.target_ip}:${data.target_port}. Attack packets are being generated at ${data.packet_rate} packets/second.`
      );
    };

    const handleSimulationAttackDetected = (data: any) => {
      console.log("Real-time attack detected:", data);
      // Show live attack progress
      setSimulationMessage(
        `LIVE ATTACK: ${data.attack_type} packet sent to ${data.target}. ${
          data.packets_sent
        } packets sent so far. Threat detected with ${Math.round(
          data.detection.confidence * 100
        )}% confidence.`
      );
    };

    const handleSimulationCompleted = (data: any) => {
      console.log("Real-time simulation completed:", data);
      setSimulationMessage(
        `Simulation completed! Total ${data.packets_sent} attack packets were sent to ${data.target}. The attack was successfully detected by the AI system.`
      );
    };

    const handleNetworkMonitoringUpdate = (data: any) => {
      if (data.simulation_active) {
        console.log("Network monitoring update during simulation:", data);
        setSimulationMessage(
          `Real-time monitoring: ${
            data.detected_attacks?.length || 0
          } attacks detected. ${
            data.total_monitored_packets || 0
          } packets analyzed.`
        );
      }
    };

    // Subscribe to WebSocket events
    webSocketService.on("simulation_started", handleSimulationStarted);
    webSocketService.on(
      "simulation_attack_detected",
      handleSimulationAttackDetected
    );
    webSocketService.on("simulation_completed", handleSimulationCompleted);
    webSocketService.on(
      "network_monitoring_update",
      handleNetworkMonitoringUpdate
    );

    // Clean up listeners on component unmount
    return () => {
      webSocketService.off("simulation_started", handleSimulationStarted);
      webSocketService.off(
        "simulation_attack_detected",
        handleSimulationAttackDetected
      );
      webSocketService.off("simulation_completed", handleSimulationCompleted);
      webSocketService.off(
        "network_monitoring_update",
        handleNetworkMonitoringUpdate
      );
    };
  }, []);

  const handleConfigChange = (field: string, value: string | number) => {
    setSimulationConfig((prev) => ({
      ...prev,
      [field]: value,
    }));
  };

  const handleStartSimulation = async () => {
    setIsStarting(true);
    try {
      await startSimulation(simulationConfig);
    } catch (error) {
      console.error("Failed to start simulation:", error);
    } finally {
      setIsStarting(false);
    }
  };

  const handleStopSimulation = async () => {
    setIsStopping(true);
    try {
      const success = await stopSimulation();
      if (!success) {
        console.log("Normal stop failed, trying force stop");
        await forceStopSimulation();
      }
    } catch (error) {
      console.error("Failed to stop simulation:", error);
      // Force stop as fallback
      await forceStopSimulation();
    } finally {
      setIsStopping(false);
    }
  };

  const handleForceStop = async () => {
    setIsStopping(true);
    try {
      await forceStopSimulation();
      setSimulationMessage("Simulation forcefully stopped and state cleared.");
    } catch (error) {
      console.error("Failed to force stop simulation:", error);
    } finally {
      setIsStopping(false);
    }
  };

  const selectedAttackType = attackTypes.find(
    (attack) => attack.id === simulationConfig.attack_type
  );

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="bg-gray-800 border border-gray-700 rounded-lg p-6">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-bold text-white">DDoS Simulation</h1>
            <p className="text-gray-400 mt-1">
              Configure and run controlled DDoS attack simulations
            </p>
          </div>
          <div className="flex items-center space-x-2">
            {connectionStatus === "connected" ? (
              <div className="flex items-center text-green-400">
                <Wifi className="w-4 h-4 mr-1" />
                <span className="text-sm">Connected</span>
              </div>
            ) : connectionStatus === "disconnected" ? (
              <div className="flex items-center text-red-400">
                <WifiOff className="w-4 h-4 mr-1" />
                <span className="text-sm">Disconnected</span>
              </div>
            ) : (
              <div className="flex items-center text-yellow-400">
                <AlertCircle className="w-4 h-4 mr-1" />
                <span className="text-sm">Checking...</span>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Current Simulation Status */}
      {state.currentSimulation && (
        <div className="bg-blue-900/20 border border-blue-700 rounded-lg p-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-3">
              <div className="w-3 h-3 bg-blue-400 rounded-full animate-pulse"></div>
              <div>
                <h3 className="text-blue-300 font-medium">
                  Simulation Running
                </h3>
                <p className="text-blue-400 text-sm">
                  {selectedAttackType?.name} →{" "}
                  {state.currentSimulation.target_ip}
                </p>
              </div>
            </div>
            <div className="text-blue-300 text-sm">
              Duration: {state.currentSimulation.duration}s
            </div>
          </div>
        </div>
      )}

      {/* Real-time Simulation Messages */}
      {simulationMessage && (
        <div className="bg-green-900/20 border border-green-700 rounded-lg p-4">
          <div className="flex items-start space-x-3">
            <Activity className="w-5 h-5 text-green-400 mt-0.5 flex-shrink-0 animate-pulse" />
            <div>
              <h3 className="text-green-300 font-medium">Real-time Update</h3>
              <p className="text-green-400 text-sm mt-1">{simulationMessage}</p>
            </div>
          </div>
        </div>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Simulation Configuration */}
        <div className="bg-gray-800 border border-gray-700 rounded-lg p-6">
          <h2 className="text-xl font-bold text-white mb-4">Configuration</h2>

          <div className="space-y-4">
            {/* Target IP */}
            <div>
              <label className="block text-sm font-medium text-gray-300 mb-2">
                Target IP Address
              </label>
              <div className="flex space-x-2">
                <input
                  type="text"
                  value={simulationConfig.target_ip}
                  onChange={(e) =>
                    handleConfigChange("target_ip", e.target.value)
                  }
                  className="flex-1 px-3 py-2 bg-gray-700 border border-gray-600 rounded-md text-white focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  placeholder="192.168.1.1"
                  disabled={!!state.currentSimulation}
                />
                <button
                  onClick={async () => {
                    setIsDetectingIP(true);
                    try {
                      const ip = await getUserIP();
                      setSimulationConfig((prev) => ({
                        ...prev,
                        target_ip: ip,
                      }));
                    } catch (error) {
                      console.error("Failed to detect IP:", error);
                    } finally {
                      setIsDetectingIP(false);
                    }
                  }}
                  disabled={!!state.currentSimulation || isDetectingIP}
                  className="px-3 py-2 bg-purple-600 hover:bg-purple-700 disabled:opacity-50 disabled:cursor-not-allowed text-white rounded-md transition-colors text-sm"
                  title="Detect your public IP address"
                >
                  {isDetectingIP ? (
                    <RefreshCw className="w-4 h-4 animate-spin" />
                  ) : (
                    <Globe className="w-4 h-4" />
                  )}
                </button>
              </div>
              <p className="text-xs text-gray-400 mt-1">
                💡 Your IP will be auto-detected, or click the globe button to
                refresh
                {state.userIP && (
                  <span className="block text-purple-400 mt-1">
                    🔍 Detected IP: {state.userIP}
                  </span>
                )}
              </p>
            </div>

            {/* Target Port */}
            <div>
              <label className="block text-sm font-medium text-gray-300 mb-2">
                Target Port
              </label>
              <input
                type="number"
                min="1"
                max="65535"
                value={simulationConfig.target_port}
                onChange={(e) =>
                  handleConfigChange("target_port", parseInt(e.target.value))
                }
                className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded-md text-white focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                placeholder="80"
                disabled={!!state.currentSimulation}
              />
            </div>

            {/* Attack Type */}
            <div>
              <label className="block text-sm font-medium text-gray-300 mb-2">
                Attack Type
              </label>
              <select
                value={simulationConfig.attack_type}
                onChange={(e) =>
                  handleConfigChange("attack_type", e.target.value)
                }
                className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded-md text-white focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                disabled={!!state.currentSimulation}
              >
                {attackTypes.map((attack) => (
                  <option key={attack.id} value={attack.id}>
                    {attack.name}
                  </option>
                ))}
              </select>

              {/* Attack Type Info */}
              {selectedAttackType && (
                <div
                  className={`mt-2 p-3 rounded-md border ${getSeverityColor(
                    selectedAttackType.severity
                  )}`}
                >
                  <div className="flex items-center space-x-2">
                    {selectedAttackType.icon}
                    <span className="font-medium">
                      {selectedAttackType.name}
                    </span>
                    <span className="text-xs px-2 py-1 rounded uppercase font-bold">
                      {selectedAttackType.severity}
                    </span>
                  </div>
                  <p className="text-sm mt-1 opacity-80">
                    {selectedAttackType.description}
                  </p>
                </div>
              )}
            </div>

            {/* Duration */}
            <div>
              <label className="block text-sm font-medium text-gray-300 mb-2">
                Duration (seconds)
              </label>
              <input
                type="number"
                min="1"
                max="300"
                value={simulationConfig.duration}
                onChange={(e) =>
                  handleConfigChange("duration", parseInt(e.target.value))
                }
                className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded-md text-white focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                disabled={!!state.currentSimulation}
              />
            </div>

            {/* Packet Rate */}
            <div>
              <label className="block text-sm font-medium text-gray-300 mb-2">
                Packet Rate (packets/sec): {simulationConfig.packet_rate}
              </label>
              <input
                type="range"
                min="100"
                max="10000"
                value={simulationConfig.packet_rate}
                onChange={(e) =>
                  handleConfigChange("packet_rate", parseInt(e.target.value))
                }
                className="w-full h-2 bg-gray-700 rounded-lg appearance-none cursor-pointer"
                disabled={!!state.currentSimulation}
              />
            </div>

            {/* Threads */}
            <div>
              <label className="block text-sm font-medium text-gray-300 mb-2">
                Threads
              </label>
              <input
                type="number"
                min="1"
                max="100"
                value={simulationConfig.num_threads}
                onChange={(e) =>
                  handleConfigChange("num_threads", parseInt(e.target.value))
                }
                className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded-md text-white focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                disabled={!!state.currentSimulation}
              />
            </div>
          </div>

          {/* Control Buttons */}
          <div className="mt-6 flex flex-col space-y-3">
            {!state.currentSimulation || !state.isRunning ? (
              <button
                onClick={handleStartSimulation}
                disabled={
                  isStarting ||
                  !simulationConfig.target_ip ||
                  connectionStatus !== "connected"
                }
                className="flex items-center justify-center px-4 py-2 bg-red-600 hover:bg-red-700 disabled:opacity-50 disabled:cursor-not-allowed text-white rounded-md transition-colors"
              >
                {isStarting ? (
                  <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin mr-2"></div>
                ) : (
                  <Play className="w-4 h-4 mr-2" />
                )}
                {isStarting ? "Starting..." : "Start Simulation"}
              </button>
            ) : (
              <div className="space-y-2">
                <button
                  onClick={handleStopSimulation}
                  disabled={isStopping}
                  className="w-full flex items-center justify-center px-4 py-2 bg-orange-600 hover:bg-orange-700 disabled:opacity-50 disabled:cursor-not-allowed text-white rounded-md transition-colors"
                >
                  {isStopping ? (
                    <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin mr-2"></div>
                  ) : (
                    <Square className="w-4 h-4 mr-2" />
                  )}
                  {isStopping ? "Stopping..." : "Stop Simulation"}
                </button>
                <button
                  onClick={handleForceStop}
                  disabled={isStopping}
                  className="w-full flex items-center justify-center px-4 py-2 bg-red-700 hover:bg-red-800 disabled:opacity-50 disabled:cursor-not-allowed text-white rounded-md transition-colors text-sm"
                  title="Force stop if normal stop doesn't work"
                >
                  <AlertCircle className="w-4 h-4 mr-2" />
                  Force Stop & Clear State
                </button>
              </div>
            )}

            {/* Connection Warning */}
            {connectionStatus !== "connected" &&
              (state.currentSimulation || state.isRunning) && (
                <div className="mt-2 p-2 bg-yellow-900/20 border border-yellow-700 rounded-md">
                  <div className="flex items-center space-x-2 text-yellow-300 text-xs">
                    <AlertCircle className="w-4 h-4 flex-shrink-0" />
                    <span>
                      Backend disconnected. Simulation state may be outdated.
                      Use "Force Stop" to clear.
                    </span>
                  </div>
                </div>
              )}
          </div>

          {/* Warning */}
          <div className="mt-4 p-3 bg-yellow-900/20 border border-yellow-700 rounded-md">
            <div className="flex items-start space-x-2">
              <AlertCircle className="w-5 h-5 text-yellow-400 mt-0.5 flex-shrink-0" />
              <div className="text-yellow-300 text-sm">
                <p className="font-medium">⚠️ Educational Use Only</p>
                <p className="mt-1">
                  Only use this simulation against systems you own or have
                  explicit permission to test. Unauthorized DDoS attacks are
                  illegal.
                </p>
              </div>
            </div>
          </div>
        </div>

        {/* Simulation History */}
        <div className="bg-gray-800 border border-gray-700 rounded-lg p-6">
          <h2 className="text-xl font-bold text-white mb-4">
            Simulation History
          </h2>

          {state.history.length === 0 ? (
            <div className="text-center py-8 text-gray-400">
              <Shield className="w-12 h-12 mx-auto mb-3 opacity-50" />
              <p>No simulations run yet</p>
              <p className="text-sm mt-1">
                Start your first simulation to see results here
              </p>
            </div>
          ) : (
            <div className="space-y-3">
              {state.history.slice(0, 10).map((sim, index) => {
                const attackType = attackTypes.find(
                  (a) => a.id === sim.attack_type
                );
                return (
                  <div
                    key={index}
                    className="bg-gray-700/50 border border-gray-600 rounded-md p-3"
                  >
                    <div className="flex items-center justify-between">
                      <div className="flex items-center space-x-3">
                        {attackType?.icon}
                        <div>
                          <p className="text-white font-medium">
                            {attackType?.name || sim.attack_type}
                          </p>
                          <p className="text-gray-400 text-sm">
                            Target: {sim.target_ip}
                          </p>
                        </div>
                      </div>
                      <div className="text-right">
                        <p className="text-gray-300 text-sm">{sim.duration}s</p>
                        <p className="text-gray-400 text-xs">
                          {sim.start_time
                            ? new Date(sim.start_time).toLocaleTimeString()
                            : "N/A"}
                        </p>
                      </div>
                    </div>
                  </div>
                );
              })}
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default Simulation;
