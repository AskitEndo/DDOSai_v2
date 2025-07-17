import React, { useState } from "react";
import { useAppContext } from "../context/AppContext";
import { Zap, Play, Square, Settings, Target } from "lucide-react";
import api from "../services/api";

const Simulation: React.FC = () => {
  const { state } = useAppContext();
  const [isRunning, setIsRunning] = useState(false);
  const [simulationConfig, setSimulationConfig] = useState({
    attack_type: "syn_flood",
    target_ip: "192.168.1.100",
    target_port: 80,
    duration: 60,
    packet_rate: 100,
    packet_size: 64,
  });

  const handleStartSimulation = async () => {
    try {
      setIsRunning(true);
      const response = await api.startSimulation(simulationConfig);
      console.log("Simulation started:", response);
    } catch (error) {
      console.error("Failed to start simulation:", error);
      setIsRunning(false);
    }
  };

  const handleStopSimulation = async () => {
    try {
      setIsRunning(false);
      // In a real implementation, we'd pass the simulation ID
      const response = await api.stopSimulation("current");
      console.log("Simulation stopped:", response);
    } catch (error) {
      console.error("Failed to stop simulation:", error);
    }
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <h1 className="text-3xl font-bold text-white">Attack Simulation</h1>
        <div className="flex items-center space-x-2 text-sm">
          <div
            className={`w-2 h-2 rounded-full ${
              isRunning ? "bg-red-500 animate-pulse" : "bg-gray-500"
            }`}
          />
          <span className="text-gray-400">
            Status: {isRunning ? "Running" : "Idle"}
          </span>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Configuration Panel */}
        <div className="bg-gray-800 border border-gray-700 rounded-lg p-6">
          <h2 className="text-xl font-semibold text-white mb-4 flex items-center">
            <Settings className="w-5 h-5 mr-2" />
            Simulation Configuration
          </h2>

          <div className="space-y-4">
            <div>
              <label className="block text-gray-300 text-sm font-medium mb-2">
                Attack Type
              </label>
              <select
                value={simulationConfig.attack_type}
                onChange={(e) =>
                  setSimulationConfig({
                    ...simulationConfig,
                    attack_type: e.target.value,
                  })
                }
                className="w-full bg-gray-700 border border-gray-600 rounded-md px-3 py-2 text-white"
                disabled={isRunning}
              >
                <option value="syn_flood">SYN Flood</option>
                <option value="udp_flood">UDP Flood</option>
                <option value="http_flood">HTTP Flood</option>
                <option value="slowloris">Slowloris</option>
              </select>
            </div>

            <div>
              <label className="block text-gray-300 text-sm font-medium mb-2">
                Target IP
              </label>
              <input
                type="text"
                value={simulationConfig.target_ip}
                onChange={(e) =>
                  setSimulationConfig({
                    ...simulationConfig,
                    target_ip: e.target.value,
                  })
                }
                className="w-full bg-gray-700 border border-gray-600 rounded-md px-3 py-2 text-white"
                disabled={isRunning}
              />
            </div>

            <div className="grid grid-cols-2 gap-4">
              <div>
                <label className="block text-gray-300 text-sm font-medium mb-2">
                  Target Port
                </label>
                <input
                  type="number"
                  value={simulationConfig.target_port}
                  onChange={(e) =>
                    setSimulationConfig({
                      ...simulationConfig,
                      target_port: parseInt(e.target.value),
                    })
                  }
                  className="w-full bg-gray-700 border border-gray-600 rounded-md px-3 py-2 text-white"
                  disabled={isRunning}
                />
              </div>
              <div>
                <label className="block text-gray-300 text-sm font-medium mb-2">
                  Duration (seconds)
                </label>
                <input
                  type="number"
                  value={simulationConfig.duration}
                  onChange={(e) =>
                    setSimulationConfig({
                      ...simulationConfig,
                      duration: parseInt(e.target.value),
                    })
                  }
                  className="w-full bg-gray-700 border border-gray-600 rounded-md px-3 py-2 text-white"
                  disabled={isRunning}
                />
              </div>
            </div>

            <div className="grid grid-cols-2 gap-4">
              <div>
                <label className="block text-gray-300 text-sm font-medium mb-2">
                  Packet Rate (pps)
                </label>
                <input
                  type="number"
                  value={simulationConfig.packet_rate}
                  onChange={(e) =>
                    setSimulationConfig({
                      ...simulationConfig,
                      packet_rate: parseInt(e.target.value),
                    })
                  }
                  className="w-full bg-gray-700 border border-gray-600 rounded-md px-3 py-2 text-white"
                  disabled={isRunning}
                />
              </div>
              <div>
                <label className="block text-gray-300 text-sm font-medium mb-2">
                  Packet Size (bytes)
                </label>
                <input
                  type="number"
                  value={simulationConfig.packet_size}
                  onChange={(e) =>
                    setSimulationConfig({
                      ...simulationConfig,
                      packet_size: parseInt(e.target.value),
                    })
                  }
                  className="w-full bg-gray-700 border border-gray-600 rounded-md px-3 py-2 text-white"
                  disabled={isRunning}
                />
              </div>
            </div>
          </div>

          <div className="mt-6 flex space-x-4">
            <button
              onClick={handleStartSimulation}
              disabled={isRunning}
              className="flex-1 flex items-center justify-center px-4 py-2 bg-red-600 text-white rounded-md hover:bg-red-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
            >
              <Play className="w-4 h-4 mr-2" />
              Start Attack
            </button>
            <button
              onClick={handleStopSimulation}
              disabled={!isRunning}
              className="flex-1 flex items-center justify-center px-4 py-2 bg-gray-600 text-white rounded-md hover:bg-gray-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
            >
              <Square className="w-4 h-4 mr-2" />
              Stop Attack
            </button>
          </div>
        </div>

        {/* Status Panel */}
        <div className="bg-gray-800 border border-gray-700 rounded-lg p-6">
          <h2 className="text-xl font-semibold text-white mb-4 flex items-center">
            <Target className="w-5 h-5 mr-2" />
            Simulation Status
          </h2>

          {state.simulation ? (
            <div className="space-y-4">
              <div className="grid grid-cols-2 gap-4">
                <div className="bg-gray-700/50 rounded p-3">
                  <p className="text-gray-400 text-sm">Status</p>
                  <p className="text-white font-medium">
                    {state.simulation.status}
                  </p>
                </div>
                <div className="bg-gray-700/50 rounded p-3">
                  <p className="text-gray-400 text-sm">Attack Type</p>
                  <p className="text-white font-medium">
                    {state.simulation.attack_type || "N/A"}
                  </p>
                </div>
                <div className="bg-gray-700/50 rounded p-3">
                  <p className="text-gray-400 text-sm">Packets Sent</p>
                  <p className="text-white font-medium">
                    {state.simulation.packets_sent}
                  </p>
                </div>
                <div className="bg-gray-700/50 rounded p-3">
                  <p className="text-gray-400 text-sm">Bytes Sent</p>
                  <p className="text-white font-medium">
                    {state.simulation.bytes_sent}
                  </p>
                </div>
              </div>

              {state.simulation.current_packet_rate && (
                <div className="bg-gray-700/50 rounded p-3">
                  <p className="text-gray-400 text-sm">Current Packet Rate</p>
                  <p className="text-white font-medium">
                    {state.simulation.current_packet_rate} pps
                  </p>
                </div>
              )}
            </div>
          ) : (
            <div className="text-center text-gray-400 py-8">
              <Zap className="w-12 h-12 mx-auto mb-2 opacity-50" />
              <p>No active simulation</p>
              <p className="text-sm mt-1">
                Configure and start an attack simulation
              </p>
            </div>
          )}
        </div>
      </div>

      {/* Warning Notice */}
      <div className="bg-yellow-900/20 border border-yellow-500/30 rounded-lg p-4">
        <div className="flex items-start">
          <div className="flex-shrink-0">
            <svg
              className="w-5 h-5 text-yellow-400"
              fill="currentColor"
              viewBox="0 0 20 20"
            >
              <path
                fillRule="evenodd"
                d="M8.257 3.099c.765-1.36 2.722-1.36 3.486 0l5.58 9.92c.75 1.334-.213 2.98-1.742 2.98H4.42c-1.53 0-2.493-1.646-1.743-2.98l5.58-9.92zM11 13a1 1 0 11-2 0 1 1 0 012 0zm-1-8a1 1 0 00-1 1v3a1 1 0 002 0V6a1 1 0 00-1-1z"
                clipRule="evenodd"
              />
            </svg>
          </div>
          <div className="ml-3">
            <h3 className="text-sm font-medium text-yellow-400">Warning</h3>
            <div className="mt-2 text-sm text-yellow-300">
              <p>
                This simulation tool is for educational and testing purposes
                only. Only use against systems you own or have explicit
                permission to test. Unauthorized use may be illegal and could
                result in serious consequences.
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Simulation;
