import React, { useState, useEffect } from "react";
import { SimulationConfig, SimulationStatus } from "../types";
import { Play, Square, Settings, AlertTriangle } from "lucide-react";

interface AttackSimulationPanelProps {
  config: SimulationConfig;
  onConfigChange: (config: SimulationConfig) => void;
  onStart: () => void;
  onStop: () => void;
  isRunning: boolean;
  disabled?: boolean;
}

const AttackSimulationPanel: React.FC<AttackSimulationPanelProps> = ({
  config,
  onConfigChange,
  onStart,
  onStop,
  isRunning,
  disabled = false,
}) => {
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [attackTypeConfig, setAttackTypeConfig] = useState<Record<string, any>>(
    {}
  );

  // Update attack-specific configuration options when attack type changes
  useEffect(() => {
    switch (config.attack_type) {
      case "syn_flood":
        setAttackTypeConfig({
          packet_size: config.packet_size || 64,
        });
        break;
      case "udp_flood":
        setAttackTypeConfig({
          packet_size: config.packet_size || 1024,
        });
        break;
      case "http_flood":
        setAttackTypeConfig({
          num_threads: config.num_threads || 10,
          num_connections: config.num_connections || 100,
          use_https: config.use_https || false,
        });
        break;
      case "slowloris":
        setAttackTypeConfig({
          num_connections: config.num_connections || 500,
          connection_rate: config.connection_rate || 50,
        });
        break;
      default:
        setAttackTypeConfig({});
    }
  }, [config.attack_type]);

  // Handle form input changes
  const handleInputChange = (
    e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement>
  ) => {
    const { name, value, type } = e.target as HTMLInputElement;

    // Handle checkbox inputs
    if (type === "checkbox") {
      const checked = (e.target as HTMLInputElement).checked;
      onConfigChange({
        ...config,
        [name]: checked,
      });
      return;
    }

    // Handle number inputs
    if (type === "number") {
      onConfigChange({
        ...config,
        [name]: parseInt(value, 10),
      });
      return;
    }

    // Handle other inputs
    onConfigChange({
      ...config,
      [name]: value,
    });
  };

  return (
    <div className="bg-dark-surface border border-dark-border rounded-lg p-6">
      <h2 className="text-xl font-semibold text-white mb-4">
        Attack Configuration
      </h2>

      <div className="space-y-4">
        {/* Basic Configuration */}
        <div>
          <label className="block text-sm font-medium text-gray-400 mb-1">
            Attack Type
          </label>
          <select
            name="attack_type"
            value={config.attack_type}
            onChange={handleInputChange}
            className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded-md text-white focus:outline-none focus:ring-2 focus:ring-primary"
            disabled={isRunning || disabled}
          >
            <option value="syn_flood">SYN Flood</option>
            <option value="udp_flood">UDP Flood</option>
            <option value="http_flood">HTTP Flood</option>
            <option value="slowloris">Slowloris</option>
          </select>
        </div>

        <div>
          <label className="block text-sm font-medium text-gray-400 mb-1">
            Target IP
          </label>
          <input
            type="text"
            name="target_ip"
            value={config.target_ip}
            onChange={handleInputChange}
            className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded-md text-white focus:outline-none focus:ring-2 focus:ring-primary"
            disabled={isRunning || disabled}
          />
        </div>

        <div>
          <label className="block text-sm font-medium text-gray-400 mb-1">
            Target Port
          </label>
          <input
            type="number"
            name="target_port"
            value={config.target_port}
            onChange={handleInputChange}
            min={1}
            max={65535}
            className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded-md text-white focus:outline-none focus:ring-2 focus:ring-primary"
            disabled={isRunning || disabled}
          />
        </div>

        <div>
          <label className="block text-sm font-medium text-gray-400 mb-1">
            Duration (seconds)
          </label>
          <input
            type="number"
            name="duration"
            value={config.duration}
            onChange={handleInputChange}
            min={5}
            max={300}
            className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded-md text-white focus:outline-none focus:ring-2 focus:ring-primary"
            disabled={isRunning || disabled}
          />
        </div>

        <div>
          <label className="block text-sm font-medium text-gray-400 mb-1">
            Packet Rate (packets/sec)
          </label>
          <input
            type="number"
            name="packet_rate"
            value={config.packet_rate}
            onChange={handleInputChange}
            min={100}
            max={10000}
            step={100}
            className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded-md text-white focus:outline-none focus:ring-2 focus:ring-primary"
            disabled={isRunning || disabled}
          />
        </div>

        {/* Advanced Options Toggle */}
        <div className="pt-2">
          <button
            type="button"
            onClick={() => setShowAdvanced(!showAdvanced)}
            className="flex items-center text-sm text-gray-400 hover:text-white transition-colors"
            disabled={isRunning || disabled}
          >
            <Settings className="w-4 h-4 mr-1" />
            {showAdvanced ? "Hide Advanced Options" : "Show Advanced Options"}
          </button>
        </div>

        {/* Attack-specific Configuration */}
        {showAdvanced && (
          <div className="pt-2 space-y-4 border-t border-gray-700">
            <h3 className="text-md font-medium text-white mt-2">
              Advanced Options
            </h3>

            {/* SYN Flood and UDP Flood Options */}
            {(config.attack_type === "syn_flood" ||
              config.attack_type === "udp_flood") && (
              <div>
                <label className="block text-sm font-medium text-gray-400 mb-1">
                  Packet Size (bytes)
                </label>
                <input
                  type="number"
                  name="packet_size"
                  value={config.packet_size || 64}
                  onChange={handleInputChange}
                  min={32}
                  max={1500}
                  className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded-md text-white focus:outline-none focus:ring-2 focus:ring-primary"
                  disabled={isRunning || disabled}
                />
              </div>
            )}

            {/* HTTP Flood Options */}
            {config.attack_type === "http_flood" && (
              <>
                <div>
                  <label className="block text-sm font-medium text-gray-400 mb-1">
                    Number of Threads
                  </label>
                  <input
                    type="number"
                    name="num_threads"
                    value={config.num_threads || 10}
                    onChange={handleInputChange}
                    min={1}
                    max={100}
                    className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded-md text-white focus:outline-none focus:ring-2 focus:ring-primary"
                    disabled={isRunning || disabled}
                  />
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-400 mb-1">
                    Connections per Thread
                  </label>
                  <input
                    type="number"
                    name="num_connections"
                    value={config.num_connections || 100}
                    onChange={handleInputChange}
                    min={10}
                    max={1000}
                    className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded-md text-white focus:outline-none focus:ring-2 focus:ring-primary"
                    disabled={isRunning || disabled}
                  />
                </div>

                <div className="flex items-center">
                  <input
                    type="checkbox"
                    id="use_https"
                    name="use_https"
                    checked={config.use_https || false}
                    onChange={handleInputChange}
                    className="w-4 h-4 bg-gray-700 border border-gray-600 rounded focus:ring-primary"
                    disabled={isRunning || disabled}
                  />
                  <label
                    htmlFor="use_https"
                    className="ml-2 text-sm text-gray-400"
                  >
                    Use HTTPS
                  </label>
                </div>
              </>
            )}

            {/* Slowloris Options */}
            {config.attack_type === "slowloris" && (
              <>
                <div>
                  <label className="block text-sm font-medium text-gray-400 mb-1">
                    Number of Connections
                  </label>
                  <input
                    type="number"
                    name="num_connections"
                    value={config.num_connections || 500}
                    onChange={handleInputChange}
                    min={100}
                    max={2000}
                    className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded-md text-white focus:outline-none focus:ring-2 focus:ring-primary"
                    disabled={isRunning || disabled}
                  />
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-400 mb-1">
                    Connection Rate (connections/sec)
                  </label>
                  <input
                    type="number"
                    name="connection_rate"
                    value={config.connection_rate || 50}
                    onChange={handleInputChange}
                    min={10}
                    max={200}
                    className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded-md text-white focus:outline-none focus:ring-2 focus:ring-primary"
                    disabled={isRunning || disabled}
                  />
                </div>
              </>
            )}
          </div>
        )}

        {/* Warning */}
        <div className="bg-warning/10 border border-warning/30 rounded-lg p-3 text-warning flex items-start mt-4">
          <AlertTriangle className="w-4 h-4 mr-2 mt-0.5 flex-shrink-0" />
          <div className="text-sm">
            Only use against systems you own or have explicit permission to
            test.
          </div>
        </div>

        {/* Action Button */}
        <div className="pt-4">
          {isRunning ? (
            <button
              onClick={onStop}
              className="w-full px-4 py-3 bg-danger/20 text-danger border border-danger/30 rounded-md hover:bg-danger/30 transition-colors flex items-center justify-center"
              disabled={disabled}
            >
              <Square className="w-4 h-4 mr-2" />
              Stop Simulation
            </button>
          ) : (
            <button
              onClick={onStart}
              className="w-full px-4 py-3 bg-primary/20 text-primary border border-primary/30 rounded-md hover:bg-primary/30 transition-colors flex items-center justify-center"
              disabled={disabled}
            >
              <Play className="w-4 h-4 mr-2" />
              Start Simulation
            </button>
          )}
        </div>
      </div>
    </div>
  );
};

export default AttackSimulationPanel;
