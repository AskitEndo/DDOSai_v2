import React from "react";
import { NetworkGraph as NetworkGraphType } from "../types";
import { Network, RefreshCw } from "lucide-react";

interface NetworkGraphProps {
  data: NetworkGraphType | null;
  loading?: boolean;
  height?: number;
  onNodeClick?: (node: any) => void;
  onRefresh?: () => void;
}

const NetworkGraph: React.FC<NetworkGraphProps> = ({
  data,
  loading = false,
  height = 400,
  onNodeClick,
  onRefresh,
}) => {
  if (loading) {
    return (
      <div className="bg-gray-800 border border-gray-700 rounded-lg p-6">
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-xl font-semibold text-white flex items-center">
            <Network className="w-5 h-5 mr-2" />
            Network Graph
          </h2>
          <button
            onClick={onRefresh}
            className="p-2 text-gray-400 hover:text-white transition-colors"
            disabled={loading}
          >
            <RefreshCw className={`w-4 h-4 ${loading ? "animate-spin" : ""}`} />
          </button>
        </div>
        <div
          className="bg-gray-700/50 rounded flex items-center justify-center"
          style={{ height: `${height}px` }}
        >
          <div className="text-center">
            <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-500 mx-auto mb-2"></div>
            <p className="text-gray-400">Loading network graph...</p>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="bg-gray-800 border border-gray-700 rounded-lg p-6">
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-xl font-semibold text-white flex items-center">
          <Network className="w-5 h-5 mr-2" />
          Network Graph
        </h2>
        <button
          onClick={onRefresh}
          className="p-2 text-gray-400 hover:text-white transition-colors"
        >
          <RefreshCw className="w-4 h-4" />
        </button>
      </div>

      <div
        className="bg-gray-700/50 rounded relative overflow-hidden"
        style={{ height: `${height}px` }}
      >
        {data && data.nodes.length > 0 ? (
          <div className="p-4 h-full overflow-auto">
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
              {data.nodes.slice(0, 12).map((node, index) => (
                <div
                  key={node.node_id}
                  className={`p-3 rounded cursor-pointer transition-colors ${
                    node.is_malicious
                      ? "bg-red-900/30 border border-red-500/50 hover:bg-red-900/50"
                      : "bg-green-900/30 border border-green-500/50 hover:bg-green-900/50"
                  }`}
                  onClick={() => onNodeClick?.(node)}
                >
                  <div className="text-white font-medium text-sm mb-1">
                    {node.ip_address}
                  </div>
                  <div className="text-xs text-gray-400">
                    Packets: {node.packet_count}
                  </div>
                  <div className="text-xs text-gray-400">
                    Threat: {node.threat_score}/100
                  </div>
                </div>
              ))}
            </div>
            {data.nodes.length > 12 && (
              <div className="text-center mt-4 text-gray-400 text-sm">
                Showing 12 of {data.nodes.length} nodes
              </div>
            )}
          </div>
        ) : (
          <div className="flex items-center justify-center h-full text-gray-400">
            <div className="text-center">
              <Network className="w-12 h-12 mx-auto mb-2 opacity-50" />
              <p>No network data available</p>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default NetworkGraph;
