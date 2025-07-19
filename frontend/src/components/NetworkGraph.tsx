import React, { useState } from "react";
import { NetworkGraph as NetworkGraphType } from "../types";
import {
  Network,
  RefreshCw,
  Search,
  AlertTriangle,
  Shield,
  Wifi,
} from "lucide-react";

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
  const [searchTerm, setSearchTerm] = useState("");
  const [filterType, setFilterType] = useState<"all" | "malicious" | "benign">(
    "all"
  );

  const getFilteredNodes = () => {
    if (!data || !Array.isArray(data.nodes)) return [];

    return data.nodes
      .filter((node) => {
        // Apply search filter
        if (searchTerm && !node.ip_address.includes(searchTerm)) {
          return false;
        }

        // Apply type filter
        if (filterType === "malicious" && !node.is_malicious) {
          return false;
        }
        if (filterType === "benign" && node.is_malicious) {
          return false;
        }

        return true;
      })
      .slice(0, 12);
  };

  const filteredNodes = data ? getFilteredNodes() : [];
  const totalNodes = data?.nodes?.length || 0;
  const maliciousCount =
    data?.nodes?.filter((n) => n.is_malicious)?.length || 0;

  if (loading) {
    return (
      <div className="bg-gray-800/70 border border-gray-700 rounded-xl p-6 shadow-lg">
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-xl font-semibold text-white flex items-center">
            <Network className="w-5 h-5 mr-2 text-cyan-400" />
            Network Graph
          </h2>
          <button
            onClick={onRefresh}
            className="p-2 rounded-full bg-gray-700/50 text-gray-400 hover:text-cyan-400 transition-colors"
            disabled={loading}
          >
            <RefreshCw className={`w-4 h-4 ${loading ? "animate-spin" : ""}`} />
          </button>
        </div>
        <div
          className="bg-gray-800/80 border border-gray-700 rounded-xl flex items-center justify-center"
          style={{ height: `${height}px` }}
        >
          <div className="text-center">
            <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-cyan-500 mx-auto mb-4"></div>
            <p className="text-gray-300 font-medium">
              Loading network graph...
            </p>
            <p className="text-gray-500 text-sm mt-2">
              Analyzing network connections
            </p>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="bg-gray-800/70 border border-gray-700 rounded-xl p-6 shadow-lg hover:shadow-xl transition-all duration-300 hover:border-cyan-700/50">
      <div className="flex flex-col md:flex-row md:items-center justify-between mb-4 space-y-3 md:space-y-0">
        <h2 className="text-xl font-semibold text-white flex items-center">
          <Network className="w-5 h-5 mr-2 text-cyan-400" />
          Network Graph
        </h2>

        <div className="flex items-center space-x-2">
          <div className="relative">
            <Search className="w-4 h-4 absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-500" />
            <input
              type="text"
              placeholder="Search IP..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              className="pl-9 pr-4 py-1.5 bg-gray-700/50 border border-gray-600 rounded-lg text-sm text-gray-300 focus:outline-none focus:ring-1 focus:ring-cyan-500 focus:border-cyan-500"
            />
          </div>

          <div className="flex rounded-lg overflow-hidden border border-gray-600">
            <button
              className={`px-3 py-1.5 text-xs font-medium ${
                filterType === "all"
                  ? "bg-gray-700 text-white"
                  : "bg-gray-800/50 text-gray-400 hover:bg-gray-700/30"
              }`}
              onClick={() => setFilterType("all")}
            >
              All
            </button>
            <button
              className={`px-3 py-1.5 text-xs font-medium flex items-center ${
                filterType === "malicious"
                  ? "bg-gray-700 text-red-400"
                  : "bg-gray-800/50 text-gray-400 hover:bg-gray-700/30"
              }`}
              onClick={() => setFilterType("malicious")}
            >
              <AlertTriangle className="w-3 h-3 mr-1" />
              Malicious
            </button>
            <button
              className={`px-3 py-1.5 text-xs font-medium flex items-center ${
                filterType === "benign"
                  ? "bg-gray-700 text-green-400"
                  : "bg-gray-800/50 text-gray-400 hover:bg-gray-700/30"
              }`}
              onClick={() => setFilterType("benign")}
            >
              <Shield className="w-3 h-3 mr-1" />
              Benign
            </button>
          </div>

          <button
            onClick={onRefresh}
            className="p-2 rounded-full bg-gray-700/50 hover:bg-gray-700 text-gray-400 hover:text-cyan-400 transition-colors"
          >
            <RefreshCw className="w-4 h-4" />
          </button>
        </div>
      </div>

      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center space-x-4">
          <div className="flex items-center text-xs">
            <div className="w-3 h-3 rounded-full bg-cyan-500 mr-1.5"></div>
            <span className="text-gray-300">{totalNodes} Nodes</span>
          </div>
          <div className="flex items-center text-xs">
            <div className="w-3 h-3 rounded-full bg-red-500 mr-1.5"></div>
            <span className="text-gray-300">{maliciousCount} Malicious</span>
          </div>
          <div className="flex items-center text-xs">
            <div className="w-3 h-3 rounded-full bg-green-500 mr-1.5"></div>
            <span className="text-gray-300">
              {totalNodes - maliciousCount} Benign
            </span>
          </div>
        </div>
      </div>

      <div
        className="bg-gray-800/80 border border-gray-700 rounded-xl relative overflow-hidden"
        style={{ height: `${height}px` }}
      >
        {data && Array.isArray(data.nodes) && data.nodes.length > 0 ? (
          <div className="p-4 h-full overflow-auto">
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
              {filteredNodes.map((node, index) => (
                <div
                  key={node.node_id || index}
                  className={`p-4 rounded-lg cursor-pointer transition-all duration-200 transform hover:-translate-y-1 hover:shadow-lg ${
                    node.is_malicious
                      ? "bg-gradient-to-br from-red-900/30 to-red-900/10 border border-red-700/30"
                      : "bg-gradient-to-br from-green-900/30 to-green-900/10 border border-green-700/30"
                  }`}
                  onClick={() => onNodeClick?.(node)}
                >
                  <div className="flex items-center justify-between mb-2">
                    <div className="flex items-center">
                      <div
                        className={`p-1.5 rounded-md ${
                          node.is_malicious
                            ? "bg-red-900/50"
                            : "bg-green-900/50"
                        } mr-2`}
                      >
                        <Wifi
                          className={`w-4 h-4 ${
                            node.is_malicious
                              ? "text-red-400"
                              : "text-green-400"
                          }`}
                        />
                      </div>
                      <div className="text-white font-medium text-sm font-mono">
                        {node.ip_address}
                      </div>
                    </div>
                    <div
                      className={`w-2 h-2 rounded-full ${
                        node.is_malicious
                          ? "bg-red-500 animate-pulse"
                          : "bg-green-500"
                      }`}
                    ></div>
                  </div>

                  <div className="mt-3 grid grid-cols-2 gap-2 text-xs">
                    <div className="text-gray-400">Packets:</div>
                    <div className="text-gray-300 text-right">
                      {node.packet_count}
                    </div>

                    <div className="text-gray-400">Connections:</div>
                    <div className="text-gray-300 text-right">
                      {(node as any).connections || 0}
                    </div>

                    <div className="text-gray-400">Threat Score:</div>
                    <div
                      className={`text-right font-medium ${
                        node.is_malicious ? "text-red-400" : "text-green-400"
                      }`}
                    >
                      {node.threat_score}/100
                    </div>
                  </div>
                </div>
              ))}
            </div>

            {totalNodes > 12 && (
              <div className="text-center mt-6 text-gray-400 text-sm bg-gray-800/50 py-2 px-4 rounded-lg border border-gray-700 inline-block mx-auto">
                Showing {filteredNodes.length} of {totalNodes} nodes
              </div>
            )}
          </div>
        ) : (
          <div className="flex items-center justify-center h-full">
            <div className="text-center bg-gray-800/80 p-8 rounded-xl border border-gray-700">
              <Network className="w-16 h-16 mx-auto mb-4 text-gray-600" />
              <p className="text-gray-300 font-medium">
                No network data available
              </p>
              <p className="text-gray-500 text-sm mt-2">
                Network connections will appear here when detected
              </p>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default NetworkGraph;
