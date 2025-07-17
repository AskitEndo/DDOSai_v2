import React, { useState, useEffect } from "react";
import { useAppContext } from "../context/AppContext";
import { FileText, Search, Filter, Download } from "lucide-react";
import { formatTimestamp } from "../utils";

const Logs: React.FC = () => {
  const { state } = useAppContext();
  const [searchTerm, setSearchTerm] = useState("");
  const [filterType, setFilterType] = useState("all");
  const [filteredLogs, setFilteredLogs] = useState(state.detections);

  useEffect(() => {
    let filtered = state.detections;

    // Filter by type
    if (filterType !== "all") {
      if (filterType === "malicious") {
        filtered = filtered.filter((d) => d.is_malicious);
      } else if (filterType === "benign") {
        filtered = filtered.filter((d) => !d.is_malicious);
      }
    }

    // Filter by search term
    if (searchTerm) {
      filtered = filtered.filter(
        (d) =>
          d.src_ip.includes(searchTerm) ||
          d.dst_ip.includes(searchTerm) ||
          d.attack_type.includes(searchTerm) ||
          d.detection_method.includes(searchTerm)
      );
    }

    setFilteredLogs(filtered);
  }, [state.detections, searchTerm, filterType]);

  const handleExportLogs = () => {
    const dataStr = JSON.stringify(filteredLogs, null, 2);
    const dataUri =
      "data:application/json;charset=utf-8," + encodeURIComponent(dataStr);

    const exportFileDefaultName = `ddos-ai-logs-${
      new Date().toISOString().split("T")[0]
    }.json`;

    const linkElement = document.createElement("a");
    linkElement.setAttribute("href", dataUri);
    linkElement.setAttribute("download", exportFileDefaultName);
    linkElement.click();
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <h1 className="text-3xl font-bold text-white">Detection Logs</h1>
        <button
          onClick={handleExportLogs}
          className="flex items-center px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 transition-colors"
        >
          <Download className="w-4 h-4 mr-2" />
          Export Logs
        </button>
      </div>

      {/* Filters */}
      <div className="bg-gray-800 border border-gray-700 rounded-lg p-4">
        <div className="flex flex-col md:flex-row gap-4">
          <div className="flex-1">
            <div className="relative">
              <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400 w-4 h-4" />
              <input
                type="text"
                placeholder="Search by IP, attack type, or method..."
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
                className="w-full pl-10 pr-4 py-2 bg-gray-700 border border-gray-600 rounded-md text-white placeholder-gray-400"
              />
            </div>
          </div>
          <div className="flex items-center space-x-2">
            <Filter className="w-4 h-4 text-gray-400" />
            <select
              value={filterType}
              onChange={(e) => setFilterType(e.target.value)}
              className="bg-gray-700 border border-gray-600 rounded-md px-3 py-2 text-white"
            >
              <option value="all">All Detections</option>
              <option value="malicious">Malicious Only</option>
              <option value="benign">Benign Only</option>
            </select>
          </div>
        </div>
      </div>

      {/* Logs Table */}
      <div className="bg-gray-800 border border-gray-700 rounded-lg overflow-hidden">
        <div className="overflow-x-auto">
          <table className="w-full">
            <thead className="bg-gray-700">
              <tr>
                <th className="px-4 py-3 text-left text-xs font-medium text-gray-300 uppercase tracking-wider">
                  Timestamp
                </th>
                <th className="px-4 py-3 text-left text-xs font-medium text-gray-300 uppercase tracking-wider">
                  Source IP
                </th>
                <th className="px-4 py-3 text-left text-xs font-medium text-gray-300 uppercase tracking-wider">
                  Destination IP
                </th>
                <th className="px-4 py-3 text-left text-xs font-medium text-gray-300 uppercase tracking-wider">
                  Status
                </th>
                <th className="px-4 py-3 text-left text-xs font-medium text-gray-300 uppercase tracking-wider">
                  Attack Type
                </th>
                <th className="px-4 py-3 text-left text-xs font-medium text-gray-300 uppercase tracking-wider">
                  Threat Score
                </th>
                <th className="px-4 py-3 text-left text-xs font-medium text-gray-300 uppercase tracking-wider">
                  Confidence
                </th>
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-700">
              {filteredLogs.length > 0 ? (
                filteredLogs.slice(0, 100).map((log, index) => (
                  <tr
                    key={`${log.packet_id}-${index}`}
                    className="hover:bg-gray-700/50"
                  >
                    <td className="px-4 py-3 text-sm text-gray-300">
                      {formatTimestamp(log.timestamp)}
                    </td>
                    <td className="px-4 py-3 text-sm text-white font-mono">
                      {log.src_ip}
                    </td>
                    <td className="px-4 py-3 text-sm text-white font-mono">
                      {log.dst_ip}
                    </td>
                    <td className="px-4 py-3 text-sm">
                      <span
                        className={`inline-flex px-2 py-1 text-xs font-semibold rounded-full ${
                          log.is_malicious
                            ? "bg-red-900/30 text-red-400"
                            : "bg-green-900/30 text-green-400"
                        }`}
                      >
                        {log.is_malicious ? "Malicious" : "Benign"}
                      </span>
                    </td>
                    <td className="px-4 py-3 text-sm text-gray-300">
                      {log.attack_type}
                    </td>
                    <td className="px-4 py-3 text-sm">
                      <span
                        className={`font-medium ${
                          log.threat_score >= 80
                            ? "text-red-400"
                            : log.threat_score >= 60
                            ? "text-yellow-400"
                            : "text-green-400"
                        }`}
                      >
                        {log.threat_score}/100
                      </span>
                    </td>
                    <td className="px-4 py-3 text-sm text-gray-300">
                      {(log.confidence * 100).toFixed(1)}%
                    </td>
                  </tr>
                ))
              ) : (
                <tr>
                  <td
                    colSpan={7}
                    className="px-4 py-8 text-center text-gray-400"
                  >
                    <FileText className="w-12 h-12 mx-auto mb-2 opacity-50" />
                    <p>No logs found matching your criteria</p>
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>

        {filteredLogs.length > 100 && (
          <div className="bg-gray-700/50 px-4 py-3 text-center text-sm text-gray-400">
            Showing first 100 of {filteredLogs.length} results
          </div>
        )}
      </div>

      {/* Summary Stats */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <div className="bg-gray-800 border border-gray-700 rounded-lg p-4">
          <div className="text-2xl font-bold text-white">
            {filteredLogs.length}
          </div>
          <div className="text-gray-400 text-sm">Total Detections</div>
        </div>
        <div className="bg-gray-800 border border-gray-700 rounded-lg p-4">
          <div className="text-2xl font-bold text-red-400">
            {filteredLogs.filter((d) => d.is_malicious).length}
          </div>
          <div className="text-gray-400 text-sm">Malicious</div>
        </div>
        <div className="bg-gray-800 border border-gray-700 rounded-lg p-4">
          <div className="text-2xl font-bold text-green-400">
            {filteredLogs.filter((d) => !d.is_malicious).length}
          </div>
          <div className="text-gray-400 text-sm">Benign</div>
        </div>
        <div className="bg-gray-800 border border-gray-700 rounded-lg p-4">
          <div className="text-2xl font-bold text-yellow-400">
            {filteredLogs.length > 0
              ? (
                  filteredLogs.reduce((sum, d) => sum + d.threat_score, 0) /
                  filteredLogs.length
                ).toFixed(1)
              : 0}
          </div>
          <div className="text-gray-400 text-sm">Avg Threat Score</div>
        </div>
      </div>
    </div>
  );
};

export default Logs;
