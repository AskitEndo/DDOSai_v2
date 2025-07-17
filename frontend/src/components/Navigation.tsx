import React from "react";
import { Link, useLocation } from "react-router-dom";
import { Shield, Activity, Zap, FileText, Settings } from "lucide-react";
import { useAppContext } from "../context/AppContext";

const Navigation: React.FC = () => {
  const location = useLocation();
  const { state } = useAppContext();

  const navItems = [
    { path: "/", label: "Dashboard", icon: Activity },
    { path: "/simulation", label: "Simulation", icon: Zap },
    { path: "/logs", label: "Logs", icon: FileText },
  ];

  return (
    <nav className="bg-dark-surface border-b border-dark-border shadow-lg">
      <div className="container mx-auto px-4">
        <div className="flex items-center justify-between h-16">
          <div className="flex items-center space-x-8">
            <div className="flex items-center space-x-2">
              <Shield className="w-8 h-8 text-primary" />
              <div className="text-xl font-bold text-primary cyber-glow">
                DDoS.AI
              </div>
            </div>
            <div className="flex space-x-4">
              {navItems.map((item) => {
                const Icon = item.icon;
                const isActive = location.pathname === item.path;

                return (
                  <Link
                    key={item.path}
                    to={item.path}
                    className={`flex items-center px-3 py-2 rounded-md text-sm font-medium transition-all duration-200 ${
                      isActive
                        ? "bg-primary/20 text-primary border border-primary/30 shadow-glow"
                        : "text-gray-300 hover:bg-gray-700/50 hover:text-white hover:border hover:border-gray-600"
                    }`}
                  >
                    <Icon className="w-4 h-4 mr-2" />
                    {item.label}
                  </Link>
                );
              })}
            </div>
          </div>
          <div className="flex items-center space-x-4">
            <div className="flex items-center space-x-2 text-sm">
              <div
                className={`w-2 h-2 rounded-full ${
                  state.isConnected ? "bg-success animate-pulse" : "bg-danger"
                }`}
              />
              <span className="text-gray-400">
                Status:{" "}
                <span
                  className={state.isConnected ? "text-success" : "text-danger"}
                >
                  {state.isConnected ? "Connected" : "Disconnected"}
                </span>
              </span>
            </div>
            {state.metrics && (
              <div className="text-sm text-gray-400">
                Threat Level:{" "}
                <span
                  className={`font-medium ${
                    state.metrics.threat_level >= 4
                      ? "text-danger"
                      : state.metrics.threat_level >= 2
                      ? "text-warning"
                      : "text-success"
                  }`}
                >
                  {state.metrics.threat_level}/5
                </span>
              </div>
            )}
            <button className="p-2 text-gray-400 hover:text-white hover:bg-gray-700/50 rounded-md transition-colors">
              <Settings className="w-4 h-4" />
            </button>
          </div>
        </div>
      </div>
    </nav>
  );
};

export default Navigation;
